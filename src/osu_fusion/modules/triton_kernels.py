# ruff: noqa: ANN001
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


def _reduce_broadcast_grad(
    grad: torch.Tensor,
    full_shape: torch.Size,
    target_shape: torch.Size,
) -> torch.Tensor:
    if full_shape == target_shape:
        return grad

    ndim_full = len(full_shape)
    ndim_target = len(target_shape)
    offset = ndim_full - ndim_target  # leading dims that don't exist in target

    reduce_dims: list[int] = []
    for i in range(ndim_full):
        j = i - offset  # corresponding index in target_shape
        if j < 0 or target_shape[j] == 1:
            reduce_dims.append(i)

    if reduce_dims:
        grad = grad.sum(dim=reduce_dims, keepdim=True)
    return grad.reshape(target_shape)


# ============================================================================
# fused_modulate: x * (1 + scale) + shift
# ============================================================================


@triton.jit
def _fused_modulate_fwd_kernel(
    X_ptr,
    Shift_ptr,
    Scale_ptr,
    Out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row: int = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X_ptr + row * N + offsets, mask=mask, other=0.0)
    shift = tl.load(Shift_ptr + row * N + offsets, mask=mask, other=0.0)
    scale = tl.load(Scale_ptr + row * N + offsets, mask=mask, other=0.0)

    out = x * (1.0 + scale) + shift
    tl.store(Out_ptr + row * N + offsets, out, mask=mask)


@triton.jit
def _fused_modulate_bwd_kernel(
    Grad_out_ptr,
    X_ptr,
    Scale_ptr,
    Grad_x_ptr,
    Grad_shift_ptr,
    Grad_scale_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row: int = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    grad_out = tl.load(Grad_out_ptr + row * N + offsets, mask=mask, other=0.0)
    x = tl.load(X_ptr + row * N + offsets, mask=mask, other=0.0)
    scale = tl.load(Scale_ptr + row * N + offsets, mask=mask, other=0.0)

    # d_out/d_x = (1 + scale)
    grad_x = grad_out * (1.0 + scale)
    # d_out/d_shift = 1
    grad_shift = grad_out
    # d_out/d_scale = x
    grad_scale = grad_out * x

    tl.store(Grad_x_ptr + row * N + offsets, grad_x, mask=mask)
    tl.store(Grad_shift_ptr + row * N + offsets, grad_shift, mask=mask)
    tl.store(Grad_scale_ptr + row * N + offsets, grad_scale, mask=mask)


class _FusedModulate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape: torch.Size = x.shape
        x_flat: torch.Tensor = x.contiguous().reshape(-1, x.shape[-1])
        shift_flat: torch.Tensor = shift.expand_as(x).contiguous().reshape(-1, x.shape[-1])
        scale_flat: torch.Tensor = scale.expand_as(x).contiguous().reshape(-1, x.shape[-1])

        M: int = x_flat.shape[0]
        N: int = x_flat.shape[1]
        out: torch.Tensor = torch.empty_like(x_flat)
        BLOCK_SIZE: int = triton.next_power_of_2(N)

        _fused_modulate_fwd_kernel[(M,)](
            x_flat,
            shift_flat,
            scale_flat,
            out,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x_flat, scale_flat)
        ctx.orig_shape = orig_shape
        ctx.shift_shape = shift.shape
        ctx.scale_shape = scale.shape
        return out.reshape(orig_shape)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flat: torch.Tensor
        scale_flat: torch.Tensor
        x_flat, scale_flat = ctx.saved_tensors
        grad_out_flat: torch.Tensor = grad_output.reshape(-1, x_flat.shape[1]).contiguous()

        M: int = x_flat.shape[0]
        N: int = x_flat.shape[1]
        grad_x: torch.Tensor = torch.empty_like(x_flat)
        grad_shift: torch.Tensor = torch.empty_like(x_flat)
        grad_scale: torch.Tensor = torch.empty_like(x_flat)
        BLOCK_SIZE: int = triton.next_power_of_2(N)

        _fused_modulate_bwd_kernel[(M,)](
            grad_out_flat,
            x_flat,
            scale_flat,
            grad_x,
            grad_shift,
            grad_scale,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        grad_x = grad_x.reshape(ctx.orig_shape)
        grad_shift = grad_shift.reshape(ctx.orig_shape)
        grad_scale = grad_scale.reshape(ctx.orig_shape)

        # Reduce gradients if shift/scale were broadcast
        grad_shift = _reduce_broadcast_grad(grad_shift, ctx.orig_shape, ctx.shift_shape)
        grad_scale = _reduce_broadcast_grad(grad_scale, ctx.orig_shape, ctx.scale_shape)

        return grad_x, grad_shift, grad_scale


def fused_modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if x.is_cuda:
        return _FusedModulate.apply(x, shift, scale)
    return x * (1.0 + scale) + shift


# ============================================================================
# fused_gated_residual: x + gate * y
# ============================================================================


@triton.jit
def _fused_gated_residual_fwd_kernel(
    X_ptr,
    Gate_ptr,
    Y_ptr,
    Out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row: int = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X_ptr + row * N + offsets, mask=mask, other=0.0)
    gate = tl.load(Gate_ptr + row * N + offsets, mask=mask, other=0.0)
    y = tl.load(Y_ptr + row * N + offsets, mask=mask, other=0.0)

    out = x + gate * y
    tl.store(Out_ptr + row * N + offsets, out, mask=mask)


@triton.jit
def _fused_gated_residual_bwd_kernel(
    Grad_out_ptr,
    Gate_ptr,
    Y_ptr,
    Grad_x_ptr,
    Grad_gate_ptr,
    Grad_y_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row: int = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    grad_out = tl.load(Grad_out_ptr + row * N + offsets, mask=mask, other=0.0)
    gate = tl.load(Gate_ptr + row * N + offsets, mask=mask, other=0.0)
    y = tl.load(Y_ptr + row * N + offsets, mask=mask, other=0.0)

    # d_out/d_x = 1
    grad_x = grad_out
    # d_out/d_gate = y
    grad_gate = grad_out * y
    # d_out/d_y = gate
    grad_y = grad_out * gate

    tl.store(Grad_x_ptr + row * N + offsets, grad_x, mask=mask)
    tl.store(Grad_gate_ptr + row * N + offsets, grad_gate, mask=mask)
    tl.store(Grad_y_ptr + row * N + offsets, grad_y, mask=mask)


class _FusedGatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        gate: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape: torch.Size = x.shape
        x_flat: torch.Tensor = x.contiguous().reshape(-1, x.shape[-1])
        gate_flat: torch.Tensor = gate.expand_as(x).contiguous().reshape(-1, x.shape[-1])
        y_flat: torch.Tensor = y.contiguous().reshape(-1, y.shape[-1])

        M: int = x_flat.shape[0]
        N: int = x_flat.shape[1]
        out: torch.Tensor = torch.empty_like(x_flat)
        BLOCK_SIZE: int = triton.next_power_of_2(N)

        _fused_gated_residual_fwd_kernel[(M,)](
            x_flat,
            gate_flat,
            y_flat,
            out,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(gate_flat, y_flat)
        ctx.orig_shape = orig_shape
        ctx.gate_shape = gate.shape
        return out.reshape(orig_shape)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_flat: torch.Tensor
        y_flat: torch.Tensor
        gate_flat, y_flat = ctx.saved_tensors
        grad_out_flat: torch.Tensor = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        M: int = grad_out_flat.shape[0]
        N: int = grad_out_flat.shape[1]
        grad_x: torch.Tensor = torch.empty_like(grad_out_flat)
        grad_gate: torch.Tensor = torch.empty_like(grad_out_flat)
        grad_y: torch.Tensor = torch.empty_like(grad_out_flat)
        BLOCK_SIZE: int = triton.next_power_of_2(N)

        _fused_gated_residual_bwd_kernel[(M,)](
            grad_out_flat,
            gate_flat,
            y_flat,
            grad_x,
            grad_gate,
            grad_y,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        grad_x = grad_x.reshape(ctx.orig_shape)
        grad_gate = grad_gate.reshape(ctx.orig_shape)
        grad_y = grad_y.reshape(ctx.orig_shape)

        # Reduce gate gradient if it was broadcast
        grad_gate = _reduce_broadcast_grad(grad_gate, ctx.orig_shape, ctx.gate_shape)

        return grad_x, grad_gate, grad_y


def fused_gated_residual(
    x: torch.Tensor,
    gate: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    if x.is_cuda:
        return _FusedGatedResidual.apply(x, gate, y)
    return x + gate * y


# ============================================================================
# fused_rms_norm: F.normalize(x, dim=-1) * scale_const * g
# ============================================================================


@triton.jit
def _fused_rms_norm_fwd_kernel(
    X_ptr,
    G_ptr,
    Out_ptr,
    RNorm_ptr,
    scale: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row: int = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X_ptr + row * N + offsets, mask=mask, other=0.0)
    g = tl.load(G_ptr + row * N + offsets, mask=mask, other=0.0)

    # L2 norm
    sum_sq = tl.sum(x * x, axis=0)
    rnorm = tl.rsqrt(sum_sq + 1e-12)

    out = x * rnorm * scale * g
    tl.store(Out_ptr + row * N + offsets, out, mask=mask)
    # Store rnorm for backward
    tl.store(RNorm_ptr + row, rnorm)


@triton.jit
def _fused_rms_norm_bwd_kernel(
    Grad_out_ptr,
    X_ptr,
    G_ptr,
    RNorm_ptr,
    Grad_x_ptr,
    Grad_g_ptr,
    scale: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row: int = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    grad_out = tl.load(
        Grad_out_ptr + row * N + offsets,
        mask=mask,
        other=0.0,
    )
    x = tl.load(X_ptr + row * N + offsets, mask=mask, other=0.0)
    g = tl.load(G_ptr + row * N + offsets, mask=mask, other=0.0)
    rnorm = tl.load(RNorm_ptr + row)

    # Forward: out = x * rnorm * scale * g
    # Let s = scale * g (element-wise)
    # out = x * rnorm * s
    s = scale * g
    x_hat = x * rnorm  # normalized x

    # grad_g = grad_out * x_hat * scale (accumulated outside kernel)
    grad_g_val = grad_out * x_hat * scale
    tl.store(Grad_g_ptr + row * N + offsets, grad_g_val, mask=mask)

    # grad_x: d(x_hat * s)/dx where x_hat = x / ||x||
    # d(x_hat)/dx_i = (1/||x||) * (delta_ij - x_hat_i * x_hat_j)
    # grad_x = rnorm * (grad_out * s - x_hat * sum(grad_out * s * x_hat))
    gs = grad_out * s
    dot = tl.sum(gs * x_hat, axis=0)
    grad_x = rnorm * (gs - x_hat * dot)

    tl.store(Grad_x_ptr + row * N + offsets, grad_x, mask=mask)


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        g: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        orig_shape: torch.Size = x.shape
        orig_dtype: torch.dtype = x.dtype
        N: int = x.shape[-1]

        # Upcast to float32 for bf16/fp16 to ensure numerical stability in norm
        compute_dtype: torch.dtype = torch.float32 if orig_dtype in (torch.bfloat16, torch.float16) else orig_dtype
        x_2d: torch.Tensor = x.to(compute_dtype).contiguous().reshape(-1, N)
        M: int = x_2d.shape[0]

        # Broadcast g to match x, then flatten to (M, N) — correct for any g shape
        g_expanded: torch.Tensor = g.expand_as(x).to(compute_dtype).contiguous().reshape(-1, N)

        out: torch.Tensor = torch.empty_like(x_2d)
        rnorm: torch.Tensor = torch.empty(M, device=x.device, dtype=compute_dtype)
        BLOCK_SIZE: int = triton.next_power_of_2(N)

        _fused_rms_norm_fwd_kernel[(M,)](
            x_2d,
            g_expanded,
            out,
            rnorm,
            scale=scale,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x_2d, g_expanded, rnorm)
        ctx.scale = scale
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = orig_dtype
        ctx.g_shape = g.shape
        return out.reshape(orig_shape).to(orig_dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        x_2d: torch.Tensor
        g_expanded: torch.Tensor
        rnorm: torch.Tensor
        x_2d, g_expanded, rnorm = ctx.saved_tensors
        M: int = x_2d.shape[0]
        N: int = x_2d.shape[1]
        compute_dtype: torch.dtype = x_2d.dtype
        grad_out_2d: torch.Tensor = grad_output.to(compute_dtype).reshape(M, N).contiguous()

        grad_x: torch.Tensor = torch.empty_like(x_2d)
        grad_g_expanded: torch.Tensor = torch.empty_like(x_2d)
        BLOCK_SIZE: int = triton.next_power_of_2(N)

        _fused_rms_norm_bwd_kernel[(M,)](
            grad_out_2d,
            x_2d,
            g_expanded,
            rnorm,
            grad_x,
            grad_g_expanded,
            scale=ctx.scale,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        grad_x = grad_x.reshape(ctx.orig_shape).to(ctx.orig_dtype)

        # Reduce grad_g back to the original g shape by summing broadcast dims
        grad_g_full: torch.Tensor = grad_g_expanded.reshape(ctx.orig_shape)
        grad_g: torch.Tensor = _reduce_broadcast_grad(grad_g_full, ctx.orig_shape, ctx.g_shape)
        grad_g = grad_g.to(ctx.orig_dtype)

        return grad_x, grad_g, None  # None for scale (not a tensor)


def fused_rms_norm(x: torch.Tensor, g: torch.Tensor, scale: float) -> torch.Tensor:
    if x.is_cuda:
        return _FusedRMSNorm.apply(x, g, scale)

    return F.normalize(x, dim=-1) * scale * g
