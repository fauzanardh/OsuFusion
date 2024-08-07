import math
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from peft.tuners.lora import LoraLayer
from peft.tuners.lora.dora import DoraLinearLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from torch.nn import functional as F  # noqa: N812


class DoraConv1dLayer(DoraLinearLayer):
    def get_weight_norm(
        self: "DoraConv1dLayer",
        weight: torch.Tensor,
        lora_weight: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = weight + scaling * lora_weight
        # the following is needed to have compatibility with the 3D weight tensors of Conv1D
        weight_norm = weight.norm(p=2, dim=(1, 2), keepdim=True).transpose(1, 0)
        return weight_norm

    def update_layer(
        self: "DoraConv1dLayer",
        *,
        base_layer: nn.Conv1d,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float,
        place_on_cpu: bool = False,
    ) -> None:
        # temporarily convert fp16 to fp32, as fp16 can cause trouble on CPU with PyTorch < 2.2
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()

        with gather_params_ctx(base_layer.parameters()):
            if base_layer.__class__.__name__ == "Linear4bit":
                base_layer = deepcopy(base_layer)

            weight = dequantize_module_weight(base_layer)
            lora_weight = torch.mm(lora_B.flatten(start_dim=1), lora_A.flatten(start_dim=1))
            lora_weight = lora_weight.reshape(weight.shape)

            if dtype_is_fp16:
                lora_weight = lora_weight.half()
            weight_norm = self.get_weight_norm(weight.to(lora_A.device), lora_weight, scaling)

        if place_on_cpu:
            weight_norm = weight_norm.to("cpu")
        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def forward(
        self: "DoraConv1dLayer",
        x: torch.Tensor,
        *,
        lora_A: nn.Conv1d,
        lora_B: nn.Conv1d,
        scaling: float,
        base_layer: nn.Conv1d,
    ) -> torch.Tensor:
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        weight = base_layer.weight
        lora_weight = torch.mm(lora_B.weight.flatten(start_dim=1), lora_A.weight.flatten(start_dim=1))
        lora_weight = lora_weight.reshape(weight.shape)
        magnitude = self.weight
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)

        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm
        result_dora = (mag_norm_scale - 1) * (
            F.conv1d(
                x,
                weight,
                bias=None,
                stride=base_layer.stride[0],
                padding=base_layer.padding[0],
                dilation=base_layer.dilation[0],
                groups=base_layer.groups,
            )
        ) + mag_norm_scale * lora_B(lora_A(x)) * scaling

        return result_dora

    def __repr__(self: "DoraConv1dLayer") -> str:
        rep = super().__repr__()
        return "lora.dora." + rep


class LoraConv1d(nn.Module, LoraLayer):
    def __init__(
        self: "LoraConv1d",
        base_layer: nn.Conv1d,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self.in_features = base_layer.in_channels
        self.out_features = base_layer.out_channels

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(
        self: "LoraConv1d",
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: Union[bool, str],
        use_rslora: bool,
        use_dora: bool = False,
    ) -> None:
        if r <= 0:
            msg = f"`r` should be a positive integer value but the value passed is {r}"
            raise ValueError(msg)

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        lora_dropout_layer = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout[adapter_name] = lora_dropout_layer

        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size[0]  # Conv1d has a tuple for kernel_size, stride, and padding
        stride = base_layer.stride[0]
        padding = base_layer.padding[0]
        self.lora_A[adapter_name] = nn.Conv1d(
            self.in_features,
            r,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.lora_B[adapter_name] = nn.Conv1d(r, self.out_features, 1, stride=1, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def dora_init(self: "LoraConv1d", adapter_name: str) -> None:
        if self.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)
        dora_layer = DoraConv1dLayer(fan_in_fan_out=False)
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=self.get_base_layer(),
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
        )
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def merge(self: "LoraConv1d", safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if safe_merge:
                orig_weights = base_layer.weight.data.clone()
                delta_weight = self.get_delta_weight(active_adapter)

                if not self.use_dora[active_adapter]:
                    orig_weights += delta_weight
                else:
                    weight_norm = (
                        self.lora_magnitude_vector[active_adapter]
                        .get_weight_norm(orig_weights, delta_weight, scaling=1)
                        .detach()
                    )
                    self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    orig_weights = dora_factor.view(-1, 1, 1) * (orig_weights + delta_weight)

                if not torch.isfinite(orig_weights).all():
                    msg = f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    raise ValueError(msg)
                base_layer.weight.data = orig_weights
            else:
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    base_layer.weight.data += delta_weight
                else:
                    weight_norm = (
                        self.lora_magnitude_vector[active_adapter]
                        .get_weight_norm(base_layer.weight.data, delta_weight, scaling=1)
                        .detach()
                    )
                    self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    new_weight = dora_factor.view(-1, 1, 1) * (base_layer.weight.data + delta_weight)
                    base_layer.weight.data = new_weight

            self.merged_adapters.append(active_adapter)

    def unmerge(self: "LoraConv1d") -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")  # noqa: B028
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A:
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self: "LoraConv1d", adapter: str) -> torch.Tensor:
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # Conv1d 1x1 case
        if self.get_base_layer().weight.size()[2:3] == (1,):
            output_tensor = (weight_B.squeeze(2) @ weight_A.squeeze(2)).unsqueeze(2) * self.scaling[adapter]
        else:
            output_tensor = (
                F.conv1d(
                    weight_A.permute(1, 0, 2),
                    weight_B,
                ).permute(1, 0, 2)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self: "LoraConv1d", x: torch.Tensor, *args: List, **kwargs: Dict) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A:
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self: "LoraConv1d") -> str:
        rep = super().__repr__()
        return "lora." + rep
