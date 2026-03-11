import shutil
import time
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_cosine_schedule_with_warmup
from matplotlib import pyplot as plt
from PIL import Image
from safetensors.torch import save_file
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from osu_fusion.data.const import BEATMAP_DIM
from osu_fusion.data.encode import SequenceEncoding, SEQ_DIM
from osu_fusion.data.dataset import BeatmapDataset
from osu_fusion.data.prepare_data import load_audio
from osu_fusion.models.diffusion_dit import DiTConfig_S, DiTConfig_M, DiTConfig_L, OsuFusionDiT


def get_total_norm(parameters: List[torch.Tensor], norm_type: float = 2.0) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type).item()


def filter_dataset(dataset: List[Path], max_length: int) -> List[Path]:
    filtered = []
    for path in tqdm(dataset, desc="Filtering dataset...", dynamic_ncols=True):
        try:
            with h5py.File(path, "r") as f:
                if f["x"].shape[0] <= max_length:
                    filtered.append(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
    return filtered


def manage_checkpoints(project_dir: Path, max_num_checkpoints: int) -> None:
    checkpoints = sorted(project_dir.rglob("checkpoint-*"), key=lambda p: int(p.stem.split("-")[1]))
    for checkpoint in checkpoints[:-max_num_checkpoints]:
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
        else:
            checkpoint.unlink()


def clear_checkpoints(project_dir: Path) -> None:
    for checkpoint in project_dir.rglob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
        else:
            checkpoint.unlink()


def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: (N, D), a: (T, D_a), c: (D_c,)
    orig_lens = torch.tensor([x.shape[0] for x, _, _ in batch], dtype=torch.int32)
    max_x_len = max(x.shape[0] for x, _, _ in batch)
    max_a_len = max(a.shape[0] for _, a, _ in batch)

    # PAD token: continuous channels = 0.0 (neutral), type flags = -1.0 (inactive), TYPE_PAD = 1.0
    pad_token = torch.zeros(SEQ_DIM)
    pad_token[SequenceEncoding.NEW_COMBO] = -1.0
    pad_token[6:18] = -1.0  # All type flags inactive
    pad_token[SequenceEncoding.TYPE_PAD] = 1.0

    padded_x = []
    for x, _, _ in batch:
        n_pad = max_x_len - x.shape[0]
        if n_pad > 0:
            pad_block = pad_token.unsqueeze(0).expand(n_pad, -1).to(x.device)
            x = torch.cat([x, pad_block], dim=0)
        padded_x.append(x)

    out_x = torch.stack(padded_x)
    out_a = torch.stack([F.pad(a, (0, 0, 0, max_a_len - a.shape[0]), value=0.0) for _, a, _ in batch])
    out_c = torch.stack([c for _, _, c in batch])
    return out_x, out_a, out_c, orig_lens


def visualize_and_log_sample(
    accelerator: Accelerator,
    model: OsuFusionDiT,
    audio_path: Path,
    step: int,
) -> None:
    a = load_audio(audio_path)
    c = np.array([4.0, 9.5, 9.5, 4.0, 6.0, 1.4, 1.0], dtype=np.float32)

    dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(accelerator.mixed_precision, torch.float32)

    a_tensor = torch.from_numpy(a).unsqueeze(0).to(accelerator.device, dtype)
    c_tensor = torch.from_numpy(c).unsqueeze(0).to(accelerator.device, dtype)

    model.eval()
    with torch.inference_mode(), accelerator.autocast():
        generated = model.sample(a_tensor, c_tensor, cond_scale=1.0)
    model.train()

    # generated: (B, N, D)
    generated = generated.cpu().detach().float()
    n_events = generated.shape[1]
    width = max(1, n_events // 50)
    fig, axs = plt.subplots(BEATMAP_DIM, 1, figsize=(width, BEATMAP_DIM * 8), sharex=True)
    for i in range(BEATMAP_DIM):
        axs[i].plot(generated[0, :, i].cpu(), color="red", linewidth=0.5)

    fig.canvas.draw()
    pil_img = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba().tobytes())
    accelerator.log({"generated": wandb.Image(pil_img)}, step=step)
    plt.close(fig)


def save_model_state(model: OsuFusionDiT, project_dir: Path) -> None:
    save_file(model.dit.state_dict(), project_dir / "dit.safetensors")
    save_file(model.length_predictor.state_dict(), project_dir / "length_predictor.safetensors")


def save_training_checkpoint(
    model: OsuFusionDiT,
    optimizer: AdamW,
    scheduler: LambdaLR,
    current_step: int,
    project_dir: Path,
    is_nan: bool = False,
) -> None:
    checkpoint_dir = project_dir / f"checkpoint-{current_step + 1}{'-nan' if is_nan else ''}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to {checkpoint_dir}...")

    checkpoint = {
        "dit_state_dict": model.dit.state_dict(),
        "length_predictor_state_dict": model.length_predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
    }

    torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")
    torch.cuda.empty_cache()


def filter_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    filtered_state_dict = {}
    model_state_dict = model.state_dict()
    for key, param in model_state_dict.items():
        if key in state_dict and param.size() == state_dict[key].size():
            filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict


def load_training_checkpoint(
    model: OsuFusionDiT,
    optimizer: AdamW,
    scheduler: LambdaLR,
    checkpoint_path: Path,
    reset_steps: bool = False,
) -> int:
    print(f"Loading checkpoint from {checkpoint_path}...")
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path / "checkpoint.pt", map_location=device)

    try:
        model.dit.load_state_dict(checkpoint["dit_state_dict"])
        if "length_predictor_state_dict" in checkpoint:
            model.length_predictor.load_state_dict(checkpoint["length_predictor_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except RuntimeError:
        filtered_state_dict = filter_state_dict(model.dit, checkpoint["dit_state_dict"])
        incompatible_keys = model.dit.load_state_dict(filtered_state_dict, strict=False)
        if len(incompatible_keys.missing_keys) > 0:
            print(f"Missing keys: {incompatible_keys.missing_keys}")
        if len(incompatible_keys.unexpected_keys) > 0:
            print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")

    if not reset_steps:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    torch.set_rng_state(checkpoint["rng_state"].cpu())
    return 0 if reset_steps else int(checkpoint_path.stem.split("-")[1])


MODEL_CONFIGS = {
    "s": DiTConfig_S,
    "m": DiTConfig_M,
    "l": DiTConfig_L,
}


def train(args: ArgumentParser) -> None:  # noqa: C901
    start_time = time.time()
    print("Initializing...")
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=ProjectConfiguration(
            project_dir=args.project_dir,
            automatic_checkpoint_naming=True,
        ),
        log_with="wandb",
    )
    accelerator.init_trackers(project_name="OsuFusion")

    config = MODEL_CONFIGS[args.model_size]
    model = OsuFusionDiT(**asdict(config))
    model.dit.set_gradient_checkpointing(args.gradient_checkpointing)
    if args.full_bf16:
        model.set_full_bf16()

    print("Loading dataset...")
    all_maps = list(args.dataset_dir.rglob("*.map.h5"))
    if args.max_length > 0:
        all_maps = filter_dataset(all_maps, args.max_length)

    dataset = BeatmapDataset(dataset=all_maps)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    steps_per_epoch = max(1, len(all_maps) // (args.batch_size * args.gradient_accumulation_steps))
    total_steps = steps_per_epoch * args.epochs

    parameters = list(model.trainable_params)
    print(f"Number of trainable parameters: {sum(p.numel() for p in parameters)}")
    optimizer = AdamW(parameters, lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=args.warmup_steps,
        num_cycles=0.5,
    )

    model, optimizer, scheduler, dataloader = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        dataloader,
    )

    current_step = (
        load_training_checkpoint(
            model,
            optimizer,
            scheduler,
            args.resume,
            args.reset_steps,
        )
        if args.resume
        else 0
    )
    starting_epoch = current_step // steps_per_epoch

    model.train()
    if args.resume is None:
        print("Clearing old checkpoints...")
        clear_checkpoints(args.project_dir)

    print("Starting training...")
    loss_history = []

    with tqdm(
        total=total_steps - current_step,
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    ) as pbar:
        for epoch in range(starting_epoch, args.epochs):
            accum_diff_loss = 0.0
            accum_length_loss = 0.0
            accum_total_loss = 0.0

            for batch in dataloader:
                metrics_total_norm = 0.0
                x, a, c, orig_lens = batch

                with accelerator.autocast(), accelerator.accumulate(model):
                    try:
                        diff_loss, length_loss = model(x, a, c, orig_lens)
                        loss = diff_loss + args.length_loss_weight * length_loss
                    except AssertionError:
                        print(f"AssertionError encountered at step {current_step + 1}, skipping batch.")
                        continue

                    accelerator.backward(loss)

                    accum_diff_loss += diff_loss.item() / args.gradient_accumulation_steps
                    accum_length_loss += length_loss.item() / args.gradient_accumulation_steps
                    accum_total_loss += loss.item() / args.gradient_accumulation_steps

                    if accelerator.sync_gradients:
                        metrics_total_norm = get_total_norm(parameters)
                        if args.clip_grad_norm > 0.0:
                            accelerator.clip_grad_norm_(parameters, args.clip_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    loss_history.append(accum_total_loss)
                    if len(loss_history) > args.save_every:
                        loss_history.pop(0)
                    avg_loss = sum(loss_history) / len(loss_history)

                    pbar.set_description(
                        f"Ep {epoch + 1}/{args.epochs} | Step {current_step + 1} | "
                        f"Loss {accum_total_loss:.4f} | Diff {accum_diff_loss:.4f} | "
                        f"Len {accum_length_loss:.4f} | Avg {avg_loss:.4f}",
                    )
                    pbar.update(1)

                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "total_loss": accum_total_loss,
                                "diff_loss": accum_diff_loss,
                                "length_loss": accum_length_loss,
                                "total_norm": metrics_total_norm,
                                "lr": scheduler.get_last_lr()[0],
                            },
                            step=current_step + 1,
                        )

                    if (current_step + 1) % args.save_every == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            accelerator.log({"save_loss": avg_loss}, step=current_step + 1)
                            save_training_checkpoint(
                                accelerator.unwrap_model(model),
                                optimizer,
                                scheduler,
                                current_step,
                                args.project_dir,
                            )
                            manage_checkpoints(args.project_dir, args.max_num_checkpoints)

                    if (
                        (current_step + 1) % args.sample_every == 0
                        and accelerator.is_main_process
                        and args.sample_audio is not None
                        and args.sample_audio.exists()
                    ):
                        print("Sampling...")
                        visualize_and_log_sample(
                            accelerator,
                            model,
                            args.sample_audio,
                            step=current_step + 1,
                        )

                    current_step += 1

                    accum_diff_loss = 0.0
                    accum_length_loss = 0.0
                    accum_total_loss = 0.0

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_model_state(accelerator.unwrap_model(model), args.project_dir)

        total_time = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        print("\n---")
        print(f"model_size: {args.model_size}")
        print(f"model_parameters: {sum(p.numel() for p in parameters)}")
        print(f"batch_size: {args.batch_size}")
        print(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")
        print(f"learning_rate: {args.lr}")
        print(f"total_epochs: {args.epochs}")
        print(f"total_steps: {total_steps}")
        print(f"total_training_time_seconds: {total_time:.2f}")
        print(f"peak_vram_mb: {peak_vram:.2f}")
        if len(loss_history) > 0:
            print(f"final_avg_loss: {sum(loss_history) / len(loss_history):.5f}")


def main() -> None:
    args = ArgumentParser(description="Train OsuFusion DiT")
    args.add_argument("--project-dir", type=Path, required=True, help="Directory for project outputs")
    args.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing the dataset")
    args.add_argument(
        "--model-size",
        type=str,
        default="s",
        choices=["s", "m", "l"],
        help="Model size: s (~46M), m (~130M), l (~400M)",
    )
    args.add_argument("--resume", type=Path, default=None, help="Path to resume from a checkpoint")
    args.add_argument("--reset-steps", action="store_true", help="Reset training steps when resuming")
    args.add_argument("--max-length", type=int, default=0, help="Maximum length of beatmaps to include")
    args.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
        help="Mixed precision mode",
    )
    args.add_argument("--full-bf16", action="store_true", help="Use full bfloat16 precision")
    args.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    args.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    args.add_argument("--clip-grad-norm", type=float, default=0.0, help="Gradient clipping norm")
    args.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer")
    args.add_argument("--length-loss-weight", type=float, default=0.1, help="Weight for length prediction loss")
    args.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    args.add_argument("--num-workers", type=int, default=2, help="Number of data loader workers")
    args.add_argument("--epochs", type=int, default=100, help="Total number of training epochs")
    args.add_argument("--warmup-steps", type=int, default=1000, help="Number of warmup steps for scheduler")
    args.add_argument("--save-every", type=int, default=1_000, help="Save checkpoint every N steps")
    args.add_argument("--max-num-checkpoints", type=int, default=5, help="Maximum number of checkpoints to keep")
    args.add_argument("--sample-every", type=int, default=1_000, help="Sample and log every N steps")
    args.add_argument("--sample-audio", type=Path, default=None, help="Path to sample audio for visualization")
    args = args.parse_args()

    train(args)


if __name__ == "__main__":
    main()
