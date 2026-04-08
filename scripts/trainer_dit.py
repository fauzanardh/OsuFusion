import time
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from bitsandbytes.optim import AdamW8bit
from diffusers.optimization import get_cosine_schedule_with_warmup
from matplotlib import pyplot as plt
from PIL import Image
from safetensors.torch import save_file
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from trainer_utils import clear_checkpoints, get_total_norm, manage_checkpoints

import wandb
from osu_fusion.data.dataset import BeatmapDataset, BucketBatchSampler, filter_maps

# from osu_fusion.data.dataset import count_num_mappers
from osu_fusion.data.encode import SEQ_DIM
from osu_fusion.data.prepare_data import load_audio
from osu_fusion.models.diffusion_dit import DiTConfig_L, DiTConfig_M, DiTConfig_S, OsuFusionDiT


def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_lens = torch.tensor([x.shape[0] for x, _, _ in batch], dtype=torch.int32)
    max_len = max(x.shape[0] for x, _, _ in batch)

    padded_x = []
    padded_a = []
    for x, a, _ in batch:
        n_pad = max_len - x.shape[0]
        if n_pad > 0:
            x = F.pad(x, (0, 0, 0, n_pad))
            a = F.pad(a, (0, 0, 0, n_pad))
        padded_x.append(x)
        padded_a.append(a)

    out_x = torch.stack(padded_x)
    out_a = torch.stack(padded_a)
    out_c = torch.stack([c for _, _, c in batch])
    # out_desc = torch.stack([d for _, _, _, d, _ in batch])
    # out_mapper = torch.stack([m for _, _, _, _, m in batch])
    return out_x, out_a, out_c, orig_lens


def visualize_and_log_sample(
    accelerator: Accelerator,
    model: OsuFusionDiT,
    audio_path: Path,
    step: int,
) -> None:
    a = load_audio(audio_path)
    c = np.array([4.0, 9.5, 9.5, 4.0, 6.0, 1.4, 1.0, 3.0], dtype=np.float32)

    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(
        accelerator.mixed_precision,
        torch.float32,
    )

    a_tensor = torch.from_numpy(a).unsqueeze(0).to(accelerator.device, dtype)
    c_tensor = torch.from_numpy(c).unsqueeze(0).to(accelerator.device, dtype)

    model.eval()
    with torch.inference_mode(), accelerator.autocast():
        generated = model.sample(a_tensor, c_tensor, cond_scale=1.0)
    model.train()

    generated = generated.cpu().detach().float()
    n_frames = generated.shape[1]

    width = max(8, min(n_frames // 100, 40))
    fig, axs = plt.subplots(SEQ_DIM, 1, figsize=(width, SEQ_DIM * 1.5), sharex=True)

    for i in range(SEQ_DIM):
        axs[i].plot(generated[0, :, i].cpu(), color="red", linewidth=0.5)

    fig.canvas.draw()
    pil_img = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba().tobytes())
    accelerator.log({"generated": wandb.Image(pil_img)}, step=step)
    plt.close(fig)


def save_model_state(model: OsuFusionDiT, project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.dit.state_dict(), project_dir / "dit.safetensors")


def save_training_checkpoint(
    model: OsuFusionDiT,
    optimizer: AdamW8bit,
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
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
    }

    torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")
    torch.cuda.empty_cache()


def filter_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    filtered_state_dict = {}
    model_state_dict = model.state_dict()
    for key, param in model_state_dict.items():
        if key in state_dict and param.size() == state_dict[key].size():
            filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict


def load_training_checkpoint(
    model: OsuFusionDiT,
    optimizer: AdamW8bit,
    scheduler: LambdaLR,
    checkpoint_path: Path,
    reset_steps: bool = False,
) -> int:
    print(f"Loading checkpoint from {checkpoint_path}...")
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path / "checkpoint.pt", map_location=device)

    try:
        model.dit.load_state_dict(checkpoint["dit_state_dict"])
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


MODEL_CONFIGS = {"s": DiTConfig_S, "m": DiTConfig_M, "l": DiTConfig_L}


def train(args: ArgumentParser) -> None:  # noqa: C901
    start_time = time.time()
    print("Initializing...")
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=ProjectConfiguration(project_dir=args.project_dir, automatic_checkpoint_naming=True),
        log_with="wandb",
    )
    accelerator.init_trackers(project_name="OsuFusion")

    # Count mappers and build dataset
    print("Loading dataset...")
    all_maps = list(args.dataset_dir.rglob("*.map.h5"))
    all_maps, all_lengths = filter_maps(all_maps, max_length=args.max_length)
    # num_mappers = count_num_mappers(args.dataset_dir)

    config = MODEL_CONFIGS[args.model_size]
    # config.num_mappers = num_mappers
    model = OsuFusionDiT(**asdict(config))
    model.dit.set_gradient_checkpointing(args.gradient_checkpointing)
    if args.full_bf16:
        model.set_full_bf16()

    dataset = BeatmapDataset(dataset=all_maps, lengths=all_lengths)
    bucket_sampler = BucketBatchSampler(
        lengths=all_lengths,
        batch_size=args.batch_size,
        bucket_boundaries=args.bucket_boundaries,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=bucket_sampler,
        num_workers=args.num_workers,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    steps_per_epoch = max(1, len(bucket_sampler) // args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs

    parameters = list(model.trainable_params)
    print(f"Number of trainable parameters: {sum(p.numel() for p in parameters)}")
    optimizer = AdamW8bit(parameters, lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=args.warmup_steps,
        num_cycles=0.5,
    )

    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)

    current_step = (
        load_training_checkpoint(model, optimizer, scheduler, args.resume, args.reset_steps) if args.resume else 0
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
            bucket_sampler.set_epoch(epoch)
            epoch_loss_history = []
            accum_diff_loss = 0.0
            accum_total_loss = 0.0

            for batch in dataloader:
                metrics_total_norm = 0.0
                x, a, c, orig_lens = batch

                with accelerator.autocast(), accelerator.accumulate(model):
                    try:
                        loss = model(x, a, c, orig_lens=orig_lens)
                    except AssertionError:
                        print(f"AssertionError encountered at step {current_step + 1}, skipping batch.")
                        continue

                    accelerator.backward(loss)

                    accum_diff_loss += loss.item() / args.gradient_accumulation_steps
                    accum_total_loss += loss.item() / args.gradient_accumulation_steps

                    if accelerator.sync_gradients:
                        metrics_total_norm = get_total_norm(parameters)
                        if args.clip_grad_norm > 0.0:
                            accelerator.clip_grad_norm_(parameters, args.clip_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    epoch_loss_history.append(accum_total_loss)
                    loss_history.append(accum_total_loss)  # Track globally
                    avg_loss = sum(epoch_loss_history) / len(epoch_loss_history)

                    pbar.set_description(
                        f"Ep {epoch + 1}/{args.epochs} | Step {current_step + 1} | "
                        f"Loss {accum_total_loss:.4f} | Avg {avg_loss:.4f}",
                    )
                    pbar.update(1)

                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "total_loss": accum_total_loss,
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
                        visualize_and_log_sample(accelerator, model, args.sample_audio, step=current_step + 1)

                    current_step += 1
                    accum_diff_loss = 0.0
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
    args.add_argument("--model-size", type=str, default="s", choices=["s", "m", "l"], help="Model size: s, m, l")
    args.add_argument("--resume", type=Path, default=None, help="Path to resume from a checkpoint")
    args.add_argument("--reset-steps", action="store_true", help="Reset training steps when resuming")
    args.add_argument("--max-length", type=int, default=0, help="Maximum length of beatmaps to include")
    args.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16", help="Mixed precision mode")
    args.add_argument("--full-bf16", action="store_true", help="Use full bfloat16 precision")
    args.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    args.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    args.add_argument("--clip-grad-norm", type=float, default=0.0, help="Gradient clipping norm")
    args.add_argument(
        "--bucket-boundaries",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
        help="Length bucket boundaries for batching (e.g., 1024 2048 4096 8192 16384)",
    )
    args.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer")
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
