import random
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Generator, List, Tuple, Union

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import FP8RecipeKwargs, ProjectConfiguration
from diffusers.optimization import get_cosine_schedule_with_warmup
from matplotlib import pyplot as plt
from PIL import Image
from safetensors.torch import save_file
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from osu_fusion.data.const import AUDIO_DIM, BEATMAP_DIM
from osu_fusion.data.dataset import FullSequenceDataset, SubsequenceDataset
from osu_fusion.models.autoencoder import AudioAutoEncoder, OsuAutoEncoder

Model = Union[OsuAutoEncoder, AudioAutoEncoder]


def get_total_norm(parameters: List[torch.Tensor], norm_type: float = 2.0) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type).item()


def filter_dataset(dataset: List[Path], max_length: int) -> List[Path]:
    filtered = [
        path
        for path in tqdm(dataset, desc="Filtering dataset...", dynamic_ncols=True)
        if np.load(path)["x"].shape[1] <= max_length
    ]
    return filtered


def cycle_dataloader(dataloader: DataLoader) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    while True:
        for batch in dataloader:
            yield batch


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(x.shape[1] for x, _, _ in batch)
    out_x = torch.stack([F.pad(x, (0, max_length - x.shape[1]), value=-1.0) for x, _, _ in batch])
    out_a = torch.stack([F.pad(a, (0, max_length - a.shape[1]), value=-1.0) for _, a, _ in batch])
    out_c = torch.stack([c for _, _, c in batch])
    return out_x, out_a, out_c


def visualize_and_log_sample(
    args: ArgumentParser,
    accelerator: Accelerator,
    model: Model,
    step: int,
) -> None:
    dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(accelerator.mixed_precision, torch.float32)

    data_key = "x" if args.osu_data else "a"
    x = torch.from_numpy(np.load(args.sample_data)[data_key]).unsqueeze(0).to(accelerator.device, dtype)

    model.eval()
    with torch.inference_mode(), accelerator.autocast():
        z, _ = model.encode(x)
        reconstructed = torch.cat(model.decode(z, apply_act=True), dim=1) if args.osu_data else model.decode(z)
    model.train()

    features = BEATMAP_DIM if args.osu_data else AUDIO_DIM
    max_features = min(6, features)
    width, height = reconstructed.shape[-1] // 150, max_features

    fig, axs = plt.subplots(height, 1, figsize=(width, height * 8), sharex=True)
    for i in range(max_features):
        axs[i].plot(reconstructed[0, i].cpu(), label="Reconstructed", color="red")
        axs[i].plot(x[0, i].cpu(), label="Original", color="blue", linestyle="--")
        axs[i].legend()

    fig.canvas.draw()
    pil_img = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba().tobytes())
    accelerator.log({"generated": wandb.Image(pil_img)}, step=step)
    plt.close(fig)


def save_model_state(model: Model, project_dir: Path) -> None:
    save_file(model.state_dict(), project_dir / "model.safetensors")


def save_training_checkpoint(
    model: Model,
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
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
    }

    torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")
    torch.cuda.empty_cache()


def load_training_checkpoint(
    model: Model,
    optimizer: AdamW,
    scheduler: LambdaLR,
    checkpoint_path: Path,
    reset_steps: bool = False,
) -> int:
    print(f"Loading checkpoint from {checkpoint_path}...")
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path / "checkpoint.pt", map_location=device)

    # Load model and optimizer states with error handling
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        print("Loading model_state_dict with strict=False due to model changes.")
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if incompatible.missing_keys:
            print(f"Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys: {incompatible.unexpected_keys}")

    # Load optimizer and scheduler states
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if not reset_steps:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    torch.set_rng_state(checkpoint["rng_state"].cpu())
    return 0 if reset_steps else int(checkpoint_path.stem.split("-")[1])


def train(args: ArgumentParser) -> None:  # noqa: C901
    print("Initializing...")
    accelerate_kwargs = [FP8RecipeKwargs(backend="msamp", opt_level="O1")] if args.mixed_precision == "fp8" else None
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=ProjectConfiguration(
            project_dir=args.project_dir,
            automatic_checkpoint_naming=True,
        ),
        log_with="wandb",
        kwargs_handlers=accelerate_kwargs,
    )
    accelerator.init_trackers(project_name="OsuFusion-AutoEncoder")

    # Initialize model
    model = OsuAutoEncoder(32, 128) if args.osu_data else AudioAutoEncoder(16, 128)
    if args.full_bf16:
        model.set_full_bf16()

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=args.total_steps,
        num_warmup_steps=args.warmup_steps,
        num_cycles=0.5,
    )

    print("Loading dataset...")
    all_maps = list(args.dataset_dir.rglob("*.map.npz"))
    if args.max_length > 0:
        all_maps = filter_dataset(all_maps, args.max_length)
    random.shuffle(all_maps)

    dataset_cls = FullSequenceDataset if args.full_sequence else SubsequenceDataset
    dataset = dataset_cls(dataset=all_maps, segment_sr=False, load_audio=not args.osu_data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=custom_collate_fn if args.full_sequence else None,
    )

    # Prepare everything with accelerator
    model, optimizer, scheduler, dataloader = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        dataloader,
    )

    # Load checkpoint if resuming
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

    if args.resume is None:
        print("Clearing old checkpoints...")
        clear_checkpoints(args.project_dir)

    print("Starting training...")
    dataloader_cycle = cycle_dataloader(dataloader)
    loss_history = []

    with tqdm(
        total=args.total_steps - current_step,
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    ) as pbar:
        while current_step < args.total_steps:
            # Initialize batch metrics
            metrics = {
                "total_loss": 0.0,
                "kl_loss": 0.0,
                "total_norm": 0.0,
            }
            if args.osu_data:
                metrics.update({"hit_loss": 0.0, "cursor_loss": 0.0})
            else:
                metrics.update({"audio_loss": 0.0})

            model.train()
            for _ in range(args.gradient_accumulation_steps):
                batch = next(dataloader_cycle)
                x, a, _ = batch
                model_input = x if args.osu_data else a

                with accelerator.autocast(), accelerator.accumulate(model):
                    # Forward pass
                    model_output = model(model_input)
                    if args.osu_data:
                        hit_loss, cursor_loss, kl_loss = model_output
                        loss = hit_loss + cursor_loss + kl_loss * args.kl_weight
                    else:
                        audio_loss, kl_loss = model_output
                        loss = audio_loss + kl_loss * args.kl_weight

                    # Backward and optimize
                    accelerator.backward(loss)
                    metrics["total_norm"] += get_total_norm(model.parameters()) / args.gradient_accumulation_steps
                    metrics["total_loss"] += loss.item() / args.gradient_accumulation_steps
                    metrics["kl_loss"] += kl_loss.item() / args.gradient_accumulation_steps

                    if args.osu_data:
                        metrics["hit_loss"] += hit_loss.item() / args.gradient_accumulation_steps
                        metrics["cursor_loss"] += cursor_loss.item() / args.gradient_accumulation_steps
                    else:
                        metrics["audio_loss"] += audio_loss.item() / args.gradient_accumulation_steps

                    if args.clip_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            # Update loss history and progress bar
            loss_history.append(metrics["total_loss"])
            if len(loss_history) > args.save_every:
                loss_history.pop(0)
            avg_loss = sum(loss_history) / len(loss_history)

            pbar.set_description(
                f"Steps: {current_step + 1}, Loss: {metrics['total_loss']:.5f}, "
                f"Avg Loss: {avg_loss:.5f}, VAE Norm: {metrics['total_norm']:.5f}, "
                f"LR: {scheduler.get_last_lr()[0]:.5f}",
            )
            pbar.update(1)

            # Logging
            if accelerator.is_main_process:
                log_data = {
                    "loss": metrics["total_loss"],
                    "kl_loss": metrics["kl_loss"],
                    "total_norm": metrics["total_norm"],
                    "lr": scheduler.get_last_lr()[0],
                }
                if args.osu_data:
                    log_data.update(
                        {
                            "hit_loss": metrics["hit_loss"],
                            "cursor_loss": metrics["cursor_loss"],
                        },
                    )
                else:
                    log_data["audio_loss"] = metrics["audio_loss"]
                accelerator.log(log_data, step=current_step + 1)

            # Save checkpoint
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

            # Sample and visualize
            if (
                (current_step + 1) % args.sample_every == 0
                and accelerator.is_main_process
                and args.sample_data is not None
                and args.sample_data.exists()
            ):
                print("Sampling...")
                visualize_and_log_sample(args, accelerator, model, step=current_step + 1)

            current_step += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_model_state(accelerator.unwrap_model(model), args.project_dir)


def main() -> None:
    args = ArgumentParser(description="Train OsuFusion AutoEncoder")
    args.add_argument("--project-dir", type=Path, required=True, help="Directory for project outputs")
    args.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing the dataset")
    args.add_argument("--resume", type=Path, default=None, help="Path to resume from a checkpoint")
    args.add_argument("--reset-steps", action="store_true", help="Reset training steps when resuming")
    args.add_argument("--osu-data", action="store_true", help="Use osu! beatmap data")
    args.add_argument("--full-sequence", action="store_true", help="Use full sequence dataset")
    args.add_argument("--max-length", type=int, default=0, help="Maximum length of beatmaps to include")
    args.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "fp8", "bf16"],
        default="bf16",
        help="Mixed precision mode",
    )
    args.add_argument("--full-bf16", action="store_true", help="Use full bfloat16 precision")
    args.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    args.add_argument("--clip-grad-norm", type=float, default=0.0, help="Gradient clipping norm")
    args.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the VAE")
    args.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    args.add_argument("--num-workers", type=int, default=2, help="Number of data loader workers")
    args.add_argument("--total-steps", type=int, default=1_000_000, help="Total number of training steps")
    args.add_argument("--warmup-steps", type=int, default=1_000, help="Number of warmup steps for scheduler")
    args.add_argument("--kl-weight", type=float, default=0.00025, help="Weight for the KL loss")
    args.add_argument("--save-every", type=int, default=1_000, help="Save checkpoint every N steps")
    args.add_argument("--max-num-checkpoints", type=int, default=5, help="Maximum number of checkpoints to keep")
    args.add_argument("--sample-every", type=int, default=1_000, help="Sample and log every N steps")
    args.add_argument("--sample-data", type=Path, default=None, help="Path to sample data for visualization")
    args = args.parse_args()

    train(args)


if __name__ == "__main__":
    main()
