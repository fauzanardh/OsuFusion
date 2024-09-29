import random
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Generator, List, Tuple, Union

import numpy as np
import torch
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

import wandb
from osu_fusion.data.const import BEATMAP_DIM
from osu_fusion.data.dataset import FullSequenceDataset, SubsequenceDataset
from osu_fusion.data.prepare_data import load_audio, normalize_context
from osu_fusion.models.diffusion import OsuFusion as DiffusionOsuFusion
from osu_fusion.models.rectified_flow import OsuFusion as RectifiedFlowOsuFusion

Model = Union[DiffusionOsuFusion, RectifiedFlowOsuFusion]


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_x = []
    out_a = []
    out_c = []
    orig_len = []

    max_length = max(x.shape[1] for x, _, _ in batch)
    for x, a, c in batch:
        orig_len.append(x.shape[1])
        x_padded = F.pad(x, (0, max_length - x.shape[1]), value=-1.0)
        a_padded = F.pad(a, (0, max_length - a.shape[1]), value=-23.0)
        out_x.append(x_padded)
        out_a.append(a_padded)
        out_c.append(c)

    out_x = torch.stack(out_x)
    out_a = torch.stack(out_a)
    out_c = torch.stack(out_c)
    orig_len = torch.tensor(orig_len)
    return out_x, out_a, out_c, orig_len


def visualize_and_log_sample(
    accelerator: Accelerator,
    model: Model,
    audio_path: Path,
    step: int,
) -> None:
    a = load_audio(audio_path)
    c = normalize_context(np.array([4.0, 9.5, 9.5, 4.0, 6.0], dtype=np.float32))

    dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float16,  # Assuming fp8 uses float16 for storage
    }.get(accelerator.mixed_precision, torch.float32)

    a_tensor = torch.from_numpy(a).unsqueeze(0).to(accelerator.device, dtype)
    c_tensor = torch.from_numpy(c).unsqueeze(0).to(accelerator.device, dtype)

    b, _, n = a_tensor.shape

    current_rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    x = torch.randn((b, BEATMAP_DIM, n), device=accelerator.device, dtype=dtype)
    torch.set_rng_state(current_rng_state)

    model.eval()
    with torch.inference_mode(), accelerator.autocast():
        generated = model.sample(a_tensor, c_tensor, x, cond_scale=1.0)
    model.train()

    width, height = generated.shape[-1] // 150, BEATMAP_DIM
    fig, axs = plt.subplots(height, 1, figsize=(width, height * 8), sharex=True)
    for i in range(BEATMAP_DIM):
        axs[i].plot(generated[0, i].cpu(), label="Reconstructed", color="red")
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

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except RuntimeError:
        print("Model changed, loading with strict=False...")
        incompatible_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if incompatible_keys.missing_keys:
            print(f"Missing keys: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")

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
    accelerator.init_trackers(project_name="OsuFusion")

    # Initialize model
    model_class = DiffusionOsuFusion if args.model_type == "diffusion" else RectifiedFlowOsuFusion
    model = model_class(
        model_dim=args.model_dim,
        attn_heads=args.model_attn_heads,
        depth=args.model_depth,
    )
    if args.full_bf16:
        model.set_full_bf16()
    model.unet.set_gradient_checkpointing(args.gradient_checkpointing)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=args.total_steps,
        num_warmup_steps=args.warmup_steps,
        num_cycles=0.5,  # half cosine (reach 0 at the end)
    )

    print("Loading dataset...")
    all_maps = list(args.dataset_dir.rglob("*.map.npz"))
    if args.max_length > 0:
        all_maps = filter_dataset(all_maps, args.max_length)
    random.shuffle(all_maps)

    dataset_cls = FullSequenceDataset if args.full_sequence else SubsequenceDataset
    dataset = dataset_cls(dataset=all_maps)
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

    model.train()
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
            metrics = {
                "loss": 0.0,
                "total_norm": 0.0,
            }

            for _ in range(args.gradient_accumulation_steps):
                batch = next(dataloader_cycle)
                with accelerator.autocast(), accelerator.accumulate(model):
                    try:
                        loss = model(*batch)
                    except AssertionError:
                        print(f"AssertionError encountered at step {current_step + 1}, skipping batch.")
                        continue

                    accelerator.backward(loss)
                    metrics["total_norm"] += get_total_norm(model.parameters()) / args.gradient_accumulation_steps
                    metrics["loss"] += loss.item() / args.gradient_accumulation_steps

                    if args.clip_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            # Update loss history and progress bar
            loss_history.append(metrics["loss"])
            if len(loss_history) > args.save_every:
                loss_history.pop(0)
            avg_loss = sum(loss_history) / len(loss_history)

            pbar.set_description(
                f"Steps: {current_step + 1}, Loss: {metrics['loss']:.5f}, "
                f"Avg Loss: {avg_loss:.5f}, Total Norm: {metrics['total_norm']:.5f}, "
                f"LR: {scheduler.get_last_lr()[0]:.5f}",
            )
            pbar.update(1)

            # Logging
            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "loss": metrics["loss"],
                        "total_norm": metrics["total_norm"],
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=current_step + 1,
                )

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

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_model_state(accelerator.unwrap_model(model), args.project_dir)


def main() -> None:
    args = ArgumentParser(description="Train OsuFusion U-Net Backbone")
    args.add_argument("--project-dir", type=Path, required=True, help="Directory for project outputs")
    args.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing the dataset")
    args.add_argument(
        "--model-type",
        type=str,
        default="diffusion",
        choices=["diffusion", "rectified-flow"],
        help="Type of model to train",
    )
    args.add_argument("--resume", type=Path, default=None, help="Path to resume from a checkpoint")
    args.add_argument("--reset-steps", action="store_true", help="Reset training steps when resuming")
    args.add_argument("--full-sequence", action="store_true", help="Use full sequence dataset")
    args.add_argument("--max-length", type=int, default=0, help="Maximum length of beatmaps to include")
    args.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16", "fp8"],
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
    args.add_argument("--model-dim", type=int, default=512, help="Dimension of the model")
    args.add_argument("--model-attn-heads", type=int, default=8, help="Number of attention heads")
    args.add_argument("--model-depth", type=int, default=12, help="Depth of the model")
    args.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer")
    args.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    args.add_argument("--num-workers", type=int, default=2, help="Number of data loader workers")
    args.add_argument("--total-steps", type=int, default=1_000_000, help="Total number of training steps")
    args.add_argument("--warmup-steps", type=int, default=1_000, help="Number of warmup steps for scheduler")
    args.add_argument("--save-every", type=int, default=1_000, help="Save checkpoint every N steps")
    args.add_argument("--max-num-checkpoints", type=int, default=5, help="Maximum number of checkpoints to keep")
    args.add_argument("--sample-every", type=int, default=1_000, help="Sample and log every N steps")
    args.add_argument("--sample-audio", type=Path, default=None, help="Path to sample audio for visualization")
    args = args.parse_args()

    train(args)


if __name__ == "__main__":
    main()