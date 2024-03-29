import os  # noqa: F401
import random
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from matplotlib import pyplot as plt
from torch.nn import functional as F  # noqa: N812
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from osu_fusion.library.dataset import SubsequenceDataset, normalize_mfcc, sanitize_input
from osu_fusion.library.osu.from_beatmap import TOTAL_DIM
from osu_fusion.models.diffusion import OsuFusion
from osu_fusion.modules.ema import EMA
from osu_fusion.scripts.dataset_creator import load_audio, normalize_context


def delete_old_checkpoints(project_dir: Path, max_num_checkpoints: int) -> None:
    checkpoints = list(project_dir.rglob("checkpoint-*"))
    checkpoints.sort(key=lambda path: int(path.stem.split("-")[1]))
    for checkpoint in checkpoints[:-max_num_checkpoints]:
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)


def clear_checkpoints(project_dir: Path) -> None:
    checkpoints = list(project_dir.rglob("checkpoint-*"))
    for checkpoint in checkpoints:
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_x = []
    out_a = []
    out_c = []
    orig_len = []

    max_length = max([x.shape[1] for x, _, _ in batch])
    for x, a, c in batch:
        orig_len.append(x.shape[1])
        x = F.pad(x, (0, max_length - x.shape[1]), mode="constant", value=-1.0)
        a = F.pad(a, (0, max_length - a.shape[1]), mode="constant", value=-1.0)
        out_x.append(x)
        out_a.append(a)
        out_c.append(c)

    out_x = torch.stack(out_x)
    out_a = torch.stack(out_a)
    out_c = torch.stack(out_c)
    orig_len = torch.tensor(orig_len)
    return out_x, out_a, out_c, orig_len


def train_step(
    accelerator: Accelerator,
    model: OsuFusion,
    ema: EMA,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    x, a, c = batch
    with accelerator.autocast():
        try:
            loss = model(x, a, c)
        except AssertionError:
            return None
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    ema.update()
    return loss


def sample_step(
    accelerator: Accelerator,
    model: OsuFusion,
    ema: EMA,
    audio_path: Path,
    audio_bpm: float,
    step: int,
) -> torch.Tensor:
    a = load_audio(audio_path)
    c = normalize_context(np.array([5.0, 4.0, 9.5, 9.5, 8.0, audio_bpm], dtype=np.float32))

    a = torch.from_numpy(a).unsqueeze(0).to(accelerator.device)
    c = torch.from_numpy(c).unsqueeze(0).to(accelerator.device)

    a = sanitize_input(normalize_mfcc(a))
    c = sanitize_input(c)

    b, _, n = a.shape

    current_rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    x = torch.randn((b, TOTAL_DIM, n), device=accelerator.device)
    torch.set_rng_state(current_rng_state)

    model.eval()
    with torch.no_grad() and accelerator.autocast():
        generated_non_ema = model.sample(a, c, x, cond_scale=1.0)
    non_ema_unet = model.unet
    model.unet = ema.model
    with torch.no_grad() and accelerator.autocast():
        generated_ema = model.sample(a, c, x, cond_scale=1.0)
    model.unet = non_ema_unet
    model.train()

    w, h = generated_non_ema.shape[-1] // 150, 7
    fig_non_ema, axs = plt.subplots(
        h,
        1,
        figsize=(w, h * 8),
        sharex=True,
    )
    for feature, ax in zip(generated_non_ema[0].cpu(), axs):
        ax.plot(feature)

    w, h = generated_ema.shape[-1] // 150, 7
    fig_ema, axs = plt.subplots(
        h,
        1,
        figsize=(w, h * 8),
        sharex=True,
    )
    for feature, ax in zip(generated_ema[0].cpu(), axs):
        ax.plot(feature)

    accelerator.log({"generated_non-ema": wandb.Image(fig_non_ema), "generated_ema": wandb.Image(fig_ema)}, step=step)
    plt.close(fig_non_ema)


def train(args: ArgumentParser) -> None:  # noqa: C901
    print("Initializing...")
    # Add your own API key here or set it as an environment variable
    # os.environ["WANDB_API_KEY"] = ""
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=args.project_dir,
            automatic_checkpoint_naming=True,
        ),
        log_with="wandb",
    )
    accelerator.init_trackers(
        project_name="OsuFusion",
    )

    model = OsuFusion(args.model_dim)
    model.unet.set_gradient_checkpointing(args.gradient_checkpointing)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.total_steps,
        pct_start=args.pct_start,
    )
    ema = EMA(model.unet)

    print("Loading dataset...")
    all_maps = list(args.dataset_dir.rglob("*.map.npz"))
    random.shuffle(all_maps)
    dataset = SubsequenceDataset(dataset=all_maps)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    model, ema, optimizer, scheduler, dataloader = accelerator.prepare(
        model,
        ema,
        optimizer,
        scheduler,
        dataloader,
    )
    model.train()

    print("Clearing old checkpoints...")
    clear_checkpoints(args.project_dir)

    print("Training...")
    iter_dataloader = iter(dataloader)
    losses = []  # Keep track of the last `args.save_every` losses
    with tqdm(total=args.total_steps, smoothing=0.0, disable=not accelerator.is_local_main_process) as pbar:
        for step in range(args.total_steps):
            batch = None
            while batch is None:
                try:
                    batch = next(iter_dataloader)
                except Exception:
                    iter_dataloader = iter(dataloader)

            loss = train_step(accelerator, model, ema, optimizer, scheduler, batch)
            if loss is None:
                continue
            if torch.isnan(loss):
                # Save the model before exiting
                accelerator.wait_for_everyone()
                accelerator.save_model(model, args.project_dir / f"checkpoint-{step + 1}-NaN")
                msg = "NaN loss encountered"
                raise RuntimeError(msg)
            loss = loss.item()
            losses.append(loss)

            if len(losses) > args.save_every:
                losses.pop(0)

            avg_loss = sum(losses) / len(losses)
            pbar.set_description(
                f"Steps: {step + 1}, loss={loss:.5f}, avg_loss={avg_loss:.5f}, lr={scheduler.get_last_lr()[0]:.5f}",
            )
            pbar.update()

            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "loss": loss,
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=step + 1,
                )

            if (step + 1) % args.save_every == 0:
                accelerator.wait_for_everyone()
                save_dir = args.project_dir / f"checkpoint-{step + 1}"
                accelerator.save_model(model, save_dir)
                accelerator.save_model(ema, save_dir / "ema")

                if accelerator.is_main_process:
                    accelerator.log({"save_loss": avg_loss}, step=step + 1)
                    delete_old_checkpoints(args.project_dir, args.max_num_checkpoints)

            if (
                (step + 1) % args.sample_every == 0
                and accelerator.is_main_process
                and args.sample_audio is not None
                and args.sample_audio.exists()
            ):
                print("Sampling...")
                sample_step(
                    accelerator,
                    model,
                    ema,
                    args.sample_audio,
                    args.sample_audio_bpm,
                    step=step + 1,
                )

    accelerator.wait_for_everyone()
    accelerator.save_model(model, args.project_dir / "checkpoint-final")


def main() -> None:
    args = ArgumentParser()
    args.add_argument("--project-dir", type=Path)
    args.add_argument("--dataset-dir", type=Path)
    args.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    args.add_argument("--gradient-checkpointing", action="store_true")
    args.add_argument("--model-dim", type=int, default=128)
    args.add_argument("--lr", type=float, default=1e-5)
    args.add_argument("--batch-size", type=int, default=16)
    args.add_argument("--num-workers", type=int, default=2)
    args.add_argument("--total-steps", type=int, default=500000)
    args.add_argument("--save-every", type=int, default=1000)
    args.add_argument("--max-num-checkpoints", type=int, default=5)
    args.add_argument("--pct-start", type=float, default=0.002)
    args.add_argument("--sample-every", type=int, default=1000)
    args.add_argument("--sample-audio", type=Path, default=None)
    args.add_argument("--sample-audio-bpm", type=float, default=180.0)
    args = args.parse_args()

    train(args)


if __name__ == "__main__":
    main()
