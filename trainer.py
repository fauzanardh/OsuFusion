import os  # noqa: F401
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from osu_fusion.library.dataset import SubsequenceDataset
from osu_fusion.models.diffusion import OsuFusion


def delete_old_checkpoints(project_dir: Path, max_num_checkpoints: int) -> None:
    checkpoints = list(project_dir.rglob("checkpoint-*"))
    checkpoints.sort(key=lambda path: int(path.stem.split("-")[1]))
    for checkpoint in checkpoints[:-max_num_checkpoints]:
        checkpoint.rmdir()


def train_step(
    accelerator: Accelerator,
    model: OsuFusion,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    x, a, c = batch
    with accelerator.autocast():
        t = model.scheduler.sample_random_times(x.shape[0], x.device)
        loss = model(x, a, t, c)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    return loss.item()


def train(args: ArgumentParser) -> None:
    print("Initializing...")
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=args.project_dir,
            automatic_checkpoint_naming=True,
        ),
    )
    model = OsuFusion(args.model_dim)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.total_steps,
        pct_start=args.pct_start,
    )

    print("Loading dataset...")
    all_maps = list(args.dataset_dir.rglob("*.map.npz"))
    random.shuffle(all_maps)
    dataset = SubsequenceDataset(dataset=all_maps, sequence_length=args.sequence_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model, optimizer, scheduler, dataloader = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        dataloader,
    )
    model.train()

    # Add your own API key here or set it as an environment variable
    # os.environ["WANDB_API_KEY"] = ""
    wandb.init(
        project="OsuFusion",
        settings=wandb.Settings(start_method="thread"),
    )

    print("Training...")
    iter_dataloader = iter(dataloader)
    losses = []  # Keep track of the last `args.save_every` losses
    with tqdm(total=args.total_steps, smoothing=1.0) as pbar:
        for step in range(args.total_steps):
            batch = None
            while batch is None:
                try:
                    batch = next(iter_dataloader)
                except Exception:
                    iter_dataloader = iter(dataloader)

            loss = train_step(accelerator, model, optimizer, scheduler, batch)
            losses.append(loss)

            if len(losses) > args.save_every:
                losses.pop(0)

            avg_loss = sum(losses) / len(losses)
            pbar.set_description(
                f"Steps: {step + 1}, loss={loss:.5f}, avg_loss={avg_loss:.5f}, lr={scheduler.get_last_lr()[0]:.5f}",
            )
            pbar.update()

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "loss": loss,
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=step + 1,
                )

            if (step + 1) % args.save_every == 0:
                accelerator.wait_for_everyone()
                accelerator.save_model(model, f"{args.project_dir}checkpoint-{step + 1}")

                if accelerator.is_main_process:
                    wandb.log({"save_loss": avg_loss}, step=step + 1)
                    delete_old_checkpoints(args.project_dir, args.max_num_checkpoints)


def main() -> None:
    args = ArgumentParser()
    args.add_argument("--project-dir", type=Path)
    args.add_argument("--dataset-dir", type=Path)
    args.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    args.add_argument("--model-dim", type=int, default=128)
    args.add_argument("--lr", type=float, default=1e-4)
    args.add_argument("--total-steps", type=int, default=100000)
    args.add_argument("--save-every", type=int, default=1000)
    args.add_argument("--max-num-checkpoints", type=int, default=5)
    args.add_argument("--pct-start", type=float, default=0.01)
    args.add_argument("--batch-size", type=int, default=64)
    args.add_argument("--num-workers", type=int, default=2)
    args.add_argument("--sequence-length", type=int, default=2048)
    args = args.parse_args()

    train(args)


if __name__ == "__main__":
    main()
