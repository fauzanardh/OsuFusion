import os  # noqa: F401
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
from osu_fusion.data.const import AUDIO_DIM, BEATMAP_DIM
from osu_fusion.data.dataset import FullSequenceDataset, SubsequenceDataset
from osu_fusion.models.autoencoder import AudioAutoEncoder, OsuAutoEncoder

Model = Union[OsuAutoEncoder, AudioAutoEncoder]


def get_total_norm(parameters: List[torch.Tensor], norm_type: float = 2.0) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return 0.0
    norms = []
    for grad in grads:
        norms.append(torch.norm(grad.detach(), norm_type))
    return torch.norm(torch.stack(norms), norm_type).item()


def filter_dataset(dataset: List[Path], max_length: int) -> List[Path]:
    filtered = []
    for path in tqdm(dataset, desc="Filtering dataset...", smoothing=0.0, dynamic_ncols=True):
        beatmap = np.load(path)
        if beatmap["x"].shape[1] <= max_length:
            filtered.append(path)
    return filtered


def cycle(dataloader: DataLoader) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    while True:
        for batch in dataloader:
            yield batch


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
        elif checkpoint.is_file():
            checkpoint.unlink()


# Since we reuse dataset from the previous script, the batch input is batch of (x, a, c)
def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Union[torch.Tensor, torch.Tensor]:
    out_x = []
    out_a = []

    max_length = max([x.shape[1] for x, _, _ in batch])
    for x, a, _ in batch:
        x = F.pad(x, (0, max_length - x.shape[1]), mode="constant", value=-1.0)
        a = F.pad(a, (0, max_length - a.shape[1]), mode="constant", value=-1.0)
        out_x.append(x)
        out_a.append(a)

    out_x = torch.stack(out_x)
    out_a = torch.stack(out_a)
    return out_x, out_a


def sample_step(
    args: ArgumentParser,
    accelerator: Accelerator,
    model: Model,
    step: int,
) -> torch.Tensor:
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    x = np.load(args.sample_data)
    x = x["x"] if args.osu_data else x["a"]
    x = torch.from_numpy(x).unsqueeze(0).to(device=accelerator.device, dtype=dtype)

    model.eval()
    with accelerator.autocast(), torch.torch.inference_mode():
        z, _ = model.encode(x)
        if args.osu_data:
            recon_hit, recon_cursor = model.decode(z, apply_act=True)
            reconstructed = torch.cat([recon_hit, recon_cursor], dim=1)
        else:
            reconstructed = model.decode(z)
    model.train()

    features = BEATMAP_DIM if args.osu_data else AUDIO_DIM
    max_features = min(6, features)

    w, h = reconstructed.shape[-1] // 150, max_features
    fig, axs = plt.subplots(
        h,
        1,
        figsize=(w, h * 8),
        sharex=True,
    )
    for i in range(max_features):
        axs[i].plot(reconstructed[0, i].cpu().to(torch.float32), label="Reconstructed", color="red", linestyle="-")
        axs[i].plot(x[0, i].cpu().to(torch.float32), label="Original", color="blue", linestyle="--")
        axs[i].legend()

    fig.canvas.draw()
    pil_img = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba().tobytes())
    accelerator.log({"generated": wandb.Image(pil_img)}, step=step)
    plt.close(fig)


def save_model_sd(model: Model, project_dir: Path) -> None:
    model_sd = model.state_dict()
    save_file(model_sd, project_dir / "model.safetensors")


def save_checkpoint(
    model: Model,
    optimizer: AdamW,
    scheduler: LambdaLR,
    current_step: int,
    project_dir: Path,
    is_nan: bool = False,
) -> None:
    checkpoint_dir = project_dir / f"checkpoint-{current_step + 1}{'-nan' if is_nan else ''}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to {project_dir}...")
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scheduler_state_dict = scheduler.state_dict()
    rng_state = torch.get_rng_state()

    torch.save(
        {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "rng_state": rng_state,
        },
        checkpoint_dir / "checkpoint.pt",
    )

    del model_state_dict
    del optimizer_state_dict
    del scheduler_state_dict
    torch.cuda.empty_cache()


def load_checkpoint(
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
    except RuntimeError:  # Model changed
        print("Model changed, loading with strict=False...")
        incompatible_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if len(incompatible_keys.missing_keys) > 0:
            print(f"Missing keys: {incompatible_keys.missing_keys}")
        if len(incompatible_keys.unexpected_keys) > 0:
            print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
    if not reset_steps:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    torch.set_rng_state(checkpoint["rng_state"].cpu())
    return 0 if reset_steps else int(checkpoint_path.stem.split("-")[1])


def train(args: ArgumentParser) -> None:  # noqa: C901
    print("Initializing...")
    # Add your own API key here or set it as an environment variable
    # os.environ["WANDB_API_KEY"] = ""
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
    accelerator.init_trackers(
        project_name="OsuFusion-AutoEncoder",
    )

    model = OsuAutoEncoder(32, 128) if args.osu_data else AudioAutoEncoder(16, 128)
    if args.full_bf16:
        model.set_full_bf16()
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

    if args.full_sequence:
        dataset = FullSequenceDataset(dataset=all_maps, segment_sr=False, load_audio=not args.osu_data)
        collator = collate_fn
    else:
        dataset = SubsequenceDataset(dataset=all_maps, segment_sr=False, load_audio=not args.osu_data)
        collator = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collator,
    )

    model, optimizer, scheduler, dataloader = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        dataloader,
    )

    current_step = (
        load_checkpoint(model, optimizer, scheduler, args.resume, args.reset_steps) if args.resume is not None else 0
    )

    model.train()

    if args.resume is None:
        print("Clearing old checkpoints...")
        clear_checkpoints(args.project_dir)

    print("Training...")
    cycle_dataloader = cycle(dataloader)
    losses = []  # Keep track of the last `args.save_every` losses
    with tqdm(
        total=args.total_steps - current_step,
        smoothing=0.0,
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    ) as pbar:
        while current_step < args.total_steps:
            batch_total_loss = 0.0
            batch_kl_loss = 0.0
            total_norm = 0.0
            if args.osu_data:
                batch_hit_loss = 0.0
                batch_cursor_loss = 0.0
            else:
                batch_audio_loss = 0.0
            for _ in range(args.gradient_accumulation_steps):
                batch = next(cycle_dataloader)
                with accelerator.autocast(), accelerator.accumulate(model):
                    try:
                        if collator is None:
                            x, a, _ = batch  # (x, a, c) from Dataset
                        else:
                            x, a = batch
                        model_input = x if args.osu_data else a
                        loss = model(model_input)
                    except AssertionError:
                        continue

                    if args.osu_data:
                        hit_loss, cursor_loss, kl_loss = loss
                        loss = hit_loss + cursor_loss + kl_loss * args.kl_weight
                    else:
                        audio_loss, kl_loss = loss
                        loss = audio_loss + kl_loss * args.kl_weight

                    accelerator.backward(loss)
                    total_norm += get_total_norm(model.parameters()) / args.gradient_accumulation_steps
                    batch_total_loss += loss.item() / args.gradient_accumulation_steps
                    batch_kl_loss += kl_loss.item() / args.gradient_accumulation_steps
                    if args.osu_data:
                        batch_hit_loss += hit_loss.item() / args.gradient_accumulation_steps
                        batch_cursor_loss += cursor_loss.item() / args.gradient_accumulation_steps
                    else:
                        batch_audio_loss += audio_loss.item() / args.gradient_accumulation_steps

                    if args.clip_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            losses.append(batch_total_loss)
            if len(losses) > args.save_every:
                losses.pop(0)

            avg_loss = sum(losses) / len(losses)
            pbar.set_description(
                f"Steps: {current_step + 1}, loss={batch_total_loss:.5f}, avg_loss={avg_loss:.5f}, total_norm={total_norm:.5f}, lr={scheduler.get_last_lr()[0]:.5f}",  # noqa: E501
            )
            pbar.update()

            if accelerator.is_main_process:
                log_dict = {
                    "loss": batch_total_loss,
                    "kl_loss": batch_kl_loss,
                    "total_norm": total_norm,
                    "lr": scheduler.get_last_lr()[0],
                }
                if args.osu_data:
                    log_dict["hit_loss"] = batch_hit_loss
                    log_dict["cursor_loss"] = batch_cursor_loss
                else:
                    log_dict["audio_loss"] = batch_audio_loss
                accelerator.log(log_dict, step=current_step + 1)

            if (current_step + 1) % args.save_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.log({"save_loss": avg_loss}, step=current_step + 1)
                    save_checkpoint(
                        accelerator.unwrap_model(model),
                        optimizer,
                        scheduler,
                        current_step,
                        args.project_dir,
                    )
                    delete_old_checkpoints(args.project_dir, args.max_num_checkpoints)

            if (
                (current_step + 1) % args.sample_every == 0
                and accelerator.is_main_process
                and args.sample_data is not None
                and args.sample_data.exists()
            ):
                print("Sampling...")
                sample_step(
                    args,
                    accelerator,
                    model,
                    step=current_step + 1,
                )

            current_step += 1

    accelerator.wait_for_everyone()
    save_model_sd(accelerator.unwrap_model(model), args.project_dir)


def main() -> None:
    args = ArgumentParser()
    args.add_argument("--project-dir", type=Path)
    args.add_argument("--dataset-dir", type=Path)
    args.add_argument("--resume", type=Path, default=None)
    args.add_argument("--reset-steps", action="store_true")
    args.add_argument("--osu-data", action="store_true")
    args.add_argument("--full-sequence", action="store_true")
    args.add_argument("--max-length", type=int, default=0)
    args.add_argument("--mixed-precision", choices=["no", "fp16", "fp8", "bf16"], default="bf16")
    args.add_argument("--full-bf16", action="store_true")
    args.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args.add_argument("--clip-grad-norm", type=float, default=0.0)
    args.add_argument("--lr", type=float, default=1e-4)
    args.add_argument("--batch-size", type=int, default=16)
    args.add_argument("--num-workers", type=int, default=2)
    args.add_argument("--total-steps", type=int, default=1000000)
    args.add_argument("--warmup-steps", type=int, default=1000)
    args.add_argument("--kl-weight", type=float, default=0.00025)
    args.add_argument("--save-every", type=int, default=1000)
    args.add_argument("--max-num-checkpoints", type=int, default=5)
    args.add_argument("--sample-every", type=int, default=1000)
    args.add_argument("--sample-data", type=Path, default=None)
    args = args.parse_args()

    train(args)


if __name__ == "__main__":
    main()
