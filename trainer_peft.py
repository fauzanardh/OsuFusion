import os  # noqa: F401
import random
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Generator, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import FP8RecipeKwargs, ProjectConfiguration
from diffusers.optimization import get_cosine_schedule_with_warmup
from matplotlib import pyplot as plt
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from torch.nn import functional as F  # noqa: N812
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from osu_fusion.library.dataset import FullSequenceDataset, RandomLengthDataset, SubsequenceDataset
from osu_fusion.library.osu.data.encode import TOTAL_DIM
from osu_fusion.models.diffusion import OsuFusion as DiffusionOsuFusion
from osu_fusion.models.rectified_flow import OsuFusion as RectifiedFlowOsuFusion
from osu_fusion.modules.lora_layers import LoraConv1d
from osu_fusion.scripts.dataset_creator import load_audio, normalize_context

wandb.require("core")
Model = Union[DiffusionOsuFusion, RectifiedFlowOsuFusion]


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
    checkpoints = list((project_dir / "loras").rglob("checkpoint-*"))
    checkpoints.sort(key=lambda path: int(path.stem.split("-")[1]))
    for checkpoint in checkpoints[:-max_num_checkpoints]:
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)


def clear_checkpoints(project_dir: Path) -> None:
    checkpoints = list((project_dir / "loras").rglob("checkpoint-*"))
    for checkpoint in checkpoints:
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
        elif checkpoint.is_file():
            checkpoint.unlink()


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


def sample_step(
    accelerator: Accelerator,
    model: Model,
    audio_path: Path,
    step: int,
) -> torch.Tensor:
    a = load_audio(audio_path)
    # CS, AR, OD, HP, SR
    c = normalize_context(np.array([4.0, 9.5, 9.5, 4.0, 6.0], dtype=np.float32))

    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    a = torch.from_numpy(a).unsqueeze(0).to(device=accelerator.device, dtype=dtype)
    c = torch.from_numpy(c).unsqueeze(0).to(device=accelerator.device, dtype=dtype)

    b, _, n = a.shape

    current_rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    x = torch.randn((b, TOTAL_DIM, n), device=accelerator.device, dtype=dtype)
    torch.set_rng_state(current_rng_state)

    model.eval()
    with accelerator.autocast():
        generated = model.sample(a, c, x, cond_scale=1.0)
    model.train()

    w, h = generated.shape[-1] // 150, TOTAL_DIM
    fig, axs = plt.subplots(
        h,
        1,
        figsize=(w, h * 8),
        sharex=True,
    )
    for feature, ax in zip(generated[0].cpu().to(torch.float32), axs):
        ax.plot(feature)

    accelerator.log({"generated": wandb.Image(fig)}, step=step)
    plt.close(fig)


def load_model_from_sd(model: Model, model_path: Path) -> None:
    sd = load_file(model_path)
    model.load_state_dict(sd)


def get_latest_loras_checkpoint(project_dir: Path) -> Path:
    checkpoints = list((project_dir / "loras").rglob("checkpoint-*"))
    checkpoints.sort(key=lambda path: int(path.stem.split("-")[1]))
    return checkpoints[-1] if checkpoints else None


def save_merged_model_sd(peft_model: PeftModel, project_dir: Path) -> None:
    merged_model = peft_model.merge_and_unload(progressbar=True)
    model_sd = merged_model.state_dict()
    save_file(model_sd, project_dir / "merged_model.safetensors")


def save_peft_checkpoint(
    peft_model: PeftModel,
    optimizer: AdamW,
    scheduler: LambdaLR,
    current_step: int,
    project_dir: Path,
) -> None:
    checkpoint_dir = project_dir / "loras" / f"checkpoint-{current_step + 1}"
    peft_model.save_pretrained(checkpoint_dir)
    optimizer_state_dict = optimizer.state_dict()
    scheduler_state_dict = scheduler.state_dict()
    rng_state = torch.get_rng_state()

    torch.save(
        {
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "rng_state": rng_state,
        },
        checkpoint_dir / "checkpoint.pt",
    )

    del optimizer_state_dict
    del scheduler_state_dict
    torch.cuda.empty_cache()


def load_peft_checkpoint(
    optimizer: AdamW,
    scheduler: LambdaLR,
    checkpoint_path: Path,
    reset_steps: bool,
) -> int:
    checkpoint = torch.load(checkpoint_path / "checkpoint.pt")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if not reset_steps:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    torch.set_rng_state(checkpoint["rng_state"])
    current_step = 0 if reset_steps else int(checkpoint_path.stem.split("-")[1])
    return current_step


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
        project_name="OsuFusion",
    )

    model_class = DiffusionOsuFusion if args.model_type == "diffusion" else RectifiedFlowOsuFusion
    model = model_class(args.model_dim, attn_infini=False)
    if args.full_bf16:
        model.set_full_bf16()
    model.unet.set_gradient_checkpointing(args.gradient_checkpointing)
    load_model_from_sd(model, args.model_path)

    custom_module_mapping = {nn.Conv1d: LoraConv1d}
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["attn.to_q", "attn.to_kv", "attn.linear", "block1.proj", "block2.proj"],
    )
    lora_config._register_custom_module(custom_module_mapping)
    model = get_peft_model(model, lora_config)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=args.total_steps,
        num_warmup_steps=args.warmup_steps,
        num_cycles=0.5,  # half cosine (reach 0 at the end)
    )

    current_step = load_peft_checkpoint(optimizer, scheduler, args.resume, args.reset_steps) if args.resume else 0
    model.print_trainable_parameters()

    print("Loading dataset...")
    all_maps = list(args.dataset_dir.rglob("*.map.npz"))
    if args.max_length > 0:
        all_maps = filter_dataset(all_maps, args.max_length)
    random.shuffle(all_maps)

    if args.full_sequence:
        dataset = FullSequenceDataset(dataset=all_maps)
        collator = collate_fn
    elif args.random_length:
        dataset = RandomLengthDataset(dataset=all_maps)
        collator = collate_fn
    else:
        dataset = SubsequenceDataset(dataset=all_maps)
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
            batch_loss = 0.0
            total_norm = 0.0
            for _ in range(args.gradient_accumulation_steps):
                batch = next(cycle_dataloader)
                with accelerator.autocast() and accelerator.accumulate(model):
                    try:
                        loss = model(*batch)
                    except AssertionError:
                        continue

                    accelerator.backward(loss)
                    total_norm += get_total_norm(model.parameters()) / args.gradient_accumulation_steps
                    batch_loss += loss.item() / args.gradient_accumulation_steps

                    if args.clip_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            losses.append(batch_loss)
            if len(losses) > args.save_every:
                losses.pop(0)

            avg_loss = sum(losses) / len(losses)
            pbar.set_description(
                f"Steps: {current_step + 1}, loss={batch_loss:.5f}, avg_loss={avg_loss:.5f}, total_norm={total_norm:.5f}, lr={scheduler.get_last_lr()[0]:.5f}",  # noqa: E501
            )
            pbar.update()

            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "loss": batch_loss,
                        "total_norm": total_norm,
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=current_step + 1,
                )

            if (current_step + 1) % args.save_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.log({"save_loss": avg_loss}, step=current_step + 1)
                    save_peft_checkpoint(
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
                and args.sample_audio is not None
                and args.sample_audio.exists()
            ):
                print("Sampling...")
                sample_step(
                    accelerator,
                    model,
                    args.sample_audio,
                    step=current_step + 1,
                )

            current_step += 1

    accelerator.wait_for_everyone()
    save_merged_model_sd(accelerator.unwrap_model(model), args.project_dir)


def main() -> None:
    args = ArgumentParser()
    args.add_argument("--project-dir", type=Path)
    args.add_argument("--dataset-dir", type=Path)
    args.add_argument("--model-path", type=Path)
    args.add_argument("--model-type", type=str, default="diffusion", choices=["diffusion", "rectified-flow"])
    args.add_argument("--resume", type=Path, default=None)
    args.add_argument("--reset-steps", action="store_true")
    args.add_argument("--full-sequence", action="store_true")
    args.add_argument("--random-length", action="store_true")
    args.add_argument("--max-length", type=int, default=0)
    args.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16", "fp8"])
    args.add_argument("--full-bf16", action="store_true")
    args.add_argument("--gradient-checkpointing", action="store_true")
    args.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args.add_argument("--clip-grad-norm", type=float, default=0.0)
    args.add_argument("--model-dim", type=int, default=512)
    args.add_argument("--model-attn-heads", type=int, default=8)
    args.add_argument("--model-depth", type=int, default=12)
    args.add_argument("--lr", type=float, default=1e-5)
    args.add_argument("--batch-size", type=int, default=4)
    args.add_argument("--num-workers", type=int, default=2)
    args.add_argument("--total-steps", type=int, default=1000000)
    args.add_argument("--save-every", type=int, default=1000)
    args.add_argument("--max-num-checkpoints", type=int, default=5)
    args.add_argument("--warmup-steps", type=int, default=1000)
    args.add_argument("--sample-every", type=int, default=1000)
    args.add_argument("--sample-audio", type=Path, default=None)
    args = args.parse_args()

    train(args)


if __name__ == "__main__":
    main()
