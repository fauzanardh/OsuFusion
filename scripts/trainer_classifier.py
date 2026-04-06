import random
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
from safetensors.torch import save_file
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from osu_fusion.data.const import MAX_LENGTH_FRAMES
from osu_fusion.data.descriptors import DESCRIPTOR_ANCESTORS, NUM_DESCRIPTORS
from osu_fusion.data.encode import SEQ_DIM, SequenceEncoding
from osu_fusion.models.classifier import ClassifierConfig_S, ClassifierConfig_M, ClassifierConfig_L, OsuFusionClassifier


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
) -> torch.Tensor:
    """
    "Asymmetric Loss For Multi-Label Classification" (Ridnik et al., 2021).
    """
    xs_pos = torch.sigmoid(logits)
    xs_neg = 1.0 - xs_pos

    if clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)

    loss_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
    loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=1e-8))

    loss = loss_pos + loss_neg

    if gamma_neg > 0 or gamma_pos > 0:
        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
        loss = loss * (1.0 - pt) ** gamma

    return -loss.mean()


def build_hierarchy_index_tensors(
    ancestors_map: Dict[int, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    children = []
    parents = []
    for child_idx, ancestor_indices in ancestors_map.items():
        for parent_idx in ancestor_indices:
            if parent_idx != child_idx:
                children.append(child_idx)
                parents.append(parent_idx)
    return torch.tensor(children, dtype=torch.long), torch.tensor(parents, dtype=torch.long)


def hierarchy_consistency_loss(
    logits: torch.Tensor,
    child_indices: torch.Tensor,
    parent_indices: torch.Tensor,
    weight: float = 0.5,
) -> torch.Tensor:
    if child_indices.numel() == 0:
        return logits.new_tensor(0.0)

    probs = torch.sigmoid(logits)
    child_probs = probs[:, child_indices]  # (B, num_pairs)
    parent_probs = probs[:, parent_indices]  # (B, num_pairs)
    violations = F.relu(child_probs - parent_probs)
    return weight * (violations**2).mean()


def get_total_norm(parameters: List[torch.Tensor], norm_type: float = 2.0) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type).item()


def manage_checkpoints(project_dir: Path, max_num_checkpoints: int) -> None:
    checkpoints = sorted(project_dir.rglob("checkpoint-*"), key=lambda p: int(p.stem.split("-")[1]))
    for checkpoint in checkpoints[:-max_num_checkpoints]:
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
        else:
            checkpoint.unlink()


class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self: "ClassifierDataset", map_files: List[Path], augment: bool = True) -> None:
        super().__init__()
        self.map_files = map_files
        self.augment = augment

    def __len__(self: "ClassifierDataset") -> int:
        return len(self.map_files)

    @staticmethod
    def _augment(x: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            x[:, SequenceEncoding.X] = -x[:, SequenceEncoding.X]

        if random.random() < 0.5:
            x[:, SequenceEncoding.Y] = -x[:, SequenceEncoding.Y]

        if random.random() < 0.5:
            T = x.shape[0]
            crop_ratio = random.uniform(0.75, 1.0)
            crop_len = max(1, int(T * crop_ratio))
            start = random.randint(0, T - crop_len)
            x = x[start : start + crop_len]
            a = a[start : start + crop_len]

        # if random.random() < 0.3:
        #     T = x.shape[0]
        #     scale = random.uniform(0.8, 1.5)
        #     new_T = max(1, int(T * scale))
        #     x = F.interpolate(x.unsqueeze(0).permute(0, 2, 1), size=new_T, mode="linear", align_corners=False)
        #     x = x.permute(0, 2, 1).squeeze(0)
        #     a = F.interpolate(a.unsqueeze(0).permute(0, 2, 1), size=new_T, mode="linear", align_corners=False)
        #     a = a.permute(0, 2, 1).squeeze(0)

        # if random.random() < 0.5:
        #     a = a + torch.randn_like(a) * 0.05

        # if random.random() < 0.3:
        #     x_offset = random.uniform(-0.05, 0.05)
        #     y_offset = random.uniform(-0.05, 0.05)
        #     x[:, SequenceEncoding.X] = (x[:, SequenceEncoding.X] + x_offset).clamp(-1.0, 1.0)
        #     x[:, SequenceEncoding.Y] = (x[:, SequenceEncoding.Y] + y_offset).clamp(-1.0, 1.0)

        return x, a

    def __getitem__(
        self: "ClassifierDataset",
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        map_file = self.map_files[index]
        with h5py.File(map_file, "r") as f:
            x = torch.from_numpy(f["x"][:]).float()
            c = torch.from_numpy(f["c"][:]).float()
            spec_path = f["spec_path"][()].decode("utf-8")

            # Load descriptor indices and convert to multi-hot
            if "descriptor_indices" in f:
                desc_idx = f["descriptor_indices"][:]
                descriptors = torch.zeros(NUM_DESCRIPTORS, dtype=torch.float32)
                for idx in desc_idx:
                    if 0 <= idx < NUM_DESCRIPTORS:
                        descriptors[idx] = 1.0
            else:
                descriptors = torch.zeros(NUM_DESCRIPTORS, dtype=torch.float32)

        audio_file = map_file.parent.parent.parent / spec_path
        with h5py.File(audio_file, "r") as audio_data:
            a = torch.from_numpy(audio_data["a"][:]).float()

        if torch.isnan(x).any() or torch.isnan(c).any() or torch.isnan(a).any():
            msg = f"NaN in {map_file}"
            raise ValueError(msg)

        if self.augment:
            x, a = self._augment(x, a)

        return x, a, c, descriptors


def classifier_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_lens = torch.tensor([x.shape[0] for x, _, _, _ in batch], dtype=torch.int32)
    max_len = max(x.shape[0] for x, _, _, _ in batch)

    padded_x = []
    padded_a = []
    for x, a, _, _ in batch:
        n_pad = max_len - x.shape[0]
        if n_pad > 0:
            x = F.pad(x, (0, 0, 0, n_pad))
            a = F.pad(a, (0, 0, 0, n_pad))
        padded_x.append(x)
        padded_a.append(a)

    out_x = torch.stack(padded_x)
    out_a = torch.stack(padded_a)
    out_c = torch.stack([c for _, _, c, _ in batch])
    out_tags = torch.stack([t for _, _, _, t in batch])
    return out_x, out_a, out_c, out_tags, orig_lens


def filter_maps(maps: List[Path], max_length: int = 0) -> List[Path]:
    filtered = []
    for path in tqdm(maps, desc="Filtering dataset...", dynamic_ncols=True):
        try:
            with h5py.File(path, "r") as f:
                x_len = f["x"].shape[0]
                if x_len > MAX_LENGTH_FRAMES:
                    continue
                if max_length > 0 and x_len > max_length:
                    continue
                if f["x"].shape[1] != SEQ_DIM:
                    continue

                spec_path = f["spec_path"][()].decode("utf-8")
                audio_file = path.parent.parent.parent / spec_path
                if not audio_file.exists():
                    continue

                if "mapper_indices" not in f:
                    continue

                if "descriptor_indices" not in f:
                    continue
            filtered.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
    print(f"Filtered dataset: {len(filtered)}/{len(maps)} maps")
    return filtered


def save_model_state(model: OsuFusionClassifier, project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), project_dir / "classifier.safetensors")


def save_training_checkpoint(
    model: OsuFusionClassifier,
    optimizer: AdamW,
    scheduler: LambdaLR,
    current_step: int,
    project_dir: Path,
) -> None:
    checkpoint_dir = project_dir / f"checkpoint-{current_step + 1}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
    }
    torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")


def load_training_checkpoint(
    model: OsuFusionClassifier,
    optimizer: AdamW,
    scheduler: LambdaLR,
    checkpoint_path: Path,
    reset_steps: bool = False,
) -> int:
    print(f"Loading checkpoint from {checkpoint_path}...")
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path / "checkpoint.pt", map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if not reset_steps:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    torch.set_rng_state(checkpoint["rng_state"].cpu())
    return 0 if reset_steps else int(checkpoint_path.stem.split("-")[1])


MODEL_CONFIGS = {"s": ClassifierConfig_S, "m": ClassifierConfig_M, "l": ClassifierConfig_L}


def train(args: ArgumentParser) -> None:  # noqa: C901
    start_time = time.time()
    print("Initializing...")
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=ProjectConfiguration(project_dir=args.project_dir, automatic_checkpoint_naming=True),
        log_with="wandb",
    )
    accelerator.init_trackers(project_name="OsuFusion-Classifier")

    print("Loading dataset...")
    all_maps = list(args.dataset_dir.rglob("*.map.h5"))
    all_maps = filter_maps(all_maps, max_length=args.max_length)

    config = MODEL_CONFIGS[args.model_size]
    model = OsuFusionClassifier(**asdict(config))
    if args.gradient_checkpointing:
        model.set_gradient_checkpointing(True)

    # Split into train/val (95/5)
    np.random.seed(42)
    np.random.shuffle(all_maps)
    split_idx = int(len(all_maps) * 0.95)
    train_maps = all_maps[:split_idx]
    val_maps = all_maps[split_idx:]
    print(f"Train: {len(train_maps)}, Val: {len(val_maps)}")

    train_dataset = ClassifierDataset(train_maps, augment=True)
    val_dataset = ClassifierDataset(val_maps, augment=False)

    hier_child_idx, hier_parent_idx = build_hierarchy_index_tensors(DESCRIPTOR_ANCESTORS)
    print(f"Hierarchy consistency pairs: {hier_child_idx.numel()}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        collate_fn=classifier_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=classifier_collate_fn,
    )

    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps))
    total_steps = steps_per_epoch * args.epochs

    parameters = list(model.parameters())
    print(f"Number of trainable parameters: {sum(p.numel() for p in parameters):,}")
    optimizer = AdamW(parameters, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=args.warmup_steps,
        num_cycles=0.5,
    )

    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
    )

    current_step = (
        load_training_checkpoint(model, optimizer, scheduler, args.resume, args.reset_steps) if args.resume else 0
    )

    device = next(model.parameters()).device
    hier_child_idx = hier_child_idx.to(device)
    hier_parent_idx = hier_parent_idx.to(device)

    model.train()
    print("Starting training...")
    loss_history = []

    with tqdm(
        total=total_steps - current_step,
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    ) as pbar:
        for epoch in range(args.epochs):
            epoch_loss_history = []
            accum_loss = 0.0

            for batch in train_dataloader:
                x, a, c, tags, orig_lens = batch

                with accelerator.autocast(), accelerator.accumulate(model):
                    logits = model(x, a, c, orig_lens=orig_lens)
                    asl = asymmetric_loss(
                        logits,
                        tags,
                        gamma_neg=args.asl_gamma_neg,
                        gamma_pos=args.asl_gamma_pos,
                        clip=args.asl_clip,
                    )
                    hier_loss = hierarchy_consistency_loss(
                        logits,
                        hier_child_idx,
                        hier_parent_idx,
                        weight=args.hierarchy_weight,
                    )
                    loss = asl + hier_loss

                    accelerator.backward(loss)
                    accum_loss += loss.item() / args.gradient_accumulation_steps

                    if accelerator.sync_gradients and args.clip_grad_norm > 0.0:
                        accelerator.clip_grad_norm_(parameters, args.clip_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    epoch_loss_history.append(accum_loss)
                    loss_history.append(accum_loss)
                    avg_loss = sum(epoch_loss_history) / len(epoch_loss_history)

                    pbar.set_description(
                        f"Ep {epoch + 1}/{args.epochs} | Step {current_step + 1} | "
                        f"Loss {accum_loss:.4f} | Avg {avg_loss:.4f}",
                    )
                    pbar.update(1)

                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "train_loss": accum_loss,
                                "lr": scheduler.get_last_lr()[0],
                            },
                            step=current_step + 1,
                        )

                    if (current_step + 1) % args.save_every == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            save_training_checkpoint(
                                accelerator.unwrap_model(model),
                                optimizer,
                                scheduler,
                                current_step,
                                args.project_dir,
                            )
                            manage_checkpoints(args.project_dir, args.max_num_checkpoints)

                    if (current_step + 1) % args.eval_every == 0:
                        model.eval()
                        val_losses = []
                        all_preds = []
                        all_tags = []
                        with torch.no_grad():
                            for val_batch in val_dataloader:
                                vx, va, vc, vtags, vorig_lens = val_batch
                                with accelerator.autocast():
                                    vlogits = model(vx, va, vc, orig_lens=vorig_lens)
                                    vasl = asymmetric_loss(
                                        vlogits,
                                        vtags,
                                        gamma_neg=args.asl_gamma_neg,
                                        gamma_pos=args.asl_gamma_pos,
                                        clip=args.asl_clip,
                                    )
                                    vhier = hierarchy_consistency_loss(
                                        vlogits,
                                        hier_child_idx,
                                        hier_parent_idx,
                                        weight=args.hierarchy_weight,
                                    )
                                    vloss = vasl + vhier
                                val_losses.append(vloss.item())
                                vpreds = (torch.sigmoid(vlogits) > 0.5).float()
                                all_preds.append(vpreds.cpu())
                                all_tags.append(vtags.cpu())

                        val_avg = sum(val_losses) / len(val_losses) if val_losses else 0.0

                        if accelerator.is_main_process and all_preds:
                            all_preds_t = torch.cat(all_preds, dim=0)
                            all_tags_t = torch.cat(all_tags, dim=0)

                            tp = (all_preds_t * all_tags_t).sum(0)
                            fp = (all_preds_t * (1 - all_tags_t)).sum(0)
                            fn = ((1 - all_preds_t) * all_tags_t).sum(0)

                            precision = tp / (tp + fp + 1e-8)
                            recall = tp / (tp + fn + 1e-8)
                            f1 = 2 * precision * recall / (precision + recall + 1e-8)

                            tag_has_positives = all_tags_t.sum(0) > 0
                            macro_precision = precision[tag_has_positives].mean().item()
                            macro_recall = recall[tag_has_positives].mean().item()
                            macro_f1 = f1[tag_has_positives].mean().item()

                            micro_precision = tp.sum().item() / (tp.sum().item() + fp.sum().item() + 1e-8)
                            micro_recall = tp.sum().item() / (tp.sum().item() + fn.sum().item() + 1e-8)
                            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)

                            exact_match = (all_preds_t == all_tags_t).all(dim=1).float().mean().item()

                            accelerator.log(
                                {
                                    "val_loss": val_avg,
                                    "val_macro_f1": macro_f1,
                                    "val_macro_precision": macro_precision,
                                    "val_macro_recall": macro_recall,
                                    "val_micro_f1": micro_f1,
                                    "val_micro_precision": micro_precision,
                                    "val_micro_recall": micro_recall,
                                    "val_exact_match": exact_match,
                                },
                                step=current_step + 1,
                            )
                            print(
                                f"\n  Val loss: {val_avg:.4f} | "
                                f"Macro F1: {macro_f1:.3f} (P:{macro_precision:.3f} R:{macro_recall:.3f}) | "
                                f"Micro F1: {micro_f1:.3f} | "
                                f"Exact: {exact_match:.3f}",
                            )
                        model.train()

                    current_step += 1
                    accum_loss = 0.0

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_model_state(accelerator.unwrap_model(model), args.project_dir)
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Final avg loss: {sum(loss_history) / len(loss_history):.5f}" if loss_history else "No loss recorded")


def main() -> None:
    args = ArgumentParser(description="Train OsuFusion Classifier")
    args.add_argument("--project-dir", type=Path, required=True, help="Directory for project outputs")
    args.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing the dataset")
    args.add_argument("--model-size", type=str, default="s", choices=["s", "m", "l"], help="Model size")
    args.add_argument("--resume", type=Path, default=None, help="Path to resume from a checkpoint")
    args.add_argument("--reset-steps", action="store_true", help="Reset training steps when resuming")
    args.add_argument("--max-length", type=int, default=0, help="Maximum length of beatmaps to include")
    args.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    args.add_argument("--gradient-checkpointing", action="store_true")
    args.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args.add_argument("--clip-grad-norm", type=float, default=1.0)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--batch-size", type=int, default=32)
    args.add_argument("--num-workers", type=int, default=2)
    args.add_argument("--epochs", type=int, default=10)
    args.add_argument("--warmup-steps", type=int, default=1000)
    args.add_argument("--save-every", type=int, default=2000)
    args.add_argument("--eval-every", type=int, default=1000)
    args.add_argument("--max-num-checkpoints", type=int, default=3)
    args.add_argument("--asl-gamma-neg", type=float, default=3.0, help="ASL gamma for negatives")
    args.add_argument("--asl-gamma-pos", type=float, default=1.0, help="ASL gamma for positives")
    args.add_argument("--asl-clip", type=float, default=0.05, help="ASL probability clipping for negatives")
    args.add_argument("--hierarchy-weight", type=float, default=0.5, help="Weight for hierarchy consistency penalty")
    args = args.parse_args()
    train(args)


if __name__ == "__main__":
    main()
