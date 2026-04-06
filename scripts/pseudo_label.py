from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from osu_fusion.data.descriptors import DESCRIPTOR_TAGS, NUM_DESCRIPTORS
from osu_fusion.data.encode import SEQ_DIM
from osu_fusion.models.classifier import (
    ClassifierConfig_L,
    ClassifierConfig_M,
    ClassifierConfig_S,
    OsuFusionClassifier,
)

MODEL_CONFIGS = {"s": ClassifierConfig_S, "m": ClassifierConfig_M, "l": ClassifierConfig_L}


def load_classifier(checkpoint_path: str, model_size: str, device: torch.device) -> OsuFusionClassifier:
    config = MODEL_CONFIGS[model_size]
    model = OsuFusionClassifier(**asdict(config))

    if checkpoint_path.endswith(".pt"):
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict)

    return model.eval().to(device)


def pseudo_label_map(  # noqa: C901
    map_path: Path,
    model: OsuFusionClassifier,
    device: torch.device,
    pos_threshold: float = 0.8,
    neg_threshold: float = 0.2,
    overwrite_real: bool = False,
) -> dict:
    try:
        with h5py.File(map_path, "r") as f:
            x = torch.from_numpy(f["x"][:]).float()
            c = torch.from_numpy(f["c"][:]).float()

            # Load existing descriptor indices and convert to multi-hot
            if "descriptor_indices" in f:
                raw_indices = f["descriptor_indices"][:]
                existing_labels = np.zeros(NUM_DESCRIPTORS, dtype=np.float32)
                for idx in raw_indices:
                    if 0 <= idx < NUM_DESCRIPTORS:
                        existing_labels[idx] = 1.0
                has_real_labels = existing_labels.sum() > 0
            else:
                existing_labels = np.zeros(NUM_DESCRIPTORS, dtype=np.float32)
                has_real_labels = False

            spec_path = f["spec_path"][()].decode("utf-8")

        # Load audio
        audio_file = map_path.parent.parent.parent / spec_path
        if not audio_file.exists():
            return {"updated": False, "reason": "audio_missing"}

        with h5py.File(audio_file, "r") as af:
            a = torch.from_numpy(af["a"][:]).float()

    except Exception as e:
        return {"updated": False, "reason": f"load_error: {e}"}

    if x.shape[1] != SEQ_DIM:
        return {"updated": False, "reason": "wrong_seq_dim"}

    with torch.inference_mode():
        x_gpu = x.unsqueeze(0).to(device)
        a_gpu = a.unsqueeze(0).to(device)
        c_gpu = c.unsqueeze(0).to(device)
        logits = model(x_gpu, a_gpu, c_gpu)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    pseudo = np.full(NUM_DESCRIPTORS, -1.0, dtype=np.float32)  # -1 = uncertain
    pseudo[probs >= pos_threshold] = 1.0
    pseudo[probs <= neg_threshold] = 0.0

    new_labels = existing_labels.copy()
    if has_real_labels and not overwrite_real:
        # Only fill in where existing is 0 and pseudo is confident
        for i in range(NUM_DESCRIPTORS):
            if existing_labels[i] == 0.0 and pseudo[i] == 1.0:
                new_labels[i] = 1.0  # Add pseudo positive
    else:
        for i in range(NUM_DESCRIPTORS):
            if pseudo[i] >= 0:
                new_labels[i] = pseudo[i]

    if np.array_equal(new_labels, existing_labels):
        return {"updated": False, "reason": "no_change"}

    try:
        # Convert multi-hot back to sparse indices for storage
        new_indices = np.where(new_labels > 0.5)[0].astype(np.int32)
        with h5py.File(map_path, "a") as f:
            if "descriptor_indices" in f:
                del f["descriptor_indices"]
            f.create_dataset("descriptor_indices", data=new_indices, compression="lzf")
    except Exception as e:
        return {"updated": False, "reason": f"write_error: {e}"}

    n_added = int((new_labels > existing_labels).sum())
    return {"updated": True, "reason": "ok", "added": n_added}


def main() -> None:  # noqa: C901
    parser = ArgumentParser(description="Pseudo-label beatmaps with trained classifier")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to classifier checkpoint")
    parser.add_argument("--model-size", type=str, default="s", choices=["s", "m", "l"])
    parser.add_argument("--pos-threshold", type=float, default=0.8, help="Positive prediction threshold")
    parser.add_argument("--neg-threshold", type=float, default=0.2, help="Negative prediction threshold")
    parser.add_argument("--overwrite-real", action="store_true", help="Overwrite existing real labels")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading classifier from {args.checkpoint} on {device}...")
    model = load_classifier(args.checkpoint, args.model_size, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    all_maps = list(args.dataset_dir.rglob("*.map.h5"))
    print(f"Found {len(all_maps)} H5 files")

    stats = {"updated": 0, "skipped": 0, "total_added": 0}

    for map_path in tqdm(all_maps, desc="Pseudo-labeling"):
        result = pseudo_label_map(
            map_path,
            model,
            device,
            pos_threshold=args.pos_threshold,
            neg_threshold=args.neg_threshold,
            overwrite_real=args.overwrite_real,
        )
        if result["updated"]:
            stats["updated"] += 1
            stats["total_added"] += result.get("added", 0)
        else:
            stats["skipped"] += 1

    print(f"\nDone: {stats['updated']} updated, {stats['skipped']} skipped")
    print(f"Total new positive labels added: {stats['total_added']}")

    print("\nSampling 100 files for tag distribution...")
    import random

    sample = random.sample(all_maps, min(100, len(all_maps)))
    tag_counts = np.zeros(NUM_DESCRIPTORS)
    for p in sample:
        try:
            with h5py.File(p, "r") as f:
                if "descriptor_indices" in f:
                    for idx in f["descriptor_indices"][:]:
                        if 0 <= idx < NUM_DESCRIPTORS:
                            tag_counts[idx] += 1
        except Exception:
            continue

    print("\nTag distribution (sample):")
    for i, name in enumerate(DESCRIPTOR_TAGS):
        if tag_counts[i] > 0:
            print(f"  {name}: {int(tag_counts[i])}/{len(sample)}")


if __name__ == "__main__":
    main()
