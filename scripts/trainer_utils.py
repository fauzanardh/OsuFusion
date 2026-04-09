import re
import shutil
from pathlib import Path
from typing import List

import torch


def get_total_norm(parameters: List[torch.Tensor], norm_type: float = 2.0) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type).item()


def _checkpoint_step(p: Path) -> int:
    m = re.match(r"checkpoint-(\d+)", p.name)
    return int(m.group(1)) if m else -1


def manage_checkpoints(project_dir: Path, max_num_checkpoints: int) -> None:
    checkpoints = sorted(
        [p for p in project_dir.glob("checkpoint-*") if _checkpoint_step(p) >= 0],
        key=_checkpoint_step,
    )
    for checkpoint in checkpoints[:-max_num_checkpoints]:
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
        else:
            checkpoint.unlink()


def clear_checkpoints(project_dir: Path) -> None:
    for checkpoint in project_dir.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
        else:
            checkpoint.unlink()
