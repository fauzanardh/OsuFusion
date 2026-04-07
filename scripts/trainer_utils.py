import shutil
from pathlib import Path
from typing import List

import torch


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


def clear_checkpoints(project_dir: Path) -> None:
    for checkpoint in project_dir.rglob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
        else:
            checkpoint.unlink()
