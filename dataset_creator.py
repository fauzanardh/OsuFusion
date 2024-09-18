import os
import random
import traceback
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

# Set environment variable before importing dataset_creator
os.environ["CREATE_DATASET"] = "1"
from osu_fusion.scripts.dataset_creator import prepare_map


def worker_task(args: Tuple[Path, Path]) -> None:
    data_dir, osu_file = args
    try:
        prepare_map(data_dir, osu_file)
    except Exception as e:
        traceback.print_exc()
        print(f"[Error] Failed to prepare map for {osu_file}: {e}")


def main() -> None:
    parser = ArgumentParser(description="OSU Dataset Creator")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory to store the dataset")
    parser.add_argument("--osu-song-dir", type=Path, required=True, help="Directory containing .osu files")
    parser.add_argument("--num-workers", type=int, default=cpu_count(), help="Number of worker processes")
    args = parser.parse_args()

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    osu_files = list(args.osu_song_dir.rglob("*.osu"))
    print(f"Found {len(osu_files)} .osu files")

    if not osu_files:
        print("No .osu files found. Exiting.")
        return
    random.shuffle(osu_files)

    task_args = [(args.dataset_dir, osu_file) for osu_file in osu_files]
    with (
        Pool(processes=args.num_workers) as pool,
        tqdm(total=len(task_args), desc="Processing Maps", dynamic_ncols=True) as pbar,
    ):
        for _ in pool.imap_unordered(worker_task, task_args):
            pbar.update(1)


if __name__ == "__main__":
    main()
