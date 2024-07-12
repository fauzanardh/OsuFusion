import os
import random
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from typing import List

from tqdm.auto import tqdm

os.environ["CREATE_DATASET"] = "1"
from osu_fusion.scripts.dataset_creator import prepare_map  # noqa: E402


def run(
    dataset_dir: Path,
    osu_files: List[Path],
    worker_index: int,
    world_size: int,
) -> None:
    data = osu_files[worker_index::world_size]
    for osu_file in tqdm(data, position=worker_index, dynamic_ncols=True):
        try:
            prepare_map(dataset_dir, osu_file)
        except Exception:
            continue


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--osu_song_dir", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    args.dataset_dir.mkdir(exist_ok=True, parents=True)
    osu_files = list(args.osu_song_dir.rglob("*.osu"))
    print(f"Found {len(osu_files)} .osu files")

    random.shuffle(osu_files)
    with Pool(args.num_workers) as pool:
        pool.starmap(
            run,
            [
                (
                    args.dataset_dir,
                    osu_files,
                    worker_index,
                    args.num_workers,
                )
                for worker_index in range(args.num_workers)
            ],
        )

        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
