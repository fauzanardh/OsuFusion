from argparse import ArgumentParser
from pathlib import Path

from osu_fusion.data.dataset import build_metadata_cache


def main() -> None:
    parser = ArgumentParser(description="Build metadata cache for the beatmap dataset")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Root directory containing .map.h5 files")
    args = parser.parse_args()

    cache_path = args.dataset_dir / "metadata_cache.json"
    build_metadata_cache(args.dataset_dir, cache_path)


if __name__ == "__main__":
    main()
