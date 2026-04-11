import json
import random
import traceback
import warnings
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from osu_fusion.data.prepare_data import prepare_map

# Suppress FutureWarnings from librosa since we are using an older version
warnings.filterwarnings("ignore", category=FutureWarning)

_descriptor_lookup: Optional[Dict[int, List[str]]] = None
_user_lookup: Optional[Dict[int, List[int]]] = None
_era_lookup: Optional[Dict[int, str]] = None


def load_descriptor_csv(path: Path) -> Dict[int, List[str]]:
    lookup: Dict[int, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            try:
                beatmap_id = int(parts[0])
            except ValueError:
                continue
            tag_name = parts[1].strip()
            if beatmap_id not in lookup:
                lookup[beatmap_id] = []
            lookup[beatmap_id].append(tag_name)
    print(f"Loaded descriptors for {len(lookup)} beatmaps")
    return lookup


def load_osu_data_json(path: Path, dataset_dir: Path) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_user_ids: set = set()
    raw_user_lookup: Dict[int, List[int]] = {}
    era_lookup: Dict[int, str] = {}

    for entry in data:
        beatmap_id = entry["id"]
        user_ids = entry["user_id"]

        if beatmap_id not in raw_user_lookup:
            raw_user_lookup[beatmap_id] = []
        for uid in user_ids:
            if uid not in raw_user_lookup[beatmap_id]:
                raw_user_lookup[beatmap_id].append(uid)
                all_user_ids.add(uid)

        submitted_date = entry.get("submitted_date")
        if submitted_date:
            era_lookup[beatmap_id] = submitted_date

    mapper_index = {uid: idx for idx, uid in enumerate(sorted(all_user_ids))}
    num_mappers = len(mapper_index)
    print(f"Built mapper index: {num_mappers} unique mappers")

    mapper_index_path = dataset_dir / "mapper_index.json"
    with mapper_index_path.open("w") as f:
        json.dump({str(k): v for k, v in mapper_index.items()}, f, indent=4)
    print(f"Saved mapper index to {mapper_index_path}")

    user_lookup: Dict[int, List[int]] = {}
    for beatmap_id, user_ids in raw_user_lookup.items():
        user_lookup[beatmap_id] = [mapper_index[uid] for uid in user_ids]

    print(f"Loaded user data for {len(user_lookup)} beatmaps")
    print(f"Loaded era data for {len(era_lookup)} beatmaps")
    return user_lookup, era_lookup


def _init_worker(
    descriptor_lookup: Optional[Dict],
    user_lookup: Optional[Dict],
    era_lookup: Optional[Dict],
) -> None:
    global _descriptor_lookup, _user_lookup, _era_lookup
    _descriptor_lookup = descriptor_lookup
    _user_lookup = user_lookup
    _era_lookup = era_lookup


def worker_task(args: Tuple[Path, Path]) -> None:
    data_dir, osu_file = args
    try:
        prepare_map(
            data_dir,
            osu_file,
            descriptor_lookup=_descriptor_lookup,
            user_lookup=_user_lookup,
            era_lookup=_era_lookup,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"\n[Error] Failed to prepare map for {osu_file}: {e}")


def main() -> None:
    parser = ArgumentParser(description="OSU Dataset Creator")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory to store the dataset")
    parser.add_argument("--osu-song-dir", type=Path, required=True, help="Directory containing .osu files")
    parser.add_argument("--num-workers", type=int, default=cpu_count(), help="Number of worker processes")
    parser.add_argument("--max-beatmaps", type=int, default=None, help="Maximum number of beatmaps to process")
    parser.add_argument("--descriptors-csv", type=Path, default=None, help="Path to beatmap_descriptors.csv")
    parser.add_argument("--osu-data-json", type=Path, default=None, help="Path to beatmap_osu_data.json")
    args = parser.parse_args()

    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    descriptor_lookup = None
    user_lookup = None
    era_lookup = None

    if args.descriptors_csv and args.descriptors_csv.exists():
        descriptor_lookup = load_descriptor_csv(args.descriptors_csv)

    if args.osu_data_json and args.osu_data_json.exists():
        user_lookup, era_lookup = load_osu_data_json(args.osu_data_json, args.dataset_dir)

    osu_files = list(args.osu_song_dir.rglob("*.osu"))
    print(f"Found {len(osu_files)} .osu files")

    if not osu_files:
        print("No .osu files found. Exiting.")
        return
    random.shuffle(osu_files)

    if args.max_beatmaps is not None:
        osu_files = osu_files[: args.max_beatmaps]
    print(f"Processing {len(osu_files)} beatmaps")

    task_args = [(args.dataset_dir, osu_file) for osu_file in osu_files]
    with (
        Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(descriptor_lookup, user_lookup, era_lookup),
        ) as pool,
        tqdm(total=len(task_args), desc="Processing Maps", dynamic_ncols=True) as pbar,
    ):
        for _ in pool.imap_unordered(worker_task, task_args):
            pbar.update(1)


if __name__ == "__main__":
    main()
