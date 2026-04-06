import json
import time
import traceback
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests
from tqdm import tqdm


def extract_beatmap_id_from_osu(osu_file: Path) -> Optional[int]:
    beatmap_id = None
    mode = None
    try:
        with osu_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("BeatmapID:"):
                    beatmap_id = int(line.split(":", 1)[1].strip())
                elif line.startswith("Mode:"):
                    mode = int(line.split(":", 1)[1].strip())
                    if mode != 0:
                        return None
                if beatmap_id is not None and mode is not None:
                    break
    except Exception:
        return None

    if mode != 0 or beatmap_id is None or beatmap_id <= 0:
        return None
    return beatmap_id


def scan_osu_files(osu_song_dir: Path) -> Set[int]:
    osu_files = list(osu_song_dir.rglob("*.osu"))
    print(f"Found {len(osu_files)} .osu files")

    ids = set()
    for osu_file in tqdm(osu_files, desc="Scanning .osu files"):
        bid = extract_beatmap_id_from_osu(osu_file)
        if bid is not None:
            ids.add(bid)

    print(f"Extracted {len(ids)} unique beatmap IDs")
    return ids


def fetch_omdb_tags(beatmap_id: int, api_key: str) -> Optional[str]:
    try:
        resp = requests.get(
            f"https://omdb.nyahh.net/api/beatmap/{beatmap_id}",
            params={"key": api_key},
            timeout=10,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data.get("Descriptors", None)
    except requests.RequestException:
        return None


def fetch_omdb_bulk(
    beatmap_ids: List[int],
    api_key: str,
    delay: float = 0.05,
) -> Dict[int, List[str]]:
    results: Dict[int, List[str]] = {}
    for bid in tqdm(beatmap_ids, desc="Fetching omdb tags"):
        descriptors_str = fetch_omdb_tags(bid, api_key)
        if descriptors_str:
            tags = [t.strip() for t in descriptors_str.split(",") if t.strip()]
            if tags:
                results[bid] = tags
        time.sleep(delay)
    print(f"Got omdb tags for {len(results)}/{len(beatmap_ids)} beatmaps")
    return results


class OsuApiClient:
    def __init__(self: "OsuApiClient", client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.token_expires_at: float = 0

    def _ensure_token(self: "OsuApiClient") -> None:
        if self.access_token is not None and time.time() < self.token_expires_at - 60:
            return
        resp = requests.post(
            "https://osu.ppy.sh/oauth/token",
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
                "scope": "public",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self.access_token = data["access_token"]
        self.token_expires_at = time.time() + data["expires_in"]

    def get_beatmaps(self: "OsuApiClient", beatmap_ids: List[int]) -> List[Dict]:
        self._ensure_token()
        resp = requests.get(
            "https://osu.ppy.sh/api/v2/beatmaps",
            params={"ids[]": beatmap_ids},
            headers={"Authorization": f"Bearer {self.access_token}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("beatmaps", [])


def fetch_osu_data(
    beatmap_ids: List[int],
    client_id: str,
    client_secret: str,
    batch_size: int = 50,
    delay: float = 0.5,
) -> List[Dict[str, Any]]:
    client = OsuApiClient(client_id, client_secret)
    results: List[Dict[str, Any]] = []

    batches = [beatmap_ids[i : i + batch_size] for i in range(0, len(beatmap_ids), batch_size)]
    for batch in tqdm(batches, desc="Fetching osu! API beatmap data"):
        try:
            beatmaps = client.get_beatmaps(batch)
            for bm in beatmaps:
                bid = bm["id"]
                owners = bm.get("owners", [])

                if owners:
                    user_ids = [o["id"] for o in owners]
                    usernames = [o.get("username", "") for o in owners]
                else:
                    user_ids = [bm.get("user_id", 0)]
                    usernames = []

                beatmapset = bm.get("beatmapset", {}) or {}
                submitted_date = beatmapset.get("submitted_date", None)

                results.append(
                    {
                        "id": bid,
                        "user_id": user_ids,
                        "username": usernames,
                        "submitted_date": submitted_date,
                    },
                )
        except Exception as e:
            print(f"\n[Error] osu! API batch failed: {e}")
            traceback.print_exc()

        time.sleep(delay)

    print(f"Got osu! data for {len(results)} beatmaps")
    return results


def save_descriptors_csv(descriptors: Dict[int, List[str]], output_path: Path) -> None:
    lines = []
    for bid in sorted(descriptors.keys()):
        for tag in descriptors[bid]:
            lines.append(f"{bid},{tag}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(lines)} descriptor entries to {output_path}")


def save_osu_data_json(data: List[Dict[str, Any]], output_path: Path) -> None:
    data_sorted = sorted(data, key=lambda x: x["id"])
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data_sorted, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data_sorted)} osu! data entries to {output_path}")


def load_existing_descriptors(path: Path) -> Dict[int, List[str]]:
    if not path.exists():
        return {}
    result: Dict[int, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            try:
                bid = int(parts[0])
            except ValueError:
                continue
            tag = parts[1].strip()
            if bid not in result:
                result[bid] = []
            result[bid].append(tag)
    return result


def load_existing_osu_data(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["id"]: entry for entry in data}


def main() -> None:
    parser = ArgumentParser(description="Fetch metadata for osu-fusion dataset")
    parser.add_argument("--osu-song-dir", type=Path, required=True, help="Path to osu! Songs directory")
    parser.add_argument("--output-descriptors", type=Path, default=Path("beatmap_descriptors.csv"))
    parser.add_argument("--output-osu-data", type=Path, default=Path("beatmap_osu_data.json"))
    parser.add_argument("--omdb-api-key", type=str, default="", help="omdb API key")
    parser.add_argument("--osu-client-id", type=str, default="", help="osu! API client ID")
    parser.add_argument("--osu-client-secret", type=str, default="", help="osu! API client secret")
    parser.add_argument("--skip-omdb", action="store_true", help="Skip omdb tag fetching")
    parser.add_argument("--skip-osu-api", action="store_true", help="Skip osu! API data fetching")
    parser.add_argument("--omdb-delay", type=float, default=0.05, help="Delay between omdb requests (seconds)")
    parser.add_argument("--osu-api-delay", type=float, default=0.5, help="Delay between osu! API batches (seconds)")
    args = parser.parse_args()

    all_beatmap_ids = sorted(scan_osu_files(args.osu_song_dir))
    print(f"Total unique beatmap IDs: {len(all_beatmap_ids)}")

    if not args.skip_omdb and args.omdb_api_key:
        existing_descriptors = load_existing_descriptors(args.output_descriptors)
        ids_needing_omdb = [bid for bid in all_beatmap_ids if bid not in existing_descriptors]
        print(
            f"Fetching omdb tags for {len(ids_needing_omdb)} new beatmaps ({len(existing_descriptors)} already have tags)...",  # noqa: E501
        )

        new_descriptors = fetch_omdb_bulk(ids_needing_omdb, args.omdb_api_key, delay=args.omdb_delay)
        existing_descriptors.update(new_descriptors)
        save_descriptors_csv(existing_descriptors, args.output_descriptors)
    else:
        print("Skipping omdb fetch")

    if not args.skip_osu_api and args.osu_client_id and args.osu_client_secret:
        existing_osu_data = load_existing_osu_data(args.output_osu_data)
        ids_needing_data = [bid for bid in all_beatmap_ids if bid not in existing_osu_data]
        print(
            f"Fetching osu! API data for {len(ids_needing_data)} new beatmaps ({len(existing_osu_data)} already have data)...",  # noqa: E501
        )

        new_osu_data = fetch_osu_data(
            ids_needing_data,
            args.osu_client_id,
            args.osu_client_secret,
            delay=args.osu_api_delay,
        )
        for entry in new_osu_data:
            existing_osu_data[entry["id"]] = entry
        save_osu_data_json(list(existing_osu_data.values()), args.output_osu_data)
    else:
        print("Skipping osu! API fetch")

    if args.output_descriptors.exists():
        descs = load_existing_descriptors(args.output_descriptors)
        print(f"Total beatmaps with descriptors: {len(descs)}")
    if args.output_osu_data.exists():
        osu_data = load_existing_osu_data(args.output_osu_data)
        print(f"Total beatmaps with osu! data: {len(osu_data)}")


if __name__ == "__main__":
    main()
