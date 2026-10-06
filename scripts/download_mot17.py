import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

DEFAULT_SEQUENCES: list[str] = [
    "MOT17-09-FRCNN",
    "MOT17-11-FRCNN",
]

BASE_URL: str = "https://huggingface.co/datasets/Morrison1025/MOT17/raw/main/train"
TARGET_FILES: list[str] = [
    "seqinfo.ini",
    "gt/gt.txt",
    "det/det.txt",
]


def download_sequence(seq_name: str, base_dir: Path) -> bool:
    seq_dir = base_dir / seq_name
    success = True

    for rel_path in TARGET_FILES:
        target_file = seq_dir / rel_path
        if target_file.exists() and target_file.stat().st_size > 0:
            print(f"Skipping {seq_name}/{rel_path} (exists, size={target_file.stat().st_size} bytes)")
            continue

        target_file.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{seq_name}/{rel_path}"
        print(f"Fetching {url} -> {target_file}")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                target_file.write_bytes(content)
                print(f"Saved {target_file} ({len(content)} bytes)")
        except Exception as exc:
            print(f"Failed to fetch {url}: {exc}", file=sys.stderr)
            success = False

    return success


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download MOT17 benchmark sequences.")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=DEFAULT_SEQUENCES,
        help="List of sequence identifiers to download",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Target directory for downloaded sequences",
    )
    args = parser.parse_args(argv)

    args.dest_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for seq in args.sequences:
        ok = download_sequence(seq, args.dest_dir)
        if not ok:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
