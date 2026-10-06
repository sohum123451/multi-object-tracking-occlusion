import argparse
import io
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple
import urllib.request
import zipfile

DEFAULT_ZIP_URL: str = "https://motchallenge.net/data/MOT17Labels.zip"
DEFAULT_DEST_DIR: Path = Path("data/MOT17/train")

REQUIRED_STRUCTURE: List[Tuple[str, str]] = [
    ("seqinfo.ini", "file"),
    ("img1", "dir"),
    ("gt/gt.txt", "file"),
    ("det", "dir"),
]


def download_and_extract_train_split(
    zip_url: str,
    dest_dir: Path,
) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MOT17 labels from {zip_url}...")

    try:
        req = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
        print(f"Downloaded {len(content)} bytes.")
    except Exception as exc:
        print(f"Failed to download from {zip_url}: {exc}", file=sys.stderr)
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            namelist = zf.namelist()
            train_entries = [name for name in namelist if name.startswith("train/")]
            print(f"Extracting {len(train_entries)} train entries to {dest_dir}...")

            for entry in train_entries:
                parts = entry.split("/", 1)
                if len(parts) < 2 or not parts[1]:
                    continue
                rel_path = parts[1]
                target_path = dest_dir / rel_path

                if entry.endswith("/"):
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(zf.read(entry))

    except Exception as exc:
        print(f"Failed to extract zip contents: {exc}", file=sys.stderr)
        return False

    # Ensure img1 directory exists for each extracted sequence
    for seq_dir in dest_dir.iterdir():
        if seq_dir.is_dir():
            (seq_dir / "img1").mkdir(parents=True, exist_ok=True)

    return True


def verify_sequence_structure(seq_dir: Path) -> Dict[str, bool]:
    status: Dict[str, bool] = {}
    for item, kind in REQUIRED_STRUCTURE:
        path = seq_dir / item
        if kind == "file":
            status[item] = path.is_file() and path.stat().st_size > 0
        elif kind == "dir":
            status[item] = path.is_dir()
    return status


def verify_folder_structure(dest_dir: Path) -> Tuple[bool, Dict[str, Dict[str, bool]]]:
    if not dest_dir.exists():
        print(f"Directory {dest_dir} does not exist.", file=sys.stderr)
        return False, {}

    seq_dirs = sorted([d for d in dest_dir.iterdir() if d.is_dir()])
    if not seq_dirs:
        print(f"No sequence directories found in {dest_dir}.", file=sys.stderr)
        return False, {}

    all_valid = True
    results: Dict[str, Dict[str, bool]] = {}

    for seq_dir in seq_dirs:
        status = verify_sequence_structure(seq_dir)
        results[seq_dir.name] = status
        if not all(status.values()):
            all_valid = False

    return all_valid, results


def print_verification_table(results: Dict[str, Dict[str, bool]]) -> None:
    headers = ["Sequence", "seqinfo.ini", "img1/", "gt/gt.txt", "det/", "Valid"]
    row_fmt = "{:<20} {:<15} {:<10} {:<15} {:<10} {:<8}"
    print("\nVerification Results:")
    print(row_fmt.format(*headers))
    print("-" * 80)

    for seq_name, status in results.items():
        v_ini = "PASS" if status.get("seqinfo.ini") else "FAIL"
        v_img = "PASS" if status.get("img1") else "FAIL"
        v_gt = "PASS" if status.get("gt/gt.txt") else "FAIL"
        v_det = "PASS" if status.get("det") else "FAIL"
        overall = "PASS" if all(status.values()) else "FAIL"
        print(row_fmt.format(seq_name, v_ini, v_img, v_gt, v_det, overall))
    print("-" * 80)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download MOT17 train split and verify sequence folder structure."
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEFAULT_DEST_DIR,
        help="Target directory for MOT17 train sequences",
    )
    parser.add_argument(
        "--zip-url",
        type=str,
        default=DEFAULT_ZIP_URL,
        help="URL to MOT17 labels zip archive",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing directory structure without downloading",
    )

    args = parser.parse_args(argv)

    if not args.verify_only:
        success = download_and_extract_train_split(args.zip_url, args.dest_dir)
        if not success:
            print("Download or extraction failed.", file=sys.stderr)
            return 1

    all_valid, verification_results = verify_folder_structure(args.dest_dir)
    print_verification_table(verification_results)

    if all_valid:
        print(f"All {len(verification_results)} sequences verified successfully.")
        return 0
    else:
        print("Folder structure verification failed for one or more sequences.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
