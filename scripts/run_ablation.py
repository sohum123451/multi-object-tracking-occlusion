import argparse
import configparser
import csv
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Sequence
import numpy as np

# Ensure numpy compatibility for older TrackEval imports under NumPy 2.0+
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRACKEVAL_ROOT = REPO_ROOT / "third_party" / "TrackEval"
if str(TRACKEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKEVAL_ROOT))

import trackeval

from src.tracking.deepsort import DeepSort
from src.tracking.embed import OSNetExtractor
from src.tracking.io import format_mot_row, load_mot_detections
from src.tracking.sort import Sort


def run_tracker_on_sequence(
    tracker_name: str,
    dets_by_frame: Dict[int, np.ndarray],
    features_by_frame: Dict[int, np.ndarray] | None,
    conf_threshold: float,
    max_age: int,
    output_path: Path,
) -> None:
    max_frame = max(dets_by_frame.keys()) if dets_by_frame else 0
    out_lines: List[str] = []

    if tracker_name == "sort":
        tracker = Sort(max_age=max_age, min_hits=3, iou_threshold=0.3)
        tracker.reset_id_counter()

        for frame in range(1, max_frame + 1):
            frame_dets = dets_by_frame.get(frame, np.empty((0, 5), dtype=float))
            if len(frame_dets) > 0:
                frame_dets = frame_dets[frame_dets[:, 4] >= conf_threshold]
            res = tracker.update(frame_dets)
            for trk in res:
                x1, y1, x2, y2, trk_id, conf = trk[:6]
                out_lines.append(
                    format_mot_row(frame, int(trk_id), np.array([x1, y1, x2, y2]), conf)
                )

    elif tracker_name == "deepsort":
        assert features_by_frame is not None
        tracker = DeepSort(
            max_age=max_age,
            min_hits=3,
            iou_threshold=0.3,
            max_cosine_distance=0.2,
            nn_budget=100,
            gating_threshold=9.4877,
        )
        tracker.reset_id_counter()

        for frame in range(1, max_frame + 1):
            frame_dets = dets_by_frame.get(frame, np.empty((0, 5), dtype=float))
            frame_feats = features_by_frame.get(frame, np.empty((0, 512), dtype=np.float32))

            if len(frame_dets) > 0:
                mask = frame_dets[:, 4] >= conf_threshold
                filtered_dets = frame_dets[mask]
                filtered_feats = frame_feats[mask]
            else:
                filtered_dets = np.empty((0, 5), dtype=float)
                filtered_feats = np.empty((0, 512), dtype=np.float32)

            boxes = filtered_dets[:, :4] if len(filtered_dets) > 0 else np.empty((0, 4), dtype=float)
            scores = filtered_dets[:, 4] if len(filtered_dets) > 0 else np.empty((0,), dtype=float)

            res = tracker.update(boxes, filtered_feats, scores)
            for trk in res:
                x1, y1, x2, y2, trk_id, conf = trk[:6]
                out_lines.append(
                    format_mot_row(frame, int(trk_id), np.array([x1, y1, x2, y2]), conf)
                )

    else:
        raise ValueError(f"Unknown tracker name: {tracker_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(out_lines), encoding="utf-8")


def run_ablation(
    seqs: List[str],
    confs: List[float],
    max_ages: List[int],
    trackers: List[str],
    dets_dir: Path,
    embeddings_dir: Path,
    gt_dir: Path,
    temp_dir: Path,
    output_csv: Path,
) -> List[Dict[str, Any]]:
    # 1. Preload detections and embeddings into memory
    print("Preloading detections and embeddings...")
    dets_cache: Dict[str, Dict[int, np.ndarray]] = {}
    feats_cache: Dict[str, Dict[int, np.ndarray]] = {}

    for seq in seqs:
        det_file = dets_dir / f"{seq}.txt"
        dets_cache[seq] = load_mot_detections(det_file)

        embed_file = embeddings_dir / f"{seq}_osnet_x0_25.npz"
        if embed_file.is_file():
            with np.load(embed_file) as f_npz:
                feats_cache[seq] = {
                    int(k.replace("frame_", "")): f_npz[k] for k in f_npz.files
                }
        else:
            print(f"Warning: Embeddings file {embed_file} not found.", file=sys.stderr)
            feats_cache[seq] = {}

    # 2. Run all combinations and store track outputs in temp_dir
    tracker_run_names: List[str] = []
    config_map: Dict[str, Dict[str, Any]] = {}

    temp_dir.mkdir(parents=True, exist_ok=True)

    print("Running tracking sweeps...")
    for tracker in trackers:
        for conf in confs:
            for max_age in max_ages:
                run_tag = f"{tracker}_conf{conf:.1f}_age{max_age}"
                tracker_run_names.append(run_tag)
                config_map[run_tag] = {
                    "tracker": tracker,
                    "conf": conf,
                    "max_age": max_age,
                }

                for seq in seqs:
                    out_path = temp_dir / run_tag / f"{seq}.txt"
                    run_tracker_on_sequence(
                        tracker_name=tracker,
                        dets_by_frame=dets_cache[seq],
                        features_by_frame=feats_cache.get(seq) if tracker == "deepsort" else None,
                        conf_threshold=conf,
                        max_age=max_age,
                        output_path=out_path,
                    )

    # 3. Run TrackEval across all configurations in one pass
    print("Evaluating configurations with TrackEval...")
    seq_info = {}
    for seq in seqs:
        ini_path = gt_dir / f"{seq}-FRCNN" / "seqinfo.ini"
        if not ini_path.is_file():
            ini_path = gt_dir / seq / "seqinfo.ini"
        if ini_path.is_file():
            config = configparser.ConfigParser()
            config.read(ini_path)
            seq_info[seq] = int(config["Sequence"]["seqLength"])
        else:
            seq_info[seq] = None

    dataset_config = {
        "GT_FOLDER": str(gt_dir),
        "TRACKERS_FOLDER": str(temp_dir),
        "OUTPUT_FOLDER": str(REPO_ROOT / "results" / "evaluation_ablation"),
        "TRACKERS_TO_EVAL": tracker_run_names,
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "SKIP_SPLIT_FOL": True,
        "TRACKER_SUB_FOLDER": "",
        "CLASSES_TO_EVAL": ["pedestrian"],
        "SEQ_INFO": seq_info,
        "GT_LOC_FORMAT": "{gt_folder}/{seq}-FRCNN/gt/gt.txt",
    }

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_RESULTS"] = False
    eval_config["PRINT_CONFIG"] = False
    eval_config["TIME_PROGRESS"] = False
    eval_config["OUTPUT_SUMMARY"] = False
    eval_config["OUTPUT_DETAILED"] = False
    eval_config["PLOT_CURVES"] = False

    evaluator = trackeval.Evaluator(eval_config)
    ds = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    res_data, _ = evaluator.evaluate([ds], metrics_list)

    # 4. Extract rows for CSV
    order_seqs = list(seqs) + ["COMBINED_SEQ"]
    all_rows: List[Dict[str, Any]] = []

    for run_tag in tracker_run_names:
        cfg = config_map[run_tag]
        for seq_key in order_seqs:
            if seq_key not in res_data["MotChallenge2DBox"][run_tag]:
                continue
            m = res_data["MotChallenge2DBox"][run_tag][seq_key]["pedestrian"]
            row = {
                "tracker": cfg["tracker"],
                "conf": cfg["conf"],
                "max_age": cfg["max_age"],
                "sequence": "COMBINED" if seq_key == "COMBINED_SEQ" else seq_key,
                "MOTA": round(float(m["CLEAR"]["MOTA"]) * 100.0, 2),
                "MOTP": round(float(m["CLEAR"]["MOTP"]) * 100.0, 2),
                "IDF1": round(float(m["Identity"]["IDF1"]) * 100.0, 2),
                "HOTA": round(float(np.mean(m["HOTA"]["HOTA"])) * 100.0, 2),
                "IDSW": int(m["CLEAR"]["IDSW"]),
                "FP": int(m["CLEAR"]["CLR_FP"]),
                "FN": int(m["CLEAR"]["CLR_FN"]),
                "Frag": int(m["CLEAR"]["Frag"]),
                "MT": int(m["CLEAR"]["MT"]),
                "ML": int(m["CLEAR"]["ML"]),
            }
            all_rows.append(row)

    # 5. Write to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tracker",
        "conf",
        "max_age",
        "sequence",
        "MOTA",
        "MOTP",
        "IDF1",
        "HOTA",
        "IDSW",
        "FP",
        "FN",
        "Frag",
        "MT",
        "ML",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    # Clean up temp tracks
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    return all_rows


def print_ablation_table(rows: List[Dict[str, Any]], combined_only: bool = True) -> None:
    if combined_only:
        display_rows = [r for r in rows if r["sequence"] == "COMBINED"]
    else:
        display_rows = rows

    headers = [
        "Tracker",
        "Conf",
        "MaxAge",
        "Seq",
        "MOTA",
        "MOTP",
        "IDF1",
        "HOTA",
        "IDSW",
        "FP",
        "FN",
        "Frag",
        "MT",
        "ML",
    ]
    col_keys = [
        "tracker",
        "conf",
        "max_age",
        "sequence",
        "MOTA",
        "MOTP",
        "IDF1",
        "HOTA",
        "IDSW",
        "FP",
        "FN",
        "Frag",
        "MT",
        "ML",
    ]
    widths = [max(len(h), 8) for h in headers]
    widths[0] = 10
    widths[3] = 10

    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    sep_str = "-|-".join("-" * w for w in widths)
    print(header_str)
    print(sep_str)
    for r in display_rows:
        row_str = " | ".join(f"{str(r[k]):<{w}}" for k, w in zip(col_keys, widths))
        print(row_str)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ablation sweep over detector confidence and tracker max_age."
    )
    parser.add_argument(
        "--confs",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5],
        help="Detector confidence thresholds to sweep",
    )
    parser.add_argument(
        "--max-ages",
        nargs="+",
        type=int,
        default=[15, 30, 60],
        help="Tracker max_age values to sweep",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["sort", "deepsort"],
        help="Trackers to include in ablation",
    )
    parser.add_argument(
        "--seq",
        nargs="+",
        default=["MOT17-02", "MOT17-04", "MOT17-09"],
        help="Sequences to evaluate",
    )
    parser.add_argument(
        "--dets-dir",
        type=Path,
        default=Path("results/dets"),
        help="Directory with cached detections",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("results/embeddings"),
        help="Directory with cached appearance embeddings",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Path to MOT ground truth sequences",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("results/ablation_tracks"),
        help="Temporary directory for tracker outputs during evaluation",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/ablation.csv"),
        help="Path to write ablation CSV",
    )

    args = parser.parse_args(argv)

    print(
        f"Starting ablation sweep on confs={args.confs}, max_ages={args.max_ages}, "
        f"trackers={args.trackers}, sequences={args.seq}..."
    )

    rows = run_ablation(
        seqs=args.seq,
        confs=args.confs,
        max_ages=args.max_ages,
        trackers=args.trackers,
        dets_dir=args.dets_dir,
        embeddings_dir=args.embeddings_dir,
        gt_dir=args.data_dir,
        temp_dir=args.temp_dir,
        output_csv=args.output_csv,
    )

    print(f"\nAblation completed! Results written to {args.output_csv}\n")
    print("Combined Ablation Results Table:")
    print_ablation_table(rows, combined_only=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
