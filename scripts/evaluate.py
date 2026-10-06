import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mot_loader import MOTSequence, load_sequence
from src.metrics.mot_evaluator import (
    MOTMetrics,
    evaluate_sequence,
    load_predictions,
)


def compute_aggregate_metrics(
    sequence_metrics: List[Dict[str, Any]], tracker_name: str
) -> Dict[str, Any]:
    total_frames = sum(m["num_frames"] for m in sequence_metrics)
    total_objects = sum(m["num_objects"] for m in sequence_metrics)
    total_predictions = sum(m["num_predictions"] for m in sequence_metrics)
    total_fp = sum(m["num_false_positives"] for m in sequence_metrics)
    total_fn = sum(m["num_misses"] for m in sequence_metrics)
    total_id_switches = sum(m["num_switches"] for m in sequence_metrics)
    total_mt = sum(m["mostly_tracked"] for m in sequence_metrics)
    total_ml = sum(m["mostly_lost"] for m in sequence_metrics)
    total_pt = sum(m["partially_tracked"] for m in sequence_metrics)
    total_gt_ids = sum(m["num_unique_gt_ids"] for m in sequence_metrics)
    total_pred_ids = sum(m["num_unique_pred_ids"] for m in sequence_metrics)

    # Reconstructed True Positives from total objects and misses
    total_tp = total_objects - total_fn

    mota = (
        (1.0 - (total_fn + total_fp + total_id_switches) / total_objects)
        if total_objects > 0
        else 0.0
    )
    recall = (total_tp / total_objects) if total_objects > 0 else 0.0
    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0

    # Weighted MOTP by True Positives
    weighted_motp_sum = sum(
        m["motp"] * (m["num_objects"] - m["num_misses"]) for m in sequence_metrics
    )
    motp = (weighted_motp_sum / total_tp) if total_tp > 0 else 0.0

    # Weighted IDF1 across sequences
    weighted_idf1_sum = sum(
        m["idf1"] * (m["num_objects"] + m["num_predictions"])
        for m in sequence_metrics
    )
    denom_idf1 = sum(
        m["num_objects"] + m["num_predictions"] for m in sequence_metrics
    )
    idf1 = (weighted_idf1_sum / denom_idf1) if denom_idf1 > 0 else 0.0

    total_elapsed = sum(m.get("elapsed_sec", 0.0) for m in sequence_metrics)
    fps = (total_frames / total_elapsed) if total_elapsed > 0 else 0.0

    return {
        "tracker": tracker_name,
        "sequence": "COMBINED",
        "num_frames": total_frames,
        "num_objects": total_objects,
        "num_predictions": total_predictions,
        "mota": float(mota),
        "motp": float(motp),
        "idf1": float(idf1),
        "recall": float(recall),
        "precision": float(precision),
        "num_false_positives": total_fp,
        "num_misses": total_fn,
        "num_switches": total_id_switches,
        "mostly_tracked": total_mt,
        "mostly_lost": total_ml,
        "partially_tracked": total_pt,
        "num_unique_gt_ids": total_gt_ids,
        "num_unique_pred_ids": total_pred_ids,
        "elapsed_sec": float(total_elapsed),
        "fps": float(fps),
    }


def save_markdown_summary(
    all_results: List[Dict[str, Any]], output_path: Path
) -> None:
    headers = [
        "Tracker",
        "Sequence",
        "MOTA (%)",
        "MOTP (%)",
        "IDF1 (%)",
        "Recall (%)",
        "Precision (%)",
        "FP",
        "FN",
        "IDSw",
        "MT",
        "ML",
        "FPS",
    ]

    lines = [
        "# Evaluation Results Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for r in all_results:
        row = [
            r["tracker"],
            r["sequence"],
            f"{r['mota'] * 100:.2f}",
            f"{r['motp'] * 100:.2f}",
            f"{r['idf1'] * 100:.2f}",
            f"{r['recall'] * 100:.2f}",
            f"{r['precision'] * 100:.2f}",
            str(r["num_false_positives"]),
            str(r["num_misses"]),
            str(r["num_switches"]),
            str(r["mostly_tracked"]),
            str(r["mostly_lost"]),
            f"{r['fps']:.1f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_csv_summary(
    all_results: List[Dict[str, Any]], output_path: Path
) -> None:
    if not all_results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(all_results[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate tracking results against MOT ground truth.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Directory containing ground truth sequences",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing tracking outputs",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["sort", "bytetrack"],
        help="Trackers to evaluate",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU match threshold for True Positive assignment",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.0,
        help="Minimum visibility filter for ground truth annotations",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/metrics.json"),
        help="Output JSON metrics path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/metrics.csv"),
        help="Output CSV metrics path",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/summary_table.md"),
        help="Output Markdown metrics path",
    )

    args = parser.parse_args(argv)

    seq_dirs = sorted([d for d in args.data_dir.iterdir() if d.is_dir()])
    if not seq_dirs:
        print(f"No sequences found in {args.data_dir}", file=sys.stderr)
        return 1

    all_table_rows: List[Dict[str, Any]] = []
    nested_json: Dict[str, Dict[str, Any]] = {}

    for tracker_name in args.trackers:
        tracker_dir = args.results_dir / tracker_name
        if not tracker_dir.exists():
            print(f"Warning: Tracker result directory {tracker_dir} not found, skipping.")
            continue

        tracker_seq_metrics: List[Dict[str, Any]] = []
        nested_json[tracker_name] = {}

        for seq_dir in seq_dirs:
            pred_file = tracker_dir / f"{seq_dir.name}.txt"
            time_file = tracker_dir / f"{seq_dir.name}_timing.json"

            if not pred_file.exists():
                print(f"Warning: Prediction file {pred_file} not found, skipping.")
                continue

            sequence = load_sequence(seq_dir)
            predictions = load_predictions(pred_file)

            elapsed_sec = 0.0
            if time_file.exists():
                try:
                    time_data = json.loads(time_file.read_text(encoding="utf-8"))
                    elapsed_sec = float(time_data.get("elapsed_sec", 0.0))
                except Exception:
                    elapsed_sec = 0.0

            metrics: MOTMetrics = evaluate_sequence(
                sequence=sequence,
                predictions_by_frame=predictions,
                iou_threshold=args.iou_threshold,
                min_visibility=args.min_visibility,
                elapsed_time_sec=elapsed_sec,
            )

            metrics_dict = metrics.to_dict()
            metrics_dict["tracker"] = tracker_name
            metrics_dict["sequence"] = sequence.name
            metrics_dict["elapsed_sec"] = elapsed_sec

            tracker_seq_metrics.append(metrics_dict)
            nested_json[tracker_name][sequence.name] = metrics_dict
            all_table_rows.append(metrics_dict)

        if tracker_seq_metrics:
            agg = compute_aggregate_metrics(tracker_seq_metrics, tracker_name)
            nested_json[tracker_name]["COMBINED"] = agg
            all_table_rows.append(agg)

    if not all_table_rows:
        print("No evaluation metrics could be computed.", file=sys.stderr)
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(nested_json, indent=2), encoding="utf-8")
    save_csv_summary(all_table_rows, args.output_csv)
    save_markdown_summary(all_table_rows, args.output_md)

    print(f"Saved evaluation metrics to:")
    print(f"  - {args.output_json}")
    print(f"  - {args.output_csv}")
    print(f"  - {args.output_md}")
    print("\nSummary Table:")
    print(args.output_md.read_text(encoding="utf-8"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
