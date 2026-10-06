import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_metrics(
    metrics_data: Dict[str, Dict[str, Any]], output_path: Path
) -> None:
    trackers = [t for t in ["sort", "bytetrack"] if t in metrics_data]
    if not trackers:
        return

    seq_key = "COMBINED"
    mota_vals = [metrics_data[t][seq_key]["mota"] * 100 for t in trackers]
    motp_vals = [metrics_data[t][seq_key]["motp"] * 100 for t in trackers]
    idf1_vals = [metrics_data[t][seq_key]["idf1"] * 100 for t in trackers]

    x = np.arange(len(trackers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(x - width, mota_vals, width, label="MOTA (%)", color="#1f77b4")
    rects2 = ax.bar(x, motp_vals, width, label="MOTP (%)", color="#2ca02c")
    rects3 = ax.bar(x + width, idf1_vals, width, label="IDF1 (%)", color="#ff7f0e")

    ax.set_ylabel("Score (%)")
    ax.set_title("Tracking Accuracy Comparison on MOT17 Benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in trackers])
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects: Any) -> None:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_error_metrics(
    metrics_data: Dict[str, Dict[str, Any]], output_path: Path
) -> None:
    trackers = [t for t in ["sort", "bytetrack"] if t in metrics_data]
    if not trackers:
        return

    seq_key = "COMBINED"
    fp_vals = [metrics_data[t][seq_key]["num_false_positives"] for t in trackers]
    fn_vals = [metrics_data[t][seq_key]["num_misses"] for t in trackers]
    idsw_vals = [metrics_data[t][seq_key]["num_switches"] for t in trackers]

    x = np.arange(len(trackers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(x - width, fp_vals, width, label="False Positives (FP)", color="#d62728")
    rects2 = ax.bar(x, fn_vals, width, label="Misses (FN)", color="#9467bd")
    rects3 = ax.bar(x + width, idsw_vals, width, label="ID Switches (IDSw)", color="#8c564b")

    ax.set_ylabel("Count")
    ax.set_title("Tracking Error Breakdown on MOT17 Benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in trackers])
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects: Any) -> None:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_fps_metrics(
    metrics_data: Dict[str, Dict[str, Any]], output_path: Path
) -> None:
    trackers = [t for t in ["sort", "bytetrack"] if t in metrics_data]
    if not trackers:
        return

    seq_key = "COMBINED"
    fps_vals = [metrics_data[t][seq_key]["fps"] for t in trackers]

    x = np.arange(len(trackers))
    width = 0.4

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    rects = ax.bar(x, fps_vals, width, color="#2b5c8f")

    ax.set_ylabel("Frames Per Second (FPS)")
    ax.set_title("Tracking Runtime Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in trackers])
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f} FPS",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate evaluation plots from results/metrics.json.")
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("results/metrics.json"),
        help="Path to computed metrics JSON file",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=Path("results/charts"),
        help="Directory where generated charts are saved",
    )
    args = parser.parse_args(argv)

    if not args.metrics_json.exists():
        print(f"Error: {args.metrics_json} does not exist. Run evaluate.py first.", file=sys.stderr)
        return 1

    metrics_data = json.loads(args.metrics_json.read_text(encoding="utf-8"))

    plot_accuracy_metrics(metrics_data, args.charts_dir / "mota_idf1_comparison.png")
    plot_error_metrics(metrics_data, args.charts_dir / "error_breakdown.png")
    plot_fps_metrics(metrics_data, args.charts_dir / "fps_comparison.png")

    print(f"Charts saved to {args.charts_dir}:")
    print(f"  - {args.charts_dir / 'mota_idf1_comparison.png'}")
    print(f"  - {args.charts_dir / 'error_breakdown.png'}")
    print(f"  - {args.charts_dir / 'fps_comparison.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
