import argparse
import configparser
import csv
from pathlib import Path
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


def evaluate_trackers(
    trackers_folder: Path,
    gt_folder: Path,
    trackers_to_eval: List[str],
    seqs: List[str],
    output_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    seq_info = {}
    for seq in seqs:
        ini_path = gt_folder / f"{seq}-FRCNN" / "seqinfo.ini"
        if not ini_path.is_file():
            ini_path = gt_folder / seq / "seqinfo.ini"
        if ini_path.is_file():
            config = configparser.ConfigParser()
            config.read(ini_path)
            seq_info[seq] = int(config["Sequence"]["seqLength"])
        else:
            seq_info[seq] = None

    dataset_config = {
        "GT_FOLDER": str(gt_folder),
        "TRACKERS_FOLDER": str(trackers_folder),
        "OUTPUT_FOLDER": str(output_dir or (REPO_ROOT / "results" / "evaluation")),
        "TRACKERS_TO_EVAL": trackers_to_eval,
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
    results_list: List[Dict[str, Any]] = []

    order_seqs = list(seqs) + ["COMBINED_SEQ"]
    for tracker in trackers_to_eval:
        for seq_key in order_seqs:
            if seq_key not in res_data["MotChallenge2DBox"][tracker]:
                continue
            m = res_data["MotChallenge2DBox"][tracker][seq_key]["pedestrian"]
            row = {
                "tracker": tracker,
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
            results_list.append(row)

    return results_list


def write_metrics_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tracker",
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
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "Tracker",
        "Sequence",
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
    widths[1] = 10

    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    sep_str = "-|-".join("-" * w for w in widths)
    print(header_str)
    print(sep_str)
    for r in rows:
        row_str = " | ".join(f"{str(r[k]):<{w}}" for k, w in zip(col_keys, widths))
        print(row_str)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate tracking predictions against ground truth using TrackEval."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Path to MOT ground truth sequences",
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=Path("results/tracks"),
        help="Path to tracker results directory",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["sort", "deepsort"],
        help="Trackers to evaluate",
    )
    parser.add_argument(
        "--seq",
        nargs="+",
        default=["MOT17-02", "MOT17-04", "MOT17-09"],
        help="Sequences to evaluate",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/metrics.csv"),
        help="Path to write evaluation CSV",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Path to TrackEval internal outputs directory",
    )

    args = parser.parse_args(argv)

    print(
        f"Evaluating trackers {args.trackers} on {args.seq} using TrackEval..."
    )
    results = evaluate_trackers(
        trackers_folder=args.tracks_dir,
        gt_folder=args.data_dir,
        trackers_to_eval=args.trackers,
        seqs=args.seq,
        output_dir=args.eval_dir,
    )

    write_metrics_csv(results, args.output_csv)
    print(f"Results written to {args.output_csv}\n")
    print_table(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
