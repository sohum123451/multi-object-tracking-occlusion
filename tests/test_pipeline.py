from pathlib import Path
import pytest

from scripts.download_mot17 import main as download_main
from scripts.evaluate import main as evaluate_main
from scripts.plot_results import main as plot_main
from scripts.run_tracking import main as run_main


def test_cli_end_to_end(tmp_path: Path) -> None:
    data_dir = Path("data/MOT17/train")
    if not data_dir.exists():
        pytest.skip("Data directory not present")

    # Run tracking into temporary results directory for sequence MOT17-09-FRCNN
    tmp_results = tmp_path / "results"
    run_code = run_main([
        "--data-dir", str(data_dir),
        "--output-dir", str(tmp_results),
        "--sequences", "MOT17-09-FRCNN",
        "--trackers", "sort", "bytetrack",
    ])
    assert run_code == 0
    assert (tmp_results / "sort" / "MOT17-09-FRCNN.txt").exists()
    assert (tmp_results / "bytetrack" / "MOT17-09-FRCNN.txt").exists()

    # Evaluate
    eval_json = tmp_results / "metrics.json"
    eval_csv = tmp_results / "metrics.csv"
    eval_md = tmp_results / "summary_table.md"
    eval_code = evaluate_main([
        "--data-dir", str(data_dir),
        "--results-dir", str(tmp_results),
        "--output-json", str(eval_json),
        "--output-csv", str(eval_csv),
        "--output-md", str(eval_md),
    ])
    assert eval_code == 0
    assert eval_json.exists()
    assert eval_csv.exists()
    assert eval_md.exists()

    # Plot
    charts_dir = tmp_results / "charts"
    plot_code = plot_main([
        "--metrics-json", str(eval_json),
        "--charts-dir", str(charts_dir),
    ])
    assert plot_code == 0
    assert (charts_dir / "mota_idf1_comparison.png").exists()
