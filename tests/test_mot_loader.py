from pathlib import Path
import numpy as np
import pytest

from src.data.mot_loader import (
    Detection,
    GroundTruth,
    load_sequence,
    parse_detections,
    parse_ground_truths,
    parse_seqinfo,
    tlbr_to_tlwh,
    tlwh_to_tlbr,
)


def test_tlwh_and_tlbr_conversions() -> None:
    tlwh = np.array([10.0, 20.0, 30.0, 40.0])
    tlbr = tlwh_to_tlbr(tlwh)
    assert np.allclose(tlbr, np.array([10.0, 20.0, 40.0, 60.0]))

    recovered = tlbr_to_tlwh(tlbr)
    assert np.allclose(recovered, tlwh)


def test_load_real_sequence(tmp_path: Path) -> None:
    seq_dir = Path("data/MOT17/train/MOT17-09-FRCNN")
    if not seq_dir.exists():
        pytest.skip("MOT17-09-FRCNN not present")

    seq = load_sequence(seq_dir)
    assert seq.name == "MOT17-09-FRCNN"
    assert seq.seq_length == 525
    assert len(seq.detections) > 0
    assert len(seq.ground_truths) > 0

    dets_by_frame = seq.get_detections_by_frame()
    assert 1 in dets_by_frame
    assert len(dets_by_frame[1]) > 0

    gt_by_frame = seq.get_ground_truths_by_frame(pedestrian_only=True)
    assert 1 in gt_by_frame
