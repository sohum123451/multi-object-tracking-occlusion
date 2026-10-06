from pathlib import Path
import numpy as np
import pytest

from src.data.mot_loader import GroundTruth, MOTSequence
from src.metrics.mot_evaluator import (
    FramePrediction,
    evaluate_sequence,
)


def create_synthetic_sequence() -> MOTSequence:
    # 2 objects moving over 3 frames
    gts = [
        # Frame 1
        GroundTruth(frame=1, track_id=1, bbox_tlwh=np.array([10.0, 10.0, 20.0, 50.0]), conf=1.0, class_id=1, visibility=1.0),
        GroundTruth(frame=1, track_id=2, bbox_tlwh=np.array([100.0, 10.0, 20.0, 50.0]), conf=1.0, class_id=1, visibility=1.0),
        # Frame 2
        GroundTruth(frame=2, track_id=1, bbox_tlwh=np.array([12.0, 10.0, 20.0, 50.0]), conf=1.0, class_id=1, visibility=1.0),
        GroundTruth(frame=2, track_id=2, bbox_tlwh=np.array([102.0, 10.0, 20.0, 50.0]), conf=1.0, class_id=1, visibility=1.0),
        # Frame 3
        GroundTruth(frame=3, track_id=1, bbox_tlwh=np.array([14.0, 10.0, 20.0, 50.0]), conf=1.0, class_id=1, visibility=1.0),
        GroundTruth(frame=3, track_id=2, bbox_tlwh=np.array([104.0, 10.0, 20.0, 50.0]), conf=1.0, class_id=1, visibility=1.0),
    ]

    return MOTSequence(
        name="SYNTHETIC",
        seq_dir=Path("synthetic"),
        seq_length=3,
        frame_rate=30,
        im_width=1920,
        im_height=1080,
        detections=[],
        ground_truths=gts,
    )


def test_evaluate_perfect_tracking() -> None:
    seq = create_synthetic_sequence()
    preds = {
        1: [
            FramePrediction(frame=1, track_id=1, bbox_tlwh=np.array([10.0, 10.0, 20.0, 50.0]), score=1.0),
            FramePrediction(frame=1, track_id=2, bbox_tlwh=np.array([100.0, 10.0, 20.0, 50.0]), score=1.0),
        ],
        2: [
            FramePrediction(frame=2, track_id=1, bbox_tlwh=np.array([12.0, 10.0, 20.0, 50.0]), score=1.0),
            FramePrediction(frame=2, track_id=2, bbox_tlwh=np.array([102.0, 10.0, 20.0, 50.0]), score=1.0),
        ],
        3: [
            FramePrediction(frame=3, track_id=1, bbox_tlwh=np.array([14.0, 10.0, 20.0, 50.0]), score=1.0),
            FramePrediction(frame=3, track_id=2, bbox_tlwh=np.array([104.0, 10.0, 20.0, 50.0]), score=1.0),
        ],
    }

    metrics = evaluate_sequence(seq, preds)
    assert np.isclose(metrics.mota, 1.0)
    assert np.isclose(metrics.motp, 1.0)
    assert np.isclose(metrics.idf1, 1.0)
    assert metrics.num_false_positives == 0
    assert metrics.num_misses == 0
    assert metrics.num_switches == 0
    assert metrics.mostly_tracked == 2
    assert metrics.mostly_lost == 0


def test_evaluate_id_switch() -> None:
    seq = create_synthetic_sequence()
    # In frame 3, prediction ID for object 1 switches from 1 to 99
    preds = {
        1: [
            FramePrediction(frame=1, track_id=1, bbox_tlwh=np.array([10.0, 10.0, 20.0, 50.0]), score=1.0),
            FramePrediction(frame=1, track_id=2, bbox_tlwh=np.array([100.0, 10.0, 20.0, 50.0]), score=1.0),
        ],
        2: [
            FramePrediction(frame=2, track_id=1, bbox_tlwh=np.array([12.0, 10.0, 20.0, 50.0]), score=1.0),
            FramePrediction(frame=2, track_id=2, bbox_tlwh=np.array([102.0, 10.0, 20.0, 50.0]), score=1.0),
        ],
        3: [
            FramePrediction(frame=3, track_id=99, bbox_tlwh=np.array([14.0, 10.0, 20.0, 50.0]), score=1.0),
            FramePrediction(frame=3, track_id=2, bbox_tlwh=np.array([104.0, 10.0, 20.0, 50.0]), score=1.0),
        ],
    }

    metrics = evaluate_sequence(seq, preds)
    assert metrics.num_switches == 1
    assert metrics.num_misses == 0
    assert metrics.num_false_positives == 0
