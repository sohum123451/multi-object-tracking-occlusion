import numpy as np
import pytest

from src.tracker.byte_tracker import ByteTracker
from src.tracker.sort_tracker import SortTracker


def test_sort_tracker_step() -> None:
    tracker = SortTracker(det_thresh=0.5, max_age=5, min_hits=1)

    frame1_dets = np.array([
        [10.0, 10.0, 50.0, 50.0, 0.9],
        [100.0, 100.0, 140.0, 140.0, 0.85],
    ])
    tracks1 = tracker.update(frame1_dets)
    assert len(tracks1) == 2

    # Frame 2: slightly shifted boxes
    frame2_dets = np.array([
        [12.0, 11.0, 52.0, 51.0, 0.9],
        [102.0, 101.0, 142.0, 141.0, 0.85],
    ])
    tracks2 = tracker.update(frame2_dets)
    assert len(tracks2) == 2

    # Check ID persistence
    ids_f1 = {t.track_id for t in tracks1}
    ids_f2 = {t.track_id for t in tracks2}
    assert ids_f1 == ids_f2


def test_bytetrack_two_stage_association() -> None:
    tracker = ByteTracker(track_thresh=0.6, track_buffer=10)

    # Frame 1: high confidence detection initializes track
    f1_dets = np.array([
        [20.0, 20.0, 60.0, 80.0, 0.9],
    ])
    tracks1 = tracker.update(f1_dets)
    assert len(tracks1) == 1
    t_id = tracks1[0].track_id

    # Frame 2: detection with score 0.4 (low score stage)
    f2_dets = np.array([
        [22.0, 21.0, 62.0, 81.0, 0.4],
    ])
    tracks2 = tracker.update(f2_dets)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == t_id
