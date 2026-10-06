import numpy as np
import pytest

from src.tracking.kalman import KalmanFilter
from src.tracking.sort import Sort, linear_assignment


def test_kalman_predict_update_synthetic_straight_line() -> None:
    kf = KalmanFilter()

    # Ground truth straight-line trajectory: center moves by (+10, +5) per frame
    v_x, v_y = 10.0, 5.0
    w, h = 50.0, 100.0
    a = w / h
    c_x0, c_y0 = 100.0, 200.0

    init_meas = np.array([c_x0, c_y0, a, h])
    mean, cov = kf.initiate(init_meas)

    for step in range(1, 15):
        c_x = c_x0 + step * v_x
        c_y = c_y0 + step * v_y
        meas = np.array([c_x, c_y, a, h])

        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, meas)

    # Position tracking error should be minimal
    assert np.isclose(mean[0], c_x, atol=1.0)
    assert np.isclose(mean[1], c_y, atol=1.0)

    # Velocity estimates should converge to the constant velocity
    estimated_vx = mean[4]
    estimated_vy = mean[5]
    assert np.isclose(estimated_vx, v_x, atol=0.5)
    assert np.isclose(estimated_vy, v_y, atol=0.5)


def test_assignment_hand_built_cost_matrix() -> None:
    # 4 candidates vs 3 detections with known optimal costs
    cost_matrix = np.array([
        [0.10, 0.90, 0.85],
        [0.80, 0.20, 0.70],
        [0.75, 0.65, 0.30],
        [0.95, 0.90, 0.85],  # exceeds threshold
    ])

    threshold = 0.5
    matches, unmatched_a, unmatched_b = linear_assignment(cost_matrix, threshold)

    matched_pairs = [tuple(m) for m in matches]
    assert (0, 0) in matched_pairs
    assert (1, 1) in matched_pairs
    assert (2, 2) in matched_pairs

    assert len(matches) == 3
    assert list(unmatched_a) == [3]
    assert len(unmatched_b) == 0


def test_track_surviving_3_frame_gap() -> None:
    # Max age allows keeping track alive across occlusion gaps
    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
    tracker.reset_id_counter()

    # Frames 1-3: initialize and confirm track with consistent motion
    det1 = np.array([[100.0, 100.0, 150.0, 200.0, 0.9]])
    out1 = tracker.update(det1)

    det2 = np.array([[105.0, 100.0, 155.0, 200.0, 0.9]])
    out2 = tracker.update(det2)

    det3 = np.array([[110.0, 100.0, 160.0, 200.0, 0.9]])
    out3 = tracker.update(det3)

    assert len(out3) == 1
    original_track_id = int(out3[0, 4])

    # Frames 4-6: empty detections (3-frame gap)
    empty = np.empty((0, 5), dtype=float)
    tracker.update(empty)
    tracker.update(empty)
    tracker.update(empty)

    # Track must still exist in internal tracker pool with time_since_update == 3
    assert len(tracker.trackers) == 1
    assert tracker.trackers[0].id == original_track_id
    assert tracker.trackers[0].time_since_update == 3

    # Frame 7: re-appearance near predicted linear trajectory
    det7 = np.array([[130.0, 100.0, 180.0, 200.0, 0.9]])
    out7 = tracker.update(det7)

    # Track must associate and retain its original ID
    assert len(out7) == 1
    assert int(out7[0, 4]) == original_track_id
