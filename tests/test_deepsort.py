import numpy as np
import pytest

from src.tracking.deepsort import DeepSort, DeepTrack, cosine_distance, mahalanobis_gating
from src.tracking.kalman import KalmanFilter
from src.tracking.sort import Sort


def test_crossing_paths_deepsort_preserves_ids_while_sort_swaps() -> None:
    # Bounding box dimensions (width=40, height=100)
    w, h = 40.0, 100.0

    def make_box(cx: float) -> list[float]:
        return [cx - w / 2, 100.0 - h / 2, cx + w / 2, 100.0 + h / 2]

    # Two distinct orthogonal synthetic embeddings (L2-normalized)
    feat1 = np.zeros(512, dtype=np.float32)
    feat1[0] = 1.0
    feat2 = np.zeros(512, dtype=np.float32)
    feat2[1] = 1.0

    # Crossing trajectories:
    # Object 1 moves left-to-right (from 100 to 170)
    # Object 2 moves right-to-left (from 180 to 110)
    # At frame 4, they cross near x=140.
    traj1 = [100.0, 115.0, 130.0, 148.0, 155.0, 170.0]
    traj2 = [180.0, 165.0, 150.0, 132.0, 125.0, 110.0]

    # 1. Run through SORT (motion-only, IoU association)
    sort_tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.1)
    sort_tracker.reset_id_counter()

    sort_final_positions = {}
    for f in range(len(traj1)):
        dets = np.array([
            make_box(traj1[f]) + [0.9],
            make_box(traj2[f]) + [0.9],
        ])
        res = sort_tracker.update(dets)
        if f == len(traj1) - 1:
            for r in res:
                sort_final_positions[int(r[4])] = (r[0] + r[2]) / 2

    # In SORT, the crossing causes an ID switch:
    # Track 1 (which started on object 1 on the left at x=100) is now following object 2 on the left (x ~ 116)
    # Track 2 (which started on object 2 on the right at x=180) is now following object 1 on the right (x ~ 164)
    assert 1 in sort_final_positions and 2 in sort_final_positions
    assert sort_final_positions[1] < 125.0
    assert sort_final_positions[2] > 155.0

    # 2. Run through DeepSORT (appearance-based matching cascade + Mahalanobis gating)
    deepsort_tracker = DeepSort(
        max_age=10,
        min_hits=3,
        iou_threshold=0.1,
        max_cosine_distance=0.4,
    )
    deepsort_tracker.reset_id_counter()

    deepsort_final_positions = {}
    for f in range(len(traj1)):
        dets = np.array([make_box(traj1[f]), make_box(traj2[f])])
        feats = np.array([feat1, feat2])
        scores = np.array([0.9, 0.9])
        res = deepsort_tracker.update(dets, feats, scores)
        if f == len(traj1) - 1:
            for r in res:
                deepsort_final_positions[int(r[4])] = (r[0] + r[2]) / 2

    # In DeepSORT, distinct appearance embeddings prevent ID swap:
    # Track 1 remains with object 1 (on the right, x ~ 169)
    # Track 2 remains with object 2 (on the left, x ~ 111)
    assert 1 in deepsort_final_positions and 2 in deepsort_final_positions
    assert deepsort_final_positions[1] > 155.0
    assert deepsort_final_positions[2] < 125.0


def test_cosine_distance_computation() -> None:
    f1 = np.zeros(512, dtype=np.float32)
    f1[0] = 1.0

    f2 = np.zeros(512, dtype=np.float32)
    f2[1] = 1.0

    # Gallery for track 0 has f1, gallery for track 1 has f2
    galleries = [np.array([f1]), np.array([f2])]
    dets = np.array([f1, f2])

    dist = cosine_distance(galleries, dets)
    assert dist.shape == (2, 2)
    assert np.isclose(dist[0, 0], 0.0, atol=1e-5)
    assert np.isclose(dist[0, 1], 1.0, atol=1e-5)
    assert np.isclose(dist[1, 0], 1.0, atol=1e-5)
    assert np.isclose(dist[1, 1], 0.0, atol=1e-5)


def test_gallery_budget_limit() -> None:
    budget = 5
    track = DeepTrack(
        bbox_xyxy=np.array([10.0, 10.0, 50.0, 100.0]),
        feature=np.ones(512, dtype=np.float32) * 0.1,
        nn_budget=budget,
    )

    for i in range(15):
        feat = np.ones(512, dtype=np.float32) * float(i + 1)
        track.update(np.array([10.0, 10.0, 50.0, 100.0]), feat)

    gallery = track.get_gallery()
    assert gallery.shape[0] == budget
    # Oldest features must have been evicted, retaining latest additions
    assert np.isclose(gallery[-1, 0], 15.0)


def test_mahalanobis_gating() -> None:
    kf = KalmanFilter()
    track = DeepTrack(
        bbox_xyxy=np.array([100.0, 100.0, 150.0, 200.0]),
        feature=np.zeros(512, dtype=np.float32),
    )

    # Nearby measurement vs distant measurement
    measurements = np.array([
        [125.0, 150.0, 0.5, 100.0],  # near
        [800.0, 800.0, 0.5, 100.0],  # distant
    ])
    cost_matrix = np.zeros((1, 2), dtype=float)
    gated = mahalanobis_gating(kf, [track], measurements, cost_matrix, gating_threshold=9.4877)

    # Close measurement stays 0.0, distant measurement is gated to 1e5
    assert np.isclose(gated[0, 0], 0.0)
    assert gated[0, 1] >= 1e4
