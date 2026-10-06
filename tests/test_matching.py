import numpy as np
import pytest

from src.tracker.matching import bbox_ious, iou_distance, linear_assignment


def test_bbox_ious_identical() -> None:
    boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
    ious = bbox_ious(boxes, boxes)
    assert ious.shape == (1, 1)
    assert np.isclose(ious[0, 0], 1.0)


def test_bbox_ious_disjoint() -> None:
    box1 = np.array([[0.0, 0.0, 10.0, 10.0]])
    box2 = np.array([[20.0, 20.0, 30.0, 30.0]])
    ious = bbox_ious(box1, box2)
    assert np.isclose(ious[0, 0], 0.0)


def test_bbox_ious_known_overlap() -> None:
    # box1: [0, 0, 10, 10] -> area 100
    # box2: [5, 0, 15, 10] -> area 100
    # intersection: [5, 0, 10, 10] -> area 50
    # union: 100 + 100 - 50 = 150
    # iou: 50 / 150 = 1/3
    box1 = np.array([[0.0, 0.0, 10.0, 10.0]])
    box2 = np.array([[5.0, 0.0, 15.0, 10.0]])
    ious = bbox_ious(box1, box2)
    assert np.isclose(ious[0, 0], 1.0 / 3.0)


def test_linear_assignment() -> None:
    cost = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
    ])
    matches, u_a, u_b = linear_assignment(cost, threshold=0.5)
    assert len(matches) == 2
    assert (0, 0) in [tuple(m) for m in matches]
    assert (1, 1) in [tuple(m) for m in matches]
    assert len(u_a) == 0
    assert len(u_b) == 0


def test_linear_assignment_with_unmatched() -> None:
    cost = np.array([
        [0.1, 0.9],
        [0.8, 0.7],  # row 1 minimum exceeds threshold 0.5
    ])
    matches, u_a, u_b = linear_assignment(cost, threshold=0.5)
    assert len(matches) == 1
    assert matches[0, 0] == 0 and matches[0, 1] == 0
    assert 1 in u_a
    assert 1 in u_b
