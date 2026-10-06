from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_ious(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.empty((boxes_a.shape[0], boxes_b.shape[0]), dtype=float)

    boxes_a = np.asarray(boxes_a, dtype=float)
    boxes_b = np.asarray(boxes_b, dtype=float)

    x1 = np.maximum(boxes_a[:, 0, None], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, 1, None], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, 3, None], boxes_b[None, :, 3])

    intersection_w = np.maximum(0.0, x2 - x1)
    intersection_h = np.maximum(0.0, y2 - y1)
    intersection = intersection_w * intersection_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    union = np.maximum(union, 1e-12)

    return intersection / union


def iou_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    ious = bbox_ious(boxes_a, boxes_b)
    return 1.0 - ious


def linear_assignment(
    cost_matrix: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: List[Tuple[int, int]] = []
    unmatched_a: List[int] = []
    unmatched_b: List[int] = []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= threshold:
            matches.append((r, c))

    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}

    for r in range(cost_matrix.shape[0]):
        if r not in matched_rows:
            unmatched_a.append(r)

    for c in range(cost_matrix.shape[1]):
        if c not in matched_cols:
            unmatched_b.append(c)

    matches_arr = (
        np.array(matches, dtype=int)
        if len(matches) > 0
        else np.empty((0, 2), dtype=int)
    )
    return matches_arr, np.array(unmatched_a, dtype=int), np.array(unmatched_b, dtype=int)
