import numpy as np
import pytest

from src.tracker.kalman_filter import KalmanFilter


def test_kalman_filter_initiate() -> None:
    kf = KalmanFilter()
    measurement = np.array([100.0, 200.0, 0.5, 120.0])
    mean, cov = kf.initiate(measurement)

    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    assert np.allclose(mean[:4], measurement)
    assert np.allclose(mean[4:], 0.0)
    assert np.all(np.diag(cov) > 0.0)


def test_kalman_filter_predict_and_update() -> None:
    kf = KalmanFilter()
    measurement = np.array([100.0, 200.0, 0.5, 120.0])
    mean, cov = kf.initiate(measurement)

    pred_mean, pred_cov = kf.predict(mean, cov)
    assert pred_mean.shape == (8,)
    assert pred_cov.shape == (8, 8)

    next_meas = np.array([105.0, 202.0, 0.5, 120.0])
    updated_mean, updated_cov = kf.update(pred_mean, pred_cov, next_meas)

    assert updated_mean.shape == (8,)
    assert updated_cov.shape == (8, 8)
    assert np.all(np.diag(updated_cov) > 0.0)
