# stdlib
from typing import Any

# third party
import numpy as np
import pytest

# ite absolute
import ite


@pytest.mark.parametrize(
    "y,y_hat,expected",
    [
        ([[1, 1], [10, 10]], [[20, 10], [200, 100]], 5050),
        ([[1, 2], [3, 4]], [[5, 16], [7, 18]], 100),
    ],
)
def test_numpy_PEHE(y: Any, y_hat: Any, expected: float) -> None:
    mock_y = np.array(y)
    mock_y_hat = np.array(y_hat)
    assert ite.utils.numpy.PEHE(mock_y, mock_y_hat) > 0


@pytest.mark.parametrize(
    "y,y_hat,expected",
    [
        ([[0, 0], [0, 0]], [[0, 0], [0, 0]], 1),
        ([[0, 0], [0, 0]], [[1, 1], [0, 0]], 1),
    ],
)
def test_tensorflow_RPol(y: Any, y_hat: Any, expected: float) -> None:
    mock_y = np.array(y)
    mock_y_hat = np.array(y_hat)
    t = [0] * mock_y.shape[-1]
    t[0] = 1
    mock_t = np.array(t)
    assert ite.utils.numpy.RPol(mock_t, mock_y, mock_y_hat) == expected


@pytest.mark.parametrize(
    "y,y_hat,expected",
    [
        ([[0, 0], [0, 0]], [[0, 0], [0, 0]], 0),
        ([[4, 5], [0, 0]], [[1, 1], [3, 3]], 0.999),
    ],
)
def test_tensorflow_ATT(y: Any, y_hat: Any, expected: float) -> None:
    mock_y = np.array(y)
    mock_y_hat = np.array(y_hat)
    t = [0] * mock_y.shape[-1]
    t[0] = 1
    mock_t = np.array(t)
    assert np.abs(ite.utils.numpy.ATT(mock_t, mock_y, mock_y_hat) - expected) < 0.001


def test_mean_confidence_interval() -> None:
    s = np.random.uniform(-1, 1, 1000)
    assert np.array(ite.utils.numpy.mean_confidence_interval(s)).shape == (2,)
