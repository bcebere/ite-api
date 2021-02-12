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


def test_mean_confidence_interval() -> None:
    s = np.random.uniform(-1, 1, 1000)
    assert np.array(ite.utils.numpy.mean_confidence_interval(s)).shape == (2,)
