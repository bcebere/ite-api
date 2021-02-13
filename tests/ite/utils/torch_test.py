# stdlib
from typing import Any

# third party
import numpy as np
import pytest
import torch

# ite absolute
import ite


@pytest.mark.parametrize(
    "y,y_hat",
    [
        ([[1, 1], [10, 10]], [[20, 10], [200, 100]]),
        ([[1, 2], [3, 4]], [[5, 16], [7, 18]]),
        ([[1, 1], [1, 1]], [[0, 0], [0, 0]]),
    ],
)
def test_torch_PEHE(y: Any, y_hat: Any) -> None:
    mock_y = torch.FloatTensor(y)
    mock_y_hat = torch.FloatTensor(y_hat)

    assert (
        abs(
            ite.utils.torch.sqrt_PEHE(mock_y, mock_y_hat).numpy()
            - ite.utils.numpy.sqrt_PEHE(np.array(y), np.array(y_hat))
        )
        < 0.001
    )


@pytest.mark.parametrize(
    "y,y_hat,expected",
    [
        ([[1, 1], [10, 10]], [[20, 10], [200, 100]], 55),
        ([[1, 2], [3, 4]], [[5, 16], [7, 18]], 10),
        ([[1, 1], [1, 1]], [[0, 0], [0, 0]], 0),
    ],
)
def test_tensorflow_ATE(y: Any, y_hat: Any, expected: float) -> None:
    mock_y = torch.FloatTensor(y)
    mock_y_hat = torch.FloatTensor(y_hat)
    assert ite.utils.torch.ATE(mock_y, mock_y_hat).numpy() == expected


def test_sigmoid_cross_entropy_with_logits() -> None:

    out = ite.utils.torch.sigmoid_cross_entropy_with_logits(
        torch.FloatTensor([0, 0]), torch.FloatTensor([1, 1])
    )
    assert out.shape == torch.Size()
