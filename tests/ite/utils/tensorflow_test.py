# stdlib
from typing import Any

# third party
import numpy as np
import pytest
import tensorflow as tf

# ite absolute
import ite


@pytest.mark.parametrize(
    "y,y_hat,expected",
    [
        ([[1, 1], [10, 10]], [[20, 10], [200, 100]], 5050),
        ([[1, 2], [3, 4]], [[5, 16], [7, 18]], 100),
        (tf.fill([3, 3, 2], 5), tf.fill([3, 3, 2], 1), 0),
    ],
)
def test_tensorflow_PEHE(y: Any, y_hat: Any, expected: float) -> None:
    mock_y = tf.convert_to_tensor(y)
    mock_y_hat = tf.convert_to_tensor(y_hat)
    assert ite.utils.tensorflow.PEHE(mock_y, mock_y_hat).numpy() == expected


@pytest.mark.parametrize(
    "y,y_hat,expected",
    [
        ([[1, 1], [10, 10]], [[20, 10], [200, 100]], 55),
        ([[1, 2], [3, 4]], [[5, 16], [7, 18]], 10),
        (tf.fill([3, 3, 2], 5), tf.fill([3, 3, 2], 1), 0),
    ],
)
def test_tensorflow_ATE(y: Any, y_hat: Any, expected: float) -> None:
    mock_y = tf.convert_to_tensor(y)
    mock_y_hat = tf.convert_to_tensor(y_hat)
    assert ite.utils.tensorflow.ATE(mock_y, mock_y_hat).numpy() == expected


@pytest.mark.parametrize(
    "y,y_hat,expected",
    [
        ([[0, 0], [0, 0]], [[0, 0], [0, 0]], 1),
        ([[0, 0], [0, 0]], [[1, 1], [0, 0]], 1),
    ],
)
def test_tensorflow_RPol(y: Any, y_hat: Any, expected: float) -> None:
    mock_y = tf.convert_to_tensor(y)
    mock_y_hat = tf.convert_to_tensor(y_hat)
    t = [0] * mock_y.shape[-1]
    t[0] = 1
    mock_t = tf.convert_to_tensor(t)
    assert ite.utils.tensorflow.RPol(mock_t, mock_y, mock_y_hat) == expected


@pytest.mark.parametrize(
    "y,y_hat,expected",
    [
        ([[0, 0], [0, 0]], [[0, 0], [0, 0]], 0),
        ([[4, 5], [0, 0]], [[1, 1], [3, 3]], 0.999),
    ],
)
def test_tensorflow_ATT(y: Any, y_hat: Any, expected: float) -> None:
    mock_y = tf.convert_to_tensor(y)
    mock_y_hat = tf.convert_to_tensor(y_hat)
    t = [0] * mock_y.shape[-1]
    t[0] = 1
    mock_t = tf.convert_to_tensor(t)
    assert (
        np.abs(ite.utils.tensorflow.ATT(mock_t, mock_y, mock_y_hat) - expected) < 0.001
    )


@pytest.mark.parametrize(
    "shape",
    [
        [2, 2, 2],
        [5, 5],
        [5],
    ],
)
def test_xavier_init(shape: tf.Variable) -> None:
    out = ite.utils.tensorflow.xavier_init(tf.convert_to_tensor(shape))
    assert tf.math.count_nonzero(out) != 0
    assert out.shape == shape
