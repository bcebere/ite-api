# stdlib
from typing import Any

# third party
import numpy as np
import pytest
import tensorflow.compat.v1 as tf

# ite absolute
import ite


@pytest.fixture(scope="function")
def session() -> tf.Session:
    sess = tf.InteractiveSession()  # noqa: F401, F841

    yield sess

    sess.close()


@pytest.mark.parametrize(
    "y,y_hat,expected",
    [
        ([[1, 1], [10, 10]], [[20, 10], [200, 100]], np.sqrt(5050)),
        ([[1, 2], [3, 4]], [[5, 16], [7, 18]], 10),
        ([[1, 1], [1, 1]], [[0, 0], [0, 0]], 0),
    ],
)
def test_tensorflow_PEHE(
    y: Any, y_hat: Any, expected: float, session: tf.Session
) -> None:
    mock_y = tf.convert_to_tensor(y)
    mock_y_hat = tf.convert_to_tensor(y_hat)

    assert ite.utils.tensorflow.sqrt_PEHE(mock_y, mock_y_hat).eval() == expected


@pytest.mark.parametrize(
    "y,y_hat,expected",
    [
        ([[1, 1], [10, 10]], [[20, 10], [200, 100]], 55),
        ([[1, 2], [3, 4]], [[5, 16], [7, 18]], 10),
        ([[1, 1], [1, 1]], [[0, 0], [0, 0]], 0),
    ],
)
def test_tensorflow_ATE(
    y: Any, y_hat: Any, expected: float, session: tf.Session
) -> None:
    mock_y = tf.convert_to_tensor(y)
    mock_y_hat = tf.convert_to_tensor(y_hat)
    assert ite.utils.tensorflow.ATE(mock_y, mock_y_hat).eval() == expected


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
