# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# stdlib
from typing import Tuple

# third party
import numpy as np
import scipy


def compute_PEHE(T_true: np.ndarray, T_est: np.ndarray) -> float:

    return np.sqrt(np.mean((T_true.reshape((-1, 1)) - T_est.reshape((-1, 1))) ** 2))


def mean_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)

    return m, h
