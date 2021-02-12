# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# stdlib
from typing import Tuple

# third party
import numpy as np
from scipy import stats


def PEHE(T_true: np.ndarray, T_est: np.ndarray) -> float:
    """
    Precision in Estimation of Heterogeneous Effect(Numpy version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    return np.sqrt(np.mean((T_true.reshape((-1, 1)) - T_est.reshape((-1, 1))) ** 2))


def mean_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Generate the mean and a confindence interval over observed data.
    Args:
        data: observed data
        confidence: confidence level
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)

    return m, h
