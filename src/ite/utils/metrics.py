# stdlib
from typing import Tuple

# third party
import numpy as np

# ite absolute
import ite.utils.numpy as utils


class Metrics:
    def __init__(self, estimated: np.ndarray, actual: np.ndarray) -> None:
        self.estimated = estimated
        self.actual = actual

    def sqrt_PEHE(self) -> float:
        return utils.sqrt_PEHE(self.estimated, self.actual)

    def ATE(self) -> float:
        return utils.ATE(self.estimated, self.actual)

    def mean_confidence_interval(self) -> Tuple[float, float]:
        return utils.mean_confidence_interval(self.estimated, self.actual)
