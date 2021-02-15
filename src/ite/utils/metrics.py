# stdlib
from typing import Any
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


class HistoricMetrics:
    def __init__(self) -> None:
        self.cache: dict = {}

    def add(self, key: str, val: float) -> None:
        if key not in self.cache:
            self.cache[key] = []
        self.cache[key].append(val)

    def mean_confidence_interval(self, key: str) -> Tuple[float, float]:
        return utils.mean_confidence_interval(self.cache[key])

    def plot(self, plt: Any) -> None:
        pass
