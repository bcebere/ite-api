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

    def add(self, key: str, val: float, group: str = "default") -> None:
        if group not in self.cache:
            self.cache[group] = {}
        if key not in self.cache[group]:
            self.cache[group][key] = []
        self.cache[group][key].append(val)

    def mean_confidence_interval(
        self, key: str, group: str = "default"
    ) -> Tuple[float, float]:
        cache_np = np.array(self.cache[group][key])[
            ~np.isnan(np.array(self.cache[group][key]))
        ]
        return utils.mean_confidence_interval(cache_np)

    def plot(self, plt: Any, with_ci: bool = False) -> None:
        fig, axs = plt.subplots(len(self.cache.keys()))
        for idx, group in enumerate(list(self.cache.keys())):
            for metric in self.cache[group]:
                axs[idx].plot(self.cache[group][metric], label=metric)
                if with_ci:
                    ci = self.mean_confidence_interval(metric, group)
                    plt.fill_between(
                        range(len(self.cache[group][metric])),
                        self.cache[group][metric] - ci[1],
                        self.cache[group][metric] + ci[1],
                        alpha=0.2,
                    )
            axs[idx].legend()
