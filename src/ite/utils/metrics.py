# stdlib
from typing import Any
from typing import List
from typing import Tuple

# third party
from matplotlib.ticker import MultipleLocator
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

    def print(self) -> None:
        print(f"sqrt_PHE = {self.sqrt_PEHE()}")
        print(f"ATE = {self.ATE()}")


class HistoricMetrics:
    def __init__(self) -> None:
        self.cache: dict = {}

    def add(self, key: str, val: float, group: str = "default") -> "HistoricMetrics":
        if group not in self.cache:
            self.cache[group] = {}
        if key not in self.cache[group]:
            self.cache[group][key] = []
        self.cache[group][key].append(val)

        return self

    def mean_confidence_interval(
        self, key: str, group: str = "default"
    ) -> Tuple[float, float]:
        cache_np = np.array(self.cache[group][key])[
            ~np.isnan(np.array(self.cache[group][key]))
        ]
        return utils.mean_confidence_interval(cache_np)

    def plot(
        self, plt: Any, with_ci: bool = False, thresholds: List[float] = []
    ) -> None:
        fig, axs = plt.subplots(len(self.cache.keys()))
        for idx, group in enumerate(list(self.cache.keys())):
            axs[idx].yaxis.set_major_locator(MultipleLocator(0.05))

            for metric in self.cache[group]:
                for thresh in thresholds:
                    axs[idx].hlines(
                        thresh,
                        xmin=0,
                        xmax=len(self.cache[group][metric]) - 1,
                        linestyles="dotted",
                    )
                break

            for metric in self.cache[group]:
                axs[idx].set_title(group)
                axs[idx].plot(self.cache[group][metric], label=metric)
                if with_ci:
                    ci = self.mean_confidence_interval(metric, group)
                    axs[idx].fill_between(
                        range(len(self.cache[group][metric])),
                        self.cache[group][metric] - ci[1],
                        self.cache[group][metric] + ci[1],
                        alpha=0.2,
                    )
            axs[idx].legend()
