# stdlib
from typing import Any
from typing import List
from typing import Tuple

# third party
import numpy as np

# ite absolute
import ite.utils.numpy as utils


class Metrics:
    """
    Helper class for comparing estimates to the reference values.
    """

    def __init__(self, estimated: np.ndarray, actual: np.ndarray) -> None:
        self.estimated = estimated
        self.actual = actual

    def sqrt_PEHE(self) -> float:
        """
        Precision in Estimation of Heterogeneous Effect(Numpy version).
        PEHE reflects the ability to capture individual variation in treatment effects.
        """
        return utils.sqrt_PEHE(self.actual, self.estimated)

    def ATE(self) -> float:
        """
        Average Treatment Effect.
        ATE measures what is the expected causal effect of the treatment across all individuals in the population.
        """

        return utils.ATE(self.actual, self.estimated)

    def MSE(self) -> float:
        """
        Mean squared error.
        Computes the mean squared error between labels and predictions.
        """

        return np.mean(utils.squared_difference(self.estimated, self.actual))

    def worst_mistakes(self, top_k: int = 5) -> List[int]:
        """
        Helper for visualising the entries with the most significant error reported to the PEHE metric.
        """

        sq_diff = utils.squared_difference(
            (self.actual[:, 1] - self.actual[:, 0]),
            (self.estimated[:, 1] - self.estimated[:, 0]),
        )
        return sq_diff.argsort()[-top_k:][::-1]

    def print(self) -> None:
        """
        Helper for printing a summary for the metrics.
        """
        print(f"sqrt_PHE = {self.sqrt_PEHE():.3f}")
        print(f"ATE = {self.ATE():.3f}")
        print(f"MSE = {self.MSE():.3f}")
        print(f"Top 5 worst mistakes(indices) = {self.worst_mistakes()}")


class HistoricMetrics:
    """
    Helper class for visualizing the evolution of specific metrics(like the training loss).
    """

    def __init__(self) -> None:
        self.cache: dict = {}

    def add(self, key: str, val: float, group: str = "default") -> "HistoricMetrics":
        """
        Add a new measure to the group:key list.
        The "group" parameter plots keys in the same diagram.
        The "key" parameter defines which list of values we append to.
        """
        if group not in self.cache:
            self.cache[group] = {}
        if key not in self.cache[group]:
            self.cache[group][key] = []
        self.cache[group][key].append(val)

        return self

    def mean_confidence_interval(
        self, key: str, group: str = "default"
    ) -> Tuple[float, float]:
        """
        Compute the mean and confidence interval for the buffered metrics.
        """
        cache_np = np.array(self.cache[group][key])[
            ~np.isnan(np.array(self.cache[group][key]))
        ]
        return utils.mean_confidence_interval(cache_np)

    def print(self) -> None:
        """
        Helper for printing a summary for the metrics.
        """
        for group in self.cache:
            print(f"{group}:")
            for metric in self.cache[group]:
                ci = self.mean_confidence_interval(metric, group)
                print(f" - {metric}: {ci[0]:.3f} +/- {ci[1]:.3f}")

    def plot(
        self, plt: Any, with_ci: bool = False, thresholds: List[float] = []
    ) -> None:
        """
        Helper for plotting the buffered metrics.
        """
        fig, axs = plt.subplots(len(self.cache.keys()))
        fig.set_size_inches(15, 5.5 * len(self.cache.keys()))
        for idx, group in enumerate(list(self.cache.keys())):
            for metric in self.cache[group]:
                for thresh in thresholds:
                    axs[idx].hlines(
                        thresh,
                        xmin=0,
                        xmax=len(self.cache[group][metric]) - 1,
                        linestyles="dotted",
                        colors="r",
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
