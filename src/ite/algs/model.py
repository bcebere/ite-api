# stdlib
from typing import Any

# third party
import pandas as pd

# ite absolute
from ite.algs.causal_multitask_gaussian_processes.model import CMGP
from ite.algs.ganite.model import Ganite
from ite.algs.ganite_torch.model import GaniteTorch
from ite.utils.metrics import HistoricMetrics
from ite.utils.metrics import Metrics


class Model:
    def __init__(self, model: str, *args: Any, **kwargs: Any) -> None:
        if model not in ["GANITE", "GANITE_TORCH", "CMGP"]:
            raise Exception(f"unsupported model: {model}")
        self.core: Any
        if model == "GANITE":
            self.core = Ganite(*args, **kwargs)
        elif model == "GANITE_TORCH":
            self.core = GaniteTorch(*args, **kwargs)
        elif model == "CMGP":
            self.core = CMGP(*args, **kwargs)

    def train(self, *args: Any, **kwargs: Any) -> HistoricMetrics:
        return self.core.train(*args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.core.predict(*args, **kwargs)

    def test(self, *args: Any, **kwargs: Any) -> Metrics:
        return self.core.test(*args, **kwargs)
