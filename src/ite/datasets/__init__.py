# stdlib
import os
from pathlib import Path
from typing import List

# third party
import numpy as np

# ite relative
from .loaders import twins

DATA_PATH = Path(os.path.dirname(__file__)) / Path("data")

try:
    os.mkdir(DATA_PATH)
except BaseException:
    pass


def load(dataset: str, train_split: float = 0.8) -> List[np.ndarray]:
    """
    Input:
        dataset: the name of the dataset to load
    Outputs:
        - Train_X, Test_X: Train and Test features
        - Train_Y: Observable outcomes
        - Train_T: Assigned treatment
        - Opt_Train_Y, Test_Y: Potential outcomes.
    """
    if dataset == "twins":
        return twins.load(DATA_PATH, train_split)
    else:
        raise Exception("Unsupported dataset")
