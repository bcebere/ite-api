# stdlib
from typing import List

# third party
from loaders import twins
import numpy as np


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
        return twins.load(train_split)
    else:
        raise Exception("Unsupported dataset")
