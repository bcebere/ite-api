# Dataset import (Jinsung Yoon, 10/11/2017)

# stdlib
from pathlib import Path
from typing import List

# third party
import numpy as np
from scipy.special import expit

# ite absolute
from ite.utils.network import download_if_needed

DATASET = "Twin_Data.csv.gz"
URL = "https://bitbucket.org/mvdschaar/mlforhealthlabpub/raw/0b0190bcd38a76c405c805f1ca774971fcd85233/data/twins/Twin_Data.csv.gz"  # noqa: E501


def preprocess(fn_csv: Path, train_rate: float = 0.8) -> List[np.ndarray]:
    """
    Input:
        fn_csv: Path to the input CSV to load
        train_rate: Train/Test split
    Outputs:
        - Train_X, Test_X: Train and Test features
        - Train_Y: Observable outcomes
        - Train_T: Assigned treatment
        - Opt_Train_Y, Test_Y: Potential outcomes.
    """
    # Data Input (11400 patients, 30 features, 2 potential outcomes)
    Data = np.loadtxt(fn_csv, delimiter=",", skiprows=1)

    # Features
    X = Data[:, :30]

    # Feature dimensions and patient numbers
    Dim = len(X[0])
    No = len(X)

    # Labels
    Opt_Y = Data[:, 30:]

    for i in range(2):
        idx = np.where(Opt_Y[:, i] > 365)
        Opt_Y[idx, i] = 365

    Opt_Y = 1 - (Opt_Y / 365.0)

    # Patient Treatment Assignment
    coef = 0 * np.random.uniform(-0.01, 0.01, size=[Dim, 1])
    Temp = expit(np.matmul(X, coef) + np.random.normal(0, 0.01, size=[No, 1]))

    Temp = Temp / (2 * np.mean(Temp))

    Temp[Temp > 1] = 1

    T = np.random.binomial(1, Temp, [No, 1])
    T = T.reshape(
        [
            No,
        ]
    )

    # Observable outcomes
    Y = np.zeros([No, 1])

    # Output
    Y = np.transpose(T) * Opt_Y[:, 1] + np.transpose(1 - T) * Opt_Y[:, 0]
    Y = np.transpose(Y)
    Y = np.reshape(
        Y,
        [
            No,
        ],
    )

    # Train / Test Division
    temp = np.random.permutation(No)
    Train_No = int(train_rate * No)
    train_idx = temp[:Train_No]
    test_idx = temp[Train_No:No]

    Train_X = X[train_idx, :]
    Train_T = T[train_idx]
    Train_Y = Y[train_idx]
    Opt_Train_Y = Opt_Y[train_idx, :]

    Test_X = X[test_idx, :]
    Test_Y = Opt_Y[test_idx, :]

    return [Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y]


def load(data_path: Path, train_split: float = 0.8) -> List[np.ndarray]:
    csv = data_path / DATASET

    download_if_needed(csv, URL)
    return preprocess(csv, train_split)
