# stdlib
from typing import Optional

# third party
import pytest

# ite absolute
import ite.datasets as ds


def test_sanity() -> None:
    with pytest.raises(BaseException):
        ds.load("test")


@pytest.mark.parametrize(
    "train_ratio",
    [0.1, 0.5, 0.8],
)
@pytest.mark.parametrize(
    "downsample",
    [None, 100, 1000],
)
def test_dataset_twins_load(train_ratio: float, downsample: Optional[int]) -> None:
    # Data Input (11400 patients, 30 features, 2 potential outcomes)

    total = 11400
    feat_count = 30
    outcomes = 2

    [Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y] = ds.load(
        "twins",
        train_ratio,
        downsample=downsample,
    )

    train_cnt = int(total * train_ratio)
    test_cnt = total - train_cnt
    if downsample:
        train_cnt = min(downsample, train_cnt)
        test_cnt = min(downsample, test_cnt)

    assert Train_X.shape == (train_cnt, feat_count)
    assert Train_T.shape == (train_cnt,)
    assert Train_Y.shape == (train_cnt,)
    assert Opt_Train_Y.shape == (train_cnt, outcomes)
    assert Test_X.shape == (test_cnt, feat_count)
    assert Test_Y.shape == (test_cnt, outcomes)
