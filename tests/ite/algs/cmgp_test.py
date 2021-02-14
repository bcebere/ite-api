# stdlib
from typing import Any

# ite absolute
import ite.algs.causal_multitask_gaussian_processes.model as alg
import ite.datasets as ds


def test_sanity() -> None:
    assert alg.CMGP() is not None


def test_cmgp_short_training(
    plt: Any,
) -> None:
    train_ratio = 0.8

    [Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y] = ds.load(
        "twins",
        train_ratio,
        downsample=1000,
    )

    dim = len(Train_X[0])
    dim_outcome = Test_Y.shape[1]

    model = alg.CMGP(
        dim=dim,
        dim_outcome=dim_outcome,
        max_gp_iterations=500,
    )
    assert model is not None

    train_metrics = model.fit(Train_X, Train_T, Train_Y, Test_X, Test_Y)

    assert "Loss_sqrt_PEHE" in train_metrics
    assert "Loss_ATE" in train_metrics

    assert (
        0.28 < train_metrics["Loss_sqrt_PEHE"]
        and train_metrics["Loss_sqrt_PEHE"] < 0.31
    )

    predicted = model.predict(Test_X)

    assert predicted.shape == (Test_X.shape[0], 2)

    test_metrics = model.test(Test_X, Test_Y)
    assert "sqrt_PEHE" in test_metrics
    assert "ATE" in test_metrics

    assert 0.28 < test_metrics["sqrt_PEHE"] and test_metrics["sqrt_PEHE"] < 0.31
