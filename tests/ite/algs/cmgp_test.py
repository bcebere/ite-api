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
        max_gp_iterations=50,
    )
    assert model is not None

    for experiment in range(3):
        dataset = ds.load(
            "twins",
            train_ratio,
            downsample=1000,
        )

        # metrics = model.train(Train_X, Train_T, Train_Y, Opt_Train_Y ,Test_X, Test_Y)
        metrics = model.train(*dataset)

        test_metrics = model.test(Test_X, Test_Y)
        assert 0.2 < test_metrics.sqrt_PEHE() and test_metrics.sqrt_PEHE() < 0.4

    metrics.plot(plt, with_ci=True)
    try:
        metrics.plot(plt, with_ci=True, thresholds=[0.2, 0.25, 0.3, 0.35])
    except BaseException as e:
        print("failed to plot(maybe rerun with --plots):", e)
