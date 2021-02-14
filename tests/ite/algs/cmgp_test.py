# stdlib
from typing import Any

# third party
import numpy as np

# ite absolute
import ite.algs.causal_multitask_gaussian_processes.model as alg
import ite.datasets as ds
from ite.utils.numpy import mean_confidence_interval


def test_sanity() -> None:
    assert alg.CMGP() is not None


def test_cmgp_short_training(
    plt: Any,
) -> None:
    train_ratio = 0.8

    plot_experiments: dict = {
        "train_sqrt_PEHE": [],
        "test_sqrt_PEHE": [],
    }

    for experiment in range(10):
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

        model.fit(Train_X, Train_T, Train_Y, Test_X, Test_Y)

        train_metrics = model.test(Train_X, Opt_Train_Y)
        assert "sqrt_PEHE" in train_metrics
        plot_experiments["train_sqrt_PEHE"].append(train_metrics["sqrt_PEHE"])

        predicted = model.predict(Test_X)

        assert predicted.shape == (Test_X.shape[0], 2)

        test_metrics = model.test(Test_X, Test_Y)
        assert "sqrt_PEHE" in test_metrics
        plot_experiments["test_sqrt_PEHE"].append(test_metrics["sqrt_PEHE"])

        assert 0.2 < test_metrics["sqrt_PEHE"] and test_metrics["sqrt_PEHE"] < 0.4

    train_sqrt_PEHE = plot_experiments["train_sqrt_PEHE"]
    test_sqrt_PEHE = plot_experiments["test_sqrt_PEHE"]

    PEHE_train_np = np.array(train_sqrt_PEHE)[~np.isnan(np.array(train_sqrt_PEHE))]
    PEHE_test_np = np.array(test_sqrt_PEHE)[~np.isnan(np.array(test_sqrt_PEHE))]

    train_ci = mean_confidence_interval(PEHE_train_np)
    test_ci = mean_confidence_interval(PEHE_test_np)

    try:
        fig, axs = plt.subplots(2)
        axs[0].plot(train_sqrt_PEHE, color="r", label="train_sqrt_PEHE")
        axs[0].fill_between(
            range(len(train_sqrt_PEHE)),
            train_sqrt_PEHE - train_ci[1],
            train_sqrt_PEHE + train_ci[1],
            color="r",
            alpha=0.2,
        )
        axs[0].legend()
        axs[1].plot(test_sqrt_PEHE, label="test_sqrt_PEHE")
        axs[1].fill_between(
            range(len(test_sqrt_PEHE)),
            test_sqrt_PEHE - test_ci[1],
            test_sqrt_PEHE + test_ci[1],
            color="b",
            alpha=0.2,
        )
        axs[1].legend()
    except BaseException as e:
        print("failed to plot(maybe rerun with --plots):", e)
