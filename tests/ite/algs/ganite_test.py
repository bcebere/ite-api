# stdlib
from typing import Any

# third party
import pytest

# ite absolute
import ite.algs.ganite.model as alg
import ite.datasets as ds


def test_sanity() -> None:
    assert alg.Ganite(10, 2) is not None


@pytest.mark.parametrize(
    "iterations",
    [
        1000,
        10000,
    ],
)
def test_ganite_short_training(plt: Any, iterations: int) -> None:
    train_ratio = 0.8

    [Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y] = ds.load(
        "twins", train_ratio
    )

    dim = len(Train_X[0])
    dim_outcome = Test_Y.shape[1]

    model = alg.Ganite(dim, dim_outcome)
    assert model is not None

    metrics = model.fit(Train_X, Train_T, Train_Y, Test_X, Test_Y, iterations)

    assert "gen_block" in metrics
    assert "D_loss" in metrics["gen_block"]
    assert "G_loss" in metrics["gen_block"]

    assert "ite_block" in metrics
    assert "I_loss" in metrics["ite_block"]
    assert "Loss_PEHE" in metrics["ite_block"]
    assert "Loss_ATE" in metrics["ite_block"]

    fig, axs = plt.subplots(2)
    axs[0].plot(metrics["gen_block"]["D_loss"], label="Cf Discriminator loss")
    axs[0].plot(metrics["gen_block"]["G_loss"], label="Cf Generator loss")
    axs[0].legend()
    axs[1].plot(metrics["ite_block"]["I_loss"], label="ITE loss")
    axs[1].plot(metrics["ite_block"]["Loss_PEHE"], label="Loss_PEHE")
    axs[1].plot(metrics["ite_block"]["Loss_ATE"], label="Loss_ATE")
    axs[1].legend()

    predicted = model.predict(Test_X)

    assert predicted.shape == (Test_X.shape[0], 2)
