# stdlib
from typing import Any

# third party
import pytest

# ite absolute
import ite.algs.ganite_torch.model as alg
import ite.datasets as ds


def test_sanity() -> None:
    assert alg.GaniteTorch(10, 10, 2) is not None


@pytest.mark.parametrize(
    "iterations",
    [
        1000,
    ],
)
@pytest.mark.parametrize(
    "num_discr_iterations",
    [
        10,
    ],
)
@pytest.mark.parametrize(
    "alpha,beta,batch_size,depth,dim_hidden",
    [
        (2, 2, 128, 5, 8),  # Optimal Hyper-parameters of GANITE(Table 7 in the paper)
    ],
)
def test_ganite_torch_short_training(
    plt: Any,
    iterations: int,
    num_discr_iterations: int,
    alpha: float,
    beta: float,
    batch_size: int,
    depth: int,
    dim_hidden: int,
) -> None:
    train_ratio = 0.8

    dataset = ds.load("twins", train_ratio)

    [Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y] = dataset

    dim = len(Train_X[0])
    dim_hidden = dim if dim_hidden == 0 else dim_hidden
    dim_outcome = Test_Y.shape[1]

    model = alg.GaniteTorch(
        dim,
        dim_hidden,
        dim_outcome,
        num_iterations=iterations,
        alpha=alpha,
        beta=beta,
        minibatch_size=batch_size,
        depth=depth,
        num_discr_iterations=num_discr_iterations,
    )
    assert model is not None

    metrics = model.train(*dataset)

    metrics.print()

    try:
        metrics.plot(plt, thresholds=[0.2, 0.25, 0.3, 0.35])
    except BaseException as e:
        print("failed to plot(maybe rerun with --plots):", e)

    predicted = model.predict(Test_X)

    assert predicted.shape == (Test_X.shape[0], 2)

    test_metrics = model.test(Test_X, Test_Y)
    test_metrics.print()

    print("Top 5 worst errors ", Test_X[test_metrics.worst_mistakes()])

    assert 0.2 < test_metrics.sqrt_PEHE() and test_metrics.sqrt_PEHE() < 0.4
