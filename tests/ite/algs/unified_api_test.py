# third party
import pytest

# ite absolute
from ite.algs.model import Model
import ite.datasets as ds


def test_unified_api_sanity() -> None:
    assert Model("GANITE", 10, 10, 2) is not None
    assert Model("GANITE_TORCH", 10, 10, 2) is not None
    assert Model("CMGP") is not None

    with pytest.raises(BaseException):
        Model("INVALID_NAME")


@pytest.mark.parametrize("ganite_ver", ["GANITE", "GANITE_TORCH"])
def test_unified_api_ganite(ganite_ver: str) -> None:
    train_ratio = 0.8
    dataset = ds.load("twins", train_ratio)
    [Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y] = dataset

    dim = len(Train_X[0])
    dim_hidden = dim
    dim_outcome = Test_Y.shape[1]

    model = Model(
        ganite_ver,
        dim,
        dim_hidden,
        dim_outcome,
        num_iterations=10,
        alpha=2,
        beta=2,
        minibatch_size=128,
        depth=2,
        num_discr_iterations=4,
    )
    assert model.core.minibatch_size == 128
    assert model.core.alpha == 2
    assert model.core.beta == 2
    assert model.core.depth == 2
    assert model.core.num_iterations == 10
    assert model.core.num_discr_iterations == 4

    metrics = model.train(*dataset)
    metrics.print()

    predicted = model.predict(Test_X)
    assert predicted.shape == (Test_X.shape[0], 2)

    test_metrics = model.test(Test_X, Test_Y)
    test_metrics.print()


def test_unified_api_cmgp() -> None:
    train_ratio = 0.8

    dataset = ds.load(
        "twins",
        train_ratio,
        downsample=1000,
    )
    [Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y] = dataset

    dim = len(Train_X[0])
    dim_outcome = Test_Y.shape[1]

    model = Model(
        "CMGP",
        dim=dim,
        dim_outcome=dim_outcome,
        max_gp_iterations=50,
    )
    assert model is not None

    metrics = model.train(*dataset)
    metrics.print()

    predicted = model.predict(Test_X)
    assert predicted.shape == (Test_X.shape[0], 2)

    test_metrics = model.test(Test_X, Test_Y)
    test_metrics.print()
