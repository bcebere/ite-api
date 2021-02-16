# stdlib
from typing import Any
from typing import List

# third party
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.space import Integer
from skopt.space import Real
from skopt.utils import use_named_args

# ite absolute
from ite.algs.ganite.model import Ganite
from ite.algs.ganite_torch.model import GaniteTorch
import ite.datasets as ds


def search(algorithm: str, iterations: int = 100) -> List[Any]:
    assert algorithm in ["GANITE", "GANITE_TORCH"]

    # load dataset
    dataset = ds.load("twins", 0.8)
    [Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y] = dataset

    dim = len(Train_X[0])
    dim_outcome = Test_Y.shape[1]

    # define the space of hyperparameters to search
    search_space = list()
    search_space.append(Integer(3, 10, name="num_discr_iterations"))
    search_space.append(Categorical([32, 64, 128, 256], name="minibatch_size"))
    search_space.append(
        Categorical(
            [dim, int(dim / 2), int(dim / 3), int(dim / 4), int(dim / 5)],
            name="dim_hidden",
        )
    )
    search_space.append(Real(1e-6, 10.0, "log-uniform", name="alpha"))
    search_space.append(Real(1e-6, 10.0, "log-uniform", name="beta"))
    search_space.append(Integer(1, 9, name="depth"))

    # define the function used to evaluate a given configuration
    @use_named_args(search_space)
    def evaluate_model(**params: Any) -> float:
        # configure the model with specific hyperparameters
        model_class: Any
        if algorithm == "GANITE":
            model_class = Ganite
        elif algorithm == "GANITE_TORCH":
            model_class = GaniteTorch
        else:
            raise Exception(f"model not supported {model_class}")

        model = model_class(
            dim,
            dim_outcome,
            num_iterations=iterations,
            **params,
        )

        model.train(*dataset)
        test_metrics = model.test(Test_X, Test_Y)
        return test_metrics.sqrt_PEHE()

    # perform optimization
    result = gp_minimize(evaluate_model, search_space)

    # summarizing finding:
    print(f"Best Accuracy: {1.0 - result.fun:.3f}")
    print(f"Best Parameters: {result.x}")

    return result.x
