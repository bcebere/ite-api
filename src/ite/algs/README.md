# Unified API
 This module contains an unified API for the GANITE and CMGP algorithms.

## Algorithms
 - [GANITE(Tensorflow)](https://github.com/bcebere/ite-api/tree/main/src/ite/algs/ganite).
 - [GANITE(PyTorch)](https://github.com/bcebere/ite-api/tree/main/src/ite/algs/ganite_torch).
 - [CMGP](https://github.com/bcebere/ite-api/tree/main/src/ite/algs/causal_multitask_gaussian_processes).

## API
The main entrypoint for the module is the `Model` class.

The constructor accepts the following parameters:

 - `mode`: Mandatory name for the model to use. Can be from `["GANITE", "GANITE_TORCH", "CMGP"]`.
 - **kwargs: Any relevant parameter for the chosen model is directly fowarded to its constructor.


It exposes the following methods:

 1. `train` : Train and test the framework. Returns a `HistoricMetrics` object with the evolution of the metrics on the test set during training.
 2. `predict`: Returns the output on a test set.
 3. `test`: Runs the predict methods and returns a `Metrics` object containing different metrics reported to the expected output.

## Hyperparameter tuning
The module provides support for hyper-parameter searching for the __GANITE__(for both Tensorflow and PyTorch version) algorithm over the [Twins](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/data/twins/) dataset.

The algorithm used for performing hyperparameter optimization is the [__Bayesian Optimization__](https://en.wikipedia.org/wiki/Bayesian_optimization).

__Bayesian Optimization__ provides a principled technique based on [Bayes Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) to direct a search of a global optimization problem that is efficient and effective. It works by building a probabilistic model of the objective function, called the surrogate function, that is then searched efficiently with an acquisition function before candidate samples are chosen to evaluate the real objective function.


For the tuning, we use the [__Scikit-Optimize__](https://scikit-optimize.github.io/stable/) library, which provides a general toolkit for [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) that can be used for hyperparameter tuning.

For __GANITE__, we try to optimize the following hyperparameters using the ranges suggested in [[3] Table 6](https://openreview.net/forum?id=ByKWUeWA-):


| Hyperparameter | Search area | Description |
| --- | --- | --- |
| dim_hidden | {dim, int(dim/2), int(dim/3), int(dim/4), int(dim/5)} | the size of the hidden layers. |
| depth |{1, 3, 5, 7, 9} | the number of hidden layers in the generator and inference blocks. |
| alpha | {0, 0.1, 0.5, 1, 2, 5, 10} | weight for the Generator block loss. |
| beta | {0, 0.1, 0.5, 1, 2, 5, 10} | weight the ITE block loss. |
| num_discr_iterations | [3, 10] | number of iterations executed by the Counterfactual discriminator. |
| minibatch_size | {32, 64, 128, 256} | the size of the dataset batches. |

`hyperparam_tuning.search`:

 - Searches for optimal hyperparameters for a model. The models can be from ["GANITE", "GANITE_TORCH"].
 - Returns an array of optimal parameters for ["num_discr_iterations", "minibatch_size", "dim_hidden", "alpha", "beta", "depth"].

## References
1. [Clairvoyance](https://openreview.net/forum?id=xnC8YwKUE3k).
2. [Scikit-Optimize](https://scikit-optimize.github.io/stable/).
3. Jinsung Yoon, James Jordon, Mihaela van der Schaar, "GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", International Conference on Learning Representations (ICLR), 2018 ([Paper](https://openreview.net/forum?id=ByKWUeWA-)).