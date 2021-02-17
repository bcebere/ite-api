# GANITE(PyTorch version)

This module contains the adapted GANITE implementation in PyTorch.
The implementation is adapted from [the original version](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/ganite/) and the [Tensorflow version](https://github.com/bcebere/ite-api/tree/main/src/ite/algs/ganite).

## Breakdown
The main entrypoint for the module is the `Ganite` class.
It contains the `CounterfactualGenerator`, `CounterfactualDiscriminator` and the `InferenceNets`.

The constructor accepts the following parameters:

 - `dim`: The number of features in X.
 - `dim_outcome`: The number of potential outcomes.
 - `dim_hidden`: hyperparameter for tuning the size of the hidden layer.
 - `depth`: hyperparameter for the number of hidden layers in the generator and inference blocks.
 - `num_iterations`: hyperparameter for the number of training epochs.
 - `alpha`: hyperparameter used for the Generator block loss.
 - `beta`: hyperparameter used for the ITE block loss.
 - `num_discr_iterations`: number of iterations executed by the discriminator.
 - `minibatch_size`: the size of the dataset batches.


It exposes the following methods:

 1. `train` : Train and test the framework. Returns a `HistoricMetrics` object with the evolution of the metrics on the test set during training.
 2. `predict`: Returns the output of the trained `InferenceNets` on a test set.
 3. `test`: Runs the predict methods and returns a `Metrics` object containing different metrics reported to the expected output.

## References

1. Jinsung Yoon, James Jordon, Mihaela van der Schaar, "GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", International Conference on Learning Representations (ICLR), 2018 ([Paper](https://openreview.net/forum?id=ByKWUeWA-)).
2. [Original code](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/ganite/).