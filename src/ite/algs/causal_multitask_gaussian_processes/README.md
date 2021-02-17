# Causal Multi-task Gaussian Processes(CMGP)

This module contains the adapted CMGP implementation.
The implementation is adapted from [the original version](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/causal_multitask_gaussian_processes_ite/).

## Main differences from the original code:
 - Ran static analysis and linters and fixed the errors. Left only the code relevant to the CMGP task.
 - Activated the initialize_hyperparameters call, as the paper mentions.
 - Moved the metrics in the `utils` module.

## Breakdown
The main entrypoint for the module is the `CMGP` class.

The constructor accepts the following parameters:

- `dim`: The number of features in X.

- `dim_outcome`: The number of potential outcomes.

- `max_gp_iterations`: Maximum number of GP iterations before stopping the training.


It exposes the following methods:

 1. `train` : Train and test the framework. Returns a `HistoricMetrics` object with the metrics on the train and tests set during training.
 2. `predict`: Returns the output on a test set.
 3. `test`: Runs the predict methods and returns a `Metrics` object containing different metrics reported to the expected output.

## References
1. Ahmed M. Alaa, Mihaela van der Schaar, "Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes", NeurIPS, 2017 ([Paper](https://arxiv.org/pdf/1704.02801.pdf)).
2. [CMGP Reference implementation](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/causal_multitask_gaussian_processes_ite/).
