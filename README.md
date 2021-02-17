# Algorithms for individualized treatment effects estimation

[![Tests](https://github.com/bcebere/ite-api/workflows/Tests/badge.svg?branch=main)](https://github.com/bcebere/ite-api/actions?query=workflow%3ATests)

Estimating Individualized Treatment Effects(ITE) is the task that approximates whether a given treatment influences or determines an outcome([read more here](https://www.vanderschaar-lab.com/individualized-treatment-effect-inference/)).

This library creates a unified API for two algorithms for ITE, **[GANITE](https://openreview.net/pdf?id=ByKWUeWA-)** and **[CMGP](https://arxiv.org/pdf/1704.02801.pdf)**, trained over the **[Twins](https://www.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html)** dataset.

## Installation

```
pip install -r requirements.txt
pip install .
```

## Testing [![Tests](https://github.com/bcebere/ite-api/workflows/Tests/badge.svg?branch=main)](https://github.com/bcebere/ite-api/actions?query=workflow%3ATests)


The library is covered by [tests](https://github.com/bcebere/ite-api/tree/main/tests/ite) implemented using **pytest**.

For the algorithms, we simulate a short training for sanity checks, and you can check the training metrics in the `plots/` folder.
```
pytest -vvs --plots -m "not slow"
```

The library is tested using Python 3.8 on **Windows**, **Linux**, and **MacOS**([see on Github Actions](https://github.com/bcebere/ite-api/actions)).

## Usage

The [notebooks/](https://github.com/bcebere/ite-api/tree/main/notebooks) folder contains several examples and use-cases for the library:


 - [Training and evaluation of GANITE(Tensorflow)](https://github.com/bcebere/ite-api/blob/main/notebooks/ganite_train_evaluation.ipynb) - Example of how to train and test the GANITE model, Tensorflow version.

 - [Training and evaluation of GANITE(PyTorch)](https://github.com/bcebere/ite-api/blob/main/notebooks/ganite_pytorch_train_evaluation.ipynb) - Example of how to train and test the GANITE model, PyTorch version.

 - [Training and evaluation of CMGP](https://github.com/bcebere/ite-api/blob/main/notebooks/cmgp_train_evaluation.ipynb) - Example of how to train and test the CMGP model.

 - [Training and evaluation of the unified API](https://github.com/bcebere/ite-api/blob/main/notebooks/unified_api_train_evaluation.ipynb) - Example of how to train and test all the model using the unified API.

 - [Hyperparameter tuning](https://github.com/bcebere/ite-api/blob/main/notebooks/hyperparam_tuning.ipynb) - Example of how to search for optimal hyperparameters for the GANITE implementations.

The [examples/](https://github.com/bcebere/ite-api/tree/main/examples) folder contains the converted notebooks to CLI scripts.

## Code breakdown
Algorithms:

- [GANITE(Tensorflow) details](https://github.com/bcebere/ite-api/tree/main/src/ite/algs/ganite).
- [GANITE(PyTorch) details](https://github.com/bcebere/ite-api/tree/main/src/ite/algs/ganite_torch).
- [CMGP details](https://github.com/bcebere/ite-api/tree/main/src/ite/algs/causal_multitask_gaussian_processes).

Library:

- [Dataset loading and pre-processing](https://github.com/bcebere/ite-api/tree/main/src/ite/datasets).
- [Metrics and helpers](https://github.com/bcebere/ite-api/tree/main/src/ite/utils).
- [Unified API](https://github.com/bcebere/ite-api/tree/main/src/ite/algs).
- [Hyperparameter tuning](https://github.com/bcebere/ite-api/tree/main/src/ite/algs).

## References
1. Jinsung Yoon, James Jordon, Mihaela van der Schaar, "GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", International Conference on Learning Representations (ICLR), 2018 ([Paper](https://openreview.net/forum?id=ByKWUeWA-)).
2. Ahmed M. Alaa, Mihaela van der Schaar, "Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes", NeurIPS, 2017 ([Paper](https://arxiv.org/pdf/1704.02801.pdf)).
3. [Original GANITE code](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/ganite/).
4. [CMGP Reference implementation](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/causal_multitask_gaussian_processes_ite/).
5. [Twins dataset](https://www.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html).
6. [Clairvoyance](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/02edab3b2b6d635470fa80184bbfd03b8bf8082d/app/clairvoyance/).
