# Algorithms for individualized treatment effects estimation

[![Tests](https://github.com/bcebere/ite-api/workflows/Tests/badge.svg?branch=main)](https://github.com/bcebere/ite-api/actions?query=workflow%3ATests)

Estimating Individualized Treatment Effects(ITE) is the task that approximates whether a given treatment influences or determines an outcome([read more here](https://www.vanderschaar-lab.com/individualized-treatment-effect-inference/)).

This library creates an unified API for two algorithms for ITE, **[GANITE](https://openreview.net/pdf?id=ByKWUeWA-)** and **[CMGP](https://arxiv.org/pdf/1704.02801.pdf)**, trained over the **[Twins](https://www.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html)** dataset.

## Algorithms

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

The [notebooks/](https://github.com/bcebere/ite-api/tree/main/notebooks) folder contains several examples and use cases for using the library:


 - [Train and evaluation of GANITE(Tensorflow)](https://github.com/bcebere/ite-api/blob/main/notebooks/ganite_train_evaluation.ipynb) - Example of how to train and test the GANITE model, Tensorflow version.

 - [Train and evaluation of GANITE(PyTorch)](https://github.com/bcebere/ite-api/blob/main/notebooks/ganite_pytorch_train_evaluation.ipynb) - Example of how to train and test the GANITE model, PyTorch version.

 - [Train and evaluation of CMGP](https://github.com/bcebere/ite-api/blob/main/notebooks/cmgp_train_evaluation.ipynb) - Example of how to train and test the CMGP model.

 - [Train and evaluation of the unified API](https://github.com/bcebere/ite-api/blob/main/notebooks/unified_api_train_evaluation.ipynb) - Example of how to train and test all the model using the unified API.

 - [Hyperparameter tuning](https://github.com/bcebere/ite-api/blob/main/notebooks/hyperparam_tuning.ipynb) - Example of how to search for optimal hyperparameters for the GANITE implementations.

The [examples/](https://github.com/bcebere/ite-api/tree/main/examples) folder contains the converted notebooks to CLI scripts.
