# Algorithms for individualized treatment effects estimation

![Tests](https://github.com/bcebere/ite-api/workflows/Tests/badge.svg?branch=main)

Estimating Individualized Treatment Effects(ITE) is the task that approximates whether a given treatment influences or determines an outcome([read more](https://www.vanderschaar-lab.com/individualized-treatment-effect-inference/)).

This library creates an unified API for two algorithms for ITE, **GANITE** and **CMGP**, trained over the **Twins** dataset.

## Algorithms

## Installation

```
pip install -r requirements.txt
pip install .
```

## Testing ![Tests](https://github.com/bcebere/ite-api/workflows/Tests/badge.svg?branch=main)
The library is covered by tests implemented using **pytest**.
For the algorithms, we simulate a short training for sanity checks, and you can check the training metrics in the `plots/` folder.
```
pytest -vvs --plots -m "not slow"

```

The library is continuously tested on **Windows**, **Linux**, and **MacOS** using [Github Actions](https://github.com/bcebere/ite-api/actions).

## Usage

The notebooks/ folder contain several examples and use cases for using the library:


 - [Train and evaluation of GANITE(Tensorflow)](https://github.com/bcebere/ite-api/blob/main/notebooks/ganite_train_evaluation.ipynb) - Example on how to train and test the GANITE model, Tensorflow version.

 - [Train and evaluation of GANITE(PyTorch)](https://github.com/bcebere/ite-api/blob/main/notebooks/ganite_pytorch_train_evaluation.ipynb) - Example on how to train and test the GANITE model, PyTorch version.

 - [Train and evaluation of CMGP](https://github.com/bcebere/ite-api/tree/main/notebooks) - Example on how to train and test the CMGP model.

 - [Train and evaluation of the unified API](https://github.com/bcebere/ite-api/blob/main/notebooks/unified_api_train_evaluation.ipynb) - Example on how to train and test all the model using the unified API.

 - [Hyperparameter tuning](https://github.com/bcebere/ite-api/blob/main/notebooks/hyperparam_tuning.ipynb) - Example on how to search for optimal hyperparameters for the GANITE implementations.