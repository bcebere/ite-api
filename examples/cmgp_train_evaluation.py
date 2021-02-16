#!/usr/bin/env python
# coding: utf-8

# # CMGP: Train and Evaluation

# ## Setup
#
# First, make sure that all the depends are installed in the current environment.
# ```
# pip install -r requirements.txt
# pip install .
# ```
#
# stdlib
import sys

# third party
from matplotlib import pyplot as plt

# ite absolute
# Import depends
import ite.algs.causal_multitask_gaussian_processes.model as alg
import ite.datasets as ds
import ite.utils.numpy as utils

# Double check that we are using the correct interpreter.
print(sys.executable)

# ## Load the Dataset
#
# The example is done using the Twins dataset.
#
# __Important__: For CGMP, we have to downsample the dataset to 1000 training items.

train_ratio = 0.8

dataset = ds.load("twins", train_ratio, downsample=1000)
[Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y] = dataset


# ## Load the model
#
# Next, we define the model.
#
#
# The constructor supports the following parameters:
#  - `dim`: The number of features in X.
#  - `dim_outcome`: The number of potential outcomes.
#  - `max_gp_iterations`: Maximum number of GP iterations before stopping the training.

dim = len(Train_X[0])
dim_outcome = Test_Y.shape[1]

model = alg.CMGP(
    dim=dim,
    dim_outcome=dim_outcome,
    max_gp_iterations=500,
)

assert model is not None


# ## Run experiments

for experiment in range(6):
    dataset = ds.load(
        "twins",
        train_ratio,
        downsample=1000,
    )

    metrics = model.train(*dataset)


# ## Plot experiments metrics
metrics.print()

metrics.plot(plt, with_ci=True, thresholds=[0.2, 0.25, 0.3, 0.35])
plt.show()


# ## Predict

hat_y = model.predict(Test_X)
utils.sqrt_PEHE(hat_y.to_numpy(), Test_Y)


# ## Test
# Will can run inferences and get metrics directly
tmetrics = model.test(Test_X, Test_Y)

tmetrics.print()
