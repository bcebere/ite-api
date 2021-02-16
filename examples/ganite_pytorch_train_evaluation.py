#!/usr/bin/env python
# coding: utf-8

# ## GANITE(PyTorch): Train and Evaluation

# ## Setup
#
#
# First, make sure that all the depends are installed in the current environment.
# ```
# pip install -r requirements.txt
# pip install .
# ```
#
# Next, we import all the dependencies necessary for the task.

# stdlib
import sys

# third party
from matplotlib import pyplot as plt
import pandas as pd

# ite absolute
# Import depends
import ite.algs.ganite_torch.model as alg
import ite.datasets as ds
import ite.utils.numpy as utils

# Double check that we are using the correct interpreter.
print(sys.executable)

# ## Load the Dataset
#
# Next, we load the Twins dataset, process the data, and sample a training set and a test set.
#


train_ratio = 0.8

dataset = ds.load("twins", train_ratio)
[Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y] = dataset

pd.DataFrame(data=Train_X[:5])


# ## Load the model
#
# Next, we define the model.
#
#
# The constructor supports the following parameters:
#  - `dim`: The number of features in X.
#  - `dim_outcome`: The number of potential outcomes.
#  - `dim_hidden`: hyperparameter for tuning the size of the hidden layer.
#  - `depth`: hyperparameter for the number of hidden layers in the generator and inference blocks.
#  - `num_iterations`: hyperparameter for the number of training epochs.
#  - `alpha`: hyperparameter used for the Generator block loss.
#  - `beta`: hyperparameter used for the ITE block loss.
#  - `num_discr_iterations`: number of iterations executed by the discriminator.
#  - `minibatch_size`: the size of the dataset batches.
#

dim = len(Train_X[0])
dim_outcome = Test_Y.shape[1]

model = alg.GaniteTorch(
    dim,
    dim_outcome,
    dim_hidden=30,
    num_iterations=3000,
    alpha=1,
    beta=10,
    minibatch_size=256,
    num_discr_iterations=6,
    depth=5,
)

assert model is not None


# ### Train

metrics = model.train(*dataset)


# ### Plot train metrics

metrics.print()

metrics.plot(plt, thresholds=[0.2, 0.25, 0.3, 0.35])
plt.show()


# ### Predict

hat_y = model.predict(Test_X)

utils.sqrt_PEHE(hat_y.to_numpy(), Test_Y)


# ### Test
# Will can run inferences and get metrics directly

test_metrics = model.test(Test_X, Test_Y)

test_metrics.print()
