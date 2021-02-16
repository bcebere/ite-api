#!/usr/bin/env python
# coding: utf-8

# # Unified API: Train and evaluation

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
from ite.algs.model import Model  # the unified API
import ite.datasets as ds

# Double check that we are using the correct interpreter.
print(sys.executable)

# ## Load the Dataset
#
# The example is done using the Twins dataset.
#
# Next, we load the dataset, process the data, and sample a training set and a test set.
#
# For CGMP, we have to downsample to 1000 training items. For the rest, we load without downsampling.

train_ratio = 0.8

full_dataloader = ds.load("twins", train_ratio)
cmgp_dataloader = ds.load("twins", train_ratio, downsample=1000)


# ## Load and train GANITE(Tensorflow version)
#
# The constructor requires the name of the chosen algorithm for the first parameter - `GANITE`.
#
# The constructor supports the same parameters as the "native" version:
#  - `dim`: The number of features in X.
#  - `dim_outcome`: The number of potential outcomes.
#  - `dim_hidden`: hyperparameter for tuning the size of the hidden layer.
#  - `depth`: hyperparameter for the number of hidden layers in the generator and inference blocks.
#  - `num_iterations`: hyperparameter for the number of training epochs.
#  - `alpha`: hyperparameter used for the Generator block loss.
#  - `beta`: hyperparameter used for the ITE block loss.
#  - `num_discr_iterations`: number of iterations executed by the discriminator.
#
#  The hyperparameters used in this experiment are from Table 7 in the paper.

dim = len(full_dataloader[0][0])
dim_outcome = full_dataloader[-1].shape[1]

ganite_model = Model(
    "GANITE",
    dim,
    dim_outcome,
    dim_hidden=8,
    num_iterations=10000,
    alpha=2,
    beta=2,
    minibatch_size=128,
    num_discr_iterations=3,
    depth=5,
)

ganite_tf_metrics = ganite_model.train(*full_dataloader)
ganite_tf_metrics.print()

ganite_tf_metrics.plot(plt, thresholds=[0.2, 0.25, 0.3, 0.35])
plt.show()


# ## Load and train GANITE(PyTorch version)
#
# The constructor requires the name of the chosen algorithm for the first parameter - `GANITE_TORCH`.
#
# The constructor supports the same parameters as the "native" version:
#  - `dim`: The number of features in X.
#  - `dim_outcome`: The number of potential outcomes.
#  - `dim_hidden`: hyperparameter for tuning the size of the hidden layer.
#  - `depth`: hyperparameter for the number of hidden layers in the generator and inference blocks.
#  - `num_iterations`: hyperparameter for the number of training epochs.
#  - `alpha`: hyperparameter used for the Generator block loss.
#  - `beta`: hyperparameter used for the ITE block loss.
#  - `num_discr_iterations`: number of iterations executed by the discriminator.
#

ganite_torch_model = Model(
    "GANITE_TORCH",
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

ganite_torch_metrics = ganite_torch_model.train(*full_dataloader)
ganite_torch_metrics.print()

ganite_torch_metrics.plot(plt, thresholds=[0.2, 0.25, 0.3, 0.35])
plt.show()


# ## Load and train CMGP
#
# The constructor requires the name of the chosen algorithm for the first parameter - `CMGP`.
#
# The constructor supports the following parameters:
#  - `dim`: The number of features in X.
#  - `dim_outcome`: The number of potential outcomes.
#  - `max_gp_iterations`: Maximum number of GP iterations before stopping the training.
#

# In[31]:


cmgp_model = Model(
    "CMGP",
    dim=dim,
    dim_outcome=dim_outcome,
    max_gp_iterations=1000,  # (optional) Maximum number of interations for the Gaussian Process
)

for experiment in range(5):
    cmgp_dataloader = ds.load("twins", train_ratio, downsample=1000)
    cmgp_metrics = cmgp_model.train(*cmgp_dataloader)

cmgp_metrics.print()

cmgp_metrics.plot(plt, thresholds=[0.2, 0.25, 0.3, 0.35])
plt.show()


# ## Evaluate the models on the test set
#


test_results = [
    [
        "GANITE",
        "{:0.3f} +/- {:0.3f}".format(
            *ganite_tf_metrics.mean_confidence_interval(
                "sqrt_PEHE", "ITE Block out-sample metrics"
            )
        ),
        "{:0.3f} +/- {:0.3f}".format(
            *ganite_tf_metrics.mean_confidence_interval(
                "ATE", "ITE Block out-sample metrics"
            )
        ),
    ],
    [
        "GANITE_TORCH",
        "{:0.3f} +/- {:0.3f}".format(
            *ganite_torch_metrics.mean_confidence_interval(
                "sqrt_PEHE", "ITE Block out-sample metrics"
            )
        ),
        "{:0.3f} +/- {:0.3f}".format(
            *ganite_torch_metrics.mean_confidence_interval(
                "ATE", "ITE Block out-sample metrics"
            )
        ),
    ],
    [
        "CMGP",
        "{:0.3f} +/- {:0.3f}".format(
            *cmgp_metrics.mean_confidence_interval("sqrt_PEHE", "out-sample metrics")
        ),
        "{:0.3f} +/- {:0.3f}".format(
            *cmgp_metrics.mean_confidence_interval("ATE", "out-sample metrics")
        ),
    ],
]

headers = ["Model", "sqrt_PEHE", "ATE"]

for result in test_results:
    print(" Model {result[0]}: ")
    print("   - sqrt_PEHE {result[1]}: ")
    print("   - ATE {result[2]}: ")
