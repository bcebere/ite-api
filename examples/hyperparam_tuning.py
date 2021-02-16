#!/usr/bin/env python
# coding: utf-8

# # Hyper-parameter tuning for GANITE

# ## Setup
#
# First, make sure that all the depends are installed in the current environment.
# ```
# pip install -r requirements.txt
# pip install .
# ```
#
# Next, we import all the dependencies necessary for the task.

# ite absolute
import ite.algs.hyperparam_tuning as tuning

param_search_names = [
    "num_discr_iterations",
    "minibatch_size",
    "dim_hidden",
    "alpha",
    "beta",
    "depth",
]


# ### GANITE(Tensorflow)
tf_best_params = tuning.search("GANITE", iterations=5000)


# ### Hyper-parameter tuning for GANITE(Tensorflow)
print("Hyper-parameter tuning for GANITE(Tensorflow)")
for idx, val in enumerate(tf_best_params):
    print(" {param_search_names[idx]} = {val}")

# ### GANITE (PyTorch)
torch_best_params = tuning.search("GANITE_TORCH")

# ### Hyper-parameter tuning results for GANITE(PyTorch)
print("Hyper-parameter tuning for GANITE(PyTorch)")
for idx, val in enumerate(torch_best_params):
    print(" {param_search_names[idx]} = {val}")
