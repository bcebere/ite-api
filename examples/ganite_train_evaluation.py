#!/usr/bin/env python
# coding: utf-8

# # GANITE(Tensorflow): Train and Evaluation
# ## Setup
#
# First, make sure that all the depends are installed in the current environment.
# ```
# pip install -r requirements.txt
# pip install .
# ```
#
# Next, we import all the dependencies necessary for the task.

# In[15]:


# stdlib
import os
import sys

# third party
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow.compat.v1 as tf

# ite absolute
# Depends
import ite.algs.ganite.model as alg
import ite.datasets as ds
import ite.utils.tensorflow as utils

# Double check that we are using the correct interpreter.
print(sys.executable)

# Disable TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ## Load the Dataset
#
# The example is done using the Twins dataset.
#
# Next, we load the dataset, process the data, and sample a training set and a test set.

train_ratio = 0.8

dataloader = ds.load("twins", train_ratio)
[Train_X, Train_T, Train_Y, Opt_Train_Y, Test_X, Test_Y] = dataloader


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

model = alg.Ganite(
    dim,
    dim_outcome,
    dim_hidden=7,
    num_iterations=10000,
    alpha=5,
    beta=0.1,
    minibatch_size=32,
    num_discr_iterations=9,
    depth=2,
)

assert model is not None


# ## Train the model
metrics = model.train(*dataloader)


# ## Plot train metrics
metrics.plot(plt, thresholds=[0.2, 0.25, 0.3, 0.35])

metrics.print()
plt.show()


# ## Predict
#
# You can use run inferences on the model and evaluate the output.
sess = tf.InteractiveSession()

hat_y = model.predict(Test_X)

utils.sqrt_PEHE(hat_y, Test_Y).eval()


# ## Test
# Will can run inferences and get metrics directly


test_metrics = model.test(Test_X, Test_Y)

test_metrics.print()
