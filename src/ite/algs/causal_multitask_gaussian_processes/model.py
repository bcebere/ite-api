# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# stdlib
from typing import Any
from typing import Tuple

# third party
import GPy
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

log_2_pi = np.log(2 * np.pi)


class CMGP:
    """
    An implementation of various Gaussian models for Causal inference building on GPy.

    """

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # This method implements the class constructor, automatically
    # invoked for every class instance
    # ----------------------------------------------------------------
    def __init__(self, mode: str = "CMGP", **kwargs: Any) -> None:
        """
        Class constructor.
        Initialize a GP object for causal inference.

        :mod: 'Multitask'
        :dim: the dimension of the input. Default is 1
        :kern: ['Matern'] or ['RBF'], Default is the Radial Basis Kernel
        :mkern: For multitask models, can select from IMC and LMC models, default is IMC
        """
        # %%%%%%%%%%%%%%%%%
        # **Set defaults**
        # %%%%%%%%%%%%%%%%%
        self.kern_list = ["RBF", "Matern"]
        self.mkern_list = ["ICM", "LCM"]
        self.mod = "Multitask"
        self.dim = 1
        self.kern = self.kern_list[0]
        self.mkern = self.mkern_list[0]
        self.mode = mode
        self.Bayesian = True
        self.Confidence = True
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # ** Read input arguments
        # ~~~~~~~~~~~~~~~~~~~~~~~
        if kwargs.__contains__("mod"):
            self.mod = kwargs["mod"]
        if kwargs.__contains__("dim"):
            self.dim = kwargs["dim"]
        if kwargs.__contains__("kern"):
            self.kern = kwargs["kern"]
        if kwargs.__contains__("mkern"):
            self.mkern = kwargs["mkern"]
        # ++++++++++++++++++++++++++++++++++++++++++
        # ** catch exceptions ** handle wrong inputs
        # ++++++++++++++++++++++++++++++++++++++++++
        try:
            if (self.dim < 1) or (type(self.dim) != int):
                raise ValueError(
                    "Invalid value for the input dimension! Input dimension has to be a positive integer."
                )
            if (self.kern not in self.kern_list) or (self.mkern not in self.mkern_list):
                raise ValueError("Invalid input!")
            if (kwargs.__contains__("mkern")) and (self.mod != "Multitask"):
                raise ValueError(
                    "Invalid input! Multitask kernels are valid only for the Multitask mode"
                )

        except ValueError:
            if self.kern not in self.kern_list:
                raise ValueError(
                    "Invalid input: The provided kernel is undefined for class GaussianProcess_Model."
                )
            elif self.mkern not in self.mkern_list:
                raise ValueError(
                    "Invalid input: The provided Multitask kernel is undefined for class GaussianProcess_Model."
                )
            else:
                raise ValueError("Invalid input for GaussianProcess_Model!")
        else:
            # *************************************************************************
            # Initialize the kernels and likelihoods depending on the specified model
            # *************************************************************************
            if self.kern == self.kern_list[0]:
                base_kernel = GPy.kern.RBF(input_dim=self.dim, ARD=True)
                self.ker = GPy.util.multioutput.ICM(
                    self.dim,
                    2,
                    base_kernel,
                    W_rank=1,
                    W=None,
                    kappa=None,
                    name="ICM",
                )
            else:
                self.ker = GPy.kern.Matern32(input_dim=self.dim)

            self.lik = GPy.likelihoods.Gaussian()

    # -----------------------------------------------------------------------------------------------------------
    # This method optimizes the model hyperparameters using the factual samples for the treated and control arms
    # ------------------------------------------------------------------------------------------------------------
    # ** Note ** all inputs to this method are positional arguments
    # ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, W: pd.DataFrame) -> None:
        """
        Optimizes the model hyperparameters using the factual samples for the treated and control arms.
        X has to be an N x dim matrix.

        :X: The input covariates
        :Y: The corresponding outcomes
        :W: The treatment assignments
        """
        # -----------------------------------------------------------------
        # Inputs: X (the features), Y (outcomes), W (treatment assignments)
        # X has to be an N x dim matrix.
        # -----------------------------------------------------------------
        # Situate the data in a pandas data frame
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Dataset = pd.DataFrame(X)
        Dataset["Y"] = Y
        Dataset["W"] = W

        self.X_train = np.array(X)

        if self.dim > 1:
            Feature_names = list(range(self.dim))
        else:
            Feature_names = [0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Catch exceptions: handle errors in the input sizes, size mismatches, or undefined
        # treatment assignments
        # ----------------------
        # try:
        #    if (Xshape[1] != self.dim) or (Yshape[1] != 1) or (Xshape[0] != Yshape[0]) or (len(W_comp)>0):
        #        raise ValueError('Invalid Inputs!')
        # except ValueError:
        #    if (Xshape[1] != self.dim):
        #        raise ValueError('Invalid input: Dimension of input covariates do not match the model dimensions')
        #    elif (Yshape[1] != 1):
        #        raise ValueError('Invalid input: Outcomes must be formatted in a 1D vector.')
        #    elif (Xshape[0] != Yshape[0]):
        #        raise ValueError('Invalid input: Outcomes and covariates do not have the same number of samples.')
        #    elif (len(W_comp)>0):
        #        raise ValueError('Invalid input: Treatment assignment vector has non-binary values.')
        # else:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Dataset0 = Dataset[Dataset["W"] == 0].copy()
        Dataset1 = Dataset[Dataset["W"] == 1].copy()
        # Extract data for the first learning task (control population)
        # `````````````````````````````````````````````````````````````````
        X0 = np.reshape(Dataset0[Feature_names].copy(), (len(Dataset0), self.dim))
        y0 = np.reshape(np.array(Dataset0["Y"].copy()), (len(Dataset0), 1))
        # Extract data for the second learning task (treated population)
        # `````````````````````````````````````````````````````````````````
        X1 = np.reshape(Dataset1[Feature_names].copy(), (len(Dataset1), self.dim))
        y1 = np.reshape(np.array(Dataset1["Y"].copy()), (len(Dataset1), 1))
        # Create an instance of a GPy Coregionalization model
        # `````````````````````````````````````````````````````````````````
        K0 = GPy.kern.Matern32(self.dim, ARD=True)  # GPy.kern.RBF(self.dim, ARD=True)
        K1 = GPy.kern.Matern32(
            self.dim
        )  # , ARD=True) #GPy.kern.RBF(self.dim, ARD=True)

        K0 = GPy.kern.RBF(self.dim, ARD=True)
        K1 = GPy.kern.RBF(self.dim, ARD=True)

        kernel_dict = {
            "CMGP": GPy.util.multioutput.LCM(
                input_dim=self.dim, num_outputs=2, kernels_list=[K0, K1]
            ),
            "NSGP": GPy.util.multioutput.ICM(
                input_dim=self.dim, num_outputs=2, kernel=K0
            ),
        }

        self.model = GPy.models.GPCoregionalizedRegression(
            X_list=[X0, X1], Y_list=[y0, y1], kernel=kernel_dict[self.mode]
        )

        # self.initialize_hyperparameters(X, Y, W)

        try:

            self.model.optimize("bfgs", max_iters=500)

        except np.linalg.LinAlgError as err:
            print("Covariance matrix not invertible. ", err)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -----------------------------------------------------------------------------------------------------------
    # This method Infers the treatment effect for a certain set of input covariates
    # ------------------------------------------------------------------------------------------------------------
    # ** Note ** all inputs to this method are positional arguments
    # This method returns the predicted ITE and posterior variance
    # but does not store them in self
    # ---------------------------------------------------------------
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Infers the treatment effect for a certain set of input covariates.
        Returns the predicted ITE and posterior variance.

        :X: The input covariates at which the outcomes need to be predicted
        """
        if self.dim == 1:
            X_ = X[:, None]
            X_0 = np.hstack([X_, np.reshape(np.array([0] * len(X)), (len(X), 1))])
            X_1 = np.hstack([X_, np.reshape(np.array([1] * len(X)), (len(X), 1))])
            noise_dict_0 = {"output_index": X_0[:, 1:].astype(int)}
            noise_dict_1 = {"output_index": X_1[:, 1:].astype(int)}
            Y_est_0 = self.model.predict(X_0, Y_metadata=noise_dict_0)[0]
            Y_est_1 = self.model.predict(X_1, Y_metadata=noise_dict_1)[0]

        else:

            X_0 = np.array(
                np.hstack([X, np.zeros_like(X[:, 1].reshape((len(X[:, 1]), 1)))])
            )
            X_1 = np.array(
                np.hstack([X, np.ones_like(X[:, 1].reshape((len(X[:, 1]), 1)))])
            )
            X_0_shape = X_0.shape
            X_1_shape = X_1.shape
            noise_dict_0 = {
                "output_index": X_0[:, X_0_shape[1] - 1]
                .reshape((X_0_shape[0], 1))
                .astype(int)
            }
            noise_dict_1 = {
                "output_index": X_1[:, X_1_shape[1] - 1]
                .reshape((X_1_shape[0], 1))
                .astype(int)
            }
            Y_est_0 = np.array(
                list(self.model.predict(X_0, Y_metadata=noise_dict_0)[0])
            )
            Y_est_1 = np.array(
                list(self.model.predict(X_1, Y_metadata=noise_dict_1)[0])
            )

        TE_est = Y_est_1 - Y_est_0

        return TE_est, Y_est_0, Y_est_1

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    # This method initializes the model's hyper-parameters before passing to the optimizer
    # Now working only for the multi-task model
    # ------------------------------------------------------------------------------------------------------------
    def initialize_hyperparameters(
        self, X: pd.DataFrame, Y: pd.DataFrame, W: pd.DataFrame
    ) -> None:
        """
        Initializes the multi-tasking model's hyper-parameters before passing to the optimizer

        :X: The input covariates
        :Y: The corresponding outcomes
        :W: The treatment assignments
        """
        # -----------------------------------------------------------------------------------
        # Output Parameters:
        # -----------------
        # :Ls0, Ls1: length scale vectors for treated and control, dimensions match self.dim
        # :s0, s1: noise variances for the two kernels
        # :a0, a1: diagonal elements of correlation matrix 0
        # :b0, b1: off-diagonal elements of correlation matrix 1
        # -----------------------------------------------------------------------------------
        Dataset = pd.DataFrame(X)
        Dataset["Y"] = Y
        Dataset["W"] = W

        if self.dim > 1:
            Feature_names = list(range(self.dim))
        else:
            Feature_names = [0]

        Dataset0 = Dataset[Dataset["W"] == 0]
        Dataset1 = Dataset[Dataset["W"] == 1]
        neigh0 = KNeighborsRegressor(n_neighbors=10)
        neigh1 = KNeighborsRegressor(n_neighbors=10)
        neigh0.fit(Dataset0[Feature_names], Dataset0["Y"])
        neigh1.fit(Dataset1[Feature_names], Dataset1["Y"])
        Dataset["Yk0"] = neigh0.predict(Dataset[Feature_names])
        Dataset["Yk1"] = neigh1.predict(Dataset[Feature_names])
        Dataset0["Yk0"] = Dataset.loc[Dataset["W"] == 0, "Yk0"]
        Dataset0["Yk1"] = Dataset.loc[Dataset["W"] == 0, "Yk1"]
        Dataset1["Yk0"] = Dataset.loc[Dataset["W"] == 1, "Yk0"]
        Dataset1["Yk1"] = Dataset.loc[Dataset["W"] == 1, "Yk1"]
        # `````````````````````````````````````````````````````
        a0 = np.sqrt(np.mean((Dataset0["Y"] - np.mean(Dataset0["Y"])) ** 2))
        a1 = np.sqrt(np.mean((Dataset1["Y"] - np.mean(Dataset1["Y"])) ** 2))
        b0 = np.mean(
            (Dataset["Yk0"] - np.mean(Dataset["Yk0"]))
            * (Dataset["Yk1"] - np.mean(Dataset["Yk1"]))
        ) / (a0 * a1)
        b1 = b0
        s0 = np.sqrt(np.mean((Dataset0["Y"] - Dataset0["Yk0"]) ** 2)) / a0
        s1 = np.sqrt(np.mean((Dataset1["Y"] - Dataset1["Yk1"]) ** 2)) / a1
        # `````````````````````````````````````````````````````
        self.model.sum.ICM0.rbf.lengthscale = 10 * np.ones(self.dim)
        self.model.sum.ICM1.rbf.lengthscale = 10 * np.ones(self.dim)

        self.model.sum.ICM0.rbf.variance = 1
        self.model.sum.ICM1.rbf.variance = 1
        self.model.sum.ICM0.B.W[0] = b0
        self.model.sum.ICM0.B.W[1] = b0

        self.model.sum.ICM1.B.W[0] = b1
        self.model.sum.ICM1.B.W[1] = b1

        self.model.sum.ICM0.B.kappa[0] = a0 ** 2
        self.model.sum.ICM0.B.kappa[1] = 1e-4
        self.model.sum.ICM1.B.kappa[0] = 1e-4
        self.model.sum.ICM1.B.kappa[1] = a1 ** 2

        self.model.mixed_noise.Gaussian_noise_0.variance = s0 ** 2
        self.model.mixed_noise.Gaussian_noise_1.variance = s1 ** 2
