# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# third party
import GPy
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# ite absolute
from ite.utils.metrics import HistoricMetrics
from ite.utils.metrics import Metrics


class CMGP:
    """
    An implementation of various Gaussian models for Causal inference building on GPy.
    """

    def __init__(
        self,
        dim: int = 1,
        dim_outcome: int = 2,
        mod: str = "Multitask",
        mode: str = "CMGP",
        max_gp_iterations: int = 1000,
    ) -> None:
        """
        Class constructor.
        Initialize a GP object for causal inference.

        :mod: 'Multitask'
        :dim: the dimension of the input. Default is 1
        :kern: ['Matern'] or ['RBF'], Default is the Radial Basis Kernel
        :mkern: For multitask models, can select from IMC and LMC models, default is IMC
        """
        # Setup
        self.dim = dim
        self.dim_outcome = dim_outcome
        self.mod = mod
        self.mode = mode
        self.max_gp_iterations = max_gp_iterations

        if (self.dim < 1) or (type(self.dim) != int):
            raise ValueError(
                "Invalid value for the input dimension! Input dimension has to be a positive integer."
            )

        self.train_perf_metrics = HistoricMetrics()

    def train(
        self,
        Train_X: pd.DataFrame,
        Train_T: pd.DataFrame,
        Train_Y: pd.DataFrame,
        Opt_Train_Y: pd.DataFrame,
        Test_X: pd.DataFrame,
        Test_Y: pd.DataFrame,
    ) -> HistoricMetrics:
        """
        Optimizes the model hyperparameters using the factual samples for the treated and control arms.
        Train_X has to be an N x dim matrix.

        :Train_X: The input covariates
        :Train_T: The treatment assignments
        :Train_Y: The corresponding outcomes
        """
        # Inputs: Train_X (the features), Train_T (treatment assignments), Train_Y (outcomes)
        # Train_X has to be an N x dim matrix.
        Dataset = pd.DataFrame(Train_X)
        Dataset["Y"] = Train_Y
        Dataset["T"] = Train_T

        if self.dim > 1:
            Feature_names = list(range(self.dim))
        else:
            Feature_names = [0]

        Dataset0 = Dataset[Dataset["T"] == 0].copy()
        Dataset1 = Dataset[Dataset["T"] == 1].copy()

        # Extract data for the first learning task (control population)
        X0 = np.reshape(Dataset0[Feature_names].copy(), (len(Dataset0), self.dim))
        y0 = np.reshape(np.array(Dataset0["Y"].copy()), (len(Dataset0), 1))

        # Extract data for the second learning task (treated population)
        X1 = np.reshape(Dataset1[Feature_names].copy(), (len(Dataset1), self.dim))
        y1 = np.reshape(np.array(Dataset1["Y"].copy()), (len(Dataset1), 1))

        # Create an instance of a GPy Coregionalization model
        K0 = GPy.kern.RBF(self.dim, ARD=True)
        K1 = GPy.kern.RBF(self.dim, ARD=True)

        kernel_dict = {
            "CMGP": GPy.util.multioutput.LCM(
                input_dim=self.dim, num_outputs=self.dim_outcome, kernels_list=[K0, K1]
            ),
            "NSGP": GPy.util.multioutput.ICM(
                input_dim=self.dim, num_outputs=self.dim_outcome, kernel=K0
            ),
        }

        self.model = GPy.models.GPCoregionalizedRegression(
            X_list=[X0, X1], Y_list=[y0, y1], kernel=kernel_dict[self.mode]
        )

        self.initialize_hyperparameters(Train_X, Train_T, Train_Y)

        try:
            self.model.optimize("bfgs", max_iters=self.max_gp_iterations)
        except np.linalg.LinAlgError as err:
            print("Covariance matrix not invertible. ", err)
            raise err

        metrics = self.test(Train_X, Opt_Train_Y)
        self.train_perf_metrics.add(
            "sqrt_PEHE", metrics.sqrt_PEHE(), "in-sample metrics"
        )
        self.train_perf_metrics.add("ATE", metrics.ATE(), "in-sample metrics")
        self.train_perf_metrics.add("MSE", metrics.MSE(), "in-sample metrics")

        metrics = self.test(Test_X, Test_Y)
        self.train_perf_metrics.add(
            "sqrt_PEHE", metrics.sqrt_PEHE(), "out-sample metrics"
        )
        self.train_perf_metrics.add("ATE", metrics.ATE(), "out-sample metrics")
        self.train_perf_metrics.add("MSE", metrics.ATE(), "out-sample metrics")

        return self.train_perf_metrics

    def train_metrics(self) -> HistoricMetrics:
        return self.train_perf_metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
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

        result = pd.DataFrame(
            0, index=np.arange(len(Y_est_0)), columns=["y_hat_0", "y_hat_1"]
        )
        result["y_hat_0"] = Y_est_0
        result["y_hat_1"] = Y_est_1

        return result

    def test(self, Test_X: pd.DataFrame, Test_Y: pd.DataFrame) -> Metrics:
        """
        Infers the treatment effect for a certain set of input covariates and generate metrics, compared to a reference.
        """
        Hat_Y = self.predict(Test_X).to_numpy()
        return Metrics(Hat_Y, Test_Y)

    def initialize_hyperparameters(
        self, X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame
    ) -> None:
        """
        Initializes the multi-tasking model's hyper-parameters before passing to the optimizer

        :X: The input covariates
        :T: The treatment assignments
        :Y: The corresponding outcomes
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
        Dataset["T"] = T

        if self.dim > 1:
            Feature_names = list(range(self.dim))
        else:
            Feature_names = [0]

        Dataset0 = Dataset[Dataset["T"] == 0].copy()
        Dataset1 = Dataset[Dataset["T"] == 1].copy()
        neigh0 = KNeighborsRegressor(n_neighbors=10)
        neigh1 = KNeighborsRegressor(n_neighbors=10)
        neigh0.fit(Dataset0[Feature_names], Dataset0["Y"])
        neigh1.fit(Dataset1[Feature_names], Dataset1["Y"])
        Dataset["Yk0"] = neigh0.predict(Dataset[Feature_names])
        Dataset["Yk1"] = neigh1.predict(Dataset[Feature_names])
        Dataset0["Yk0"] = Dataset.loc[Dataset["T"] == 0, "Yk0"]
        Dataset0["Yk1"] = Dataset.loc[Dataset["T"] == 0, "Yk1"]
        Dataset1["Yk0"] = Dataset.loc[Dataset["T"] == 1, "Yk0"]
        Dataset1["Yk1"] = Dataset.loc[Dataset["T"] == 1, "Yk1"]

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
