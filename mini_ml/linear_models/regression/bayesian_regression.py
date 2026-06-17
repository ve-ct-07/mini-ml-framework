import numpy as np


class BayesianRegression:
    """
    Bayesian Linear Regression with Gaussian basis functions.
    """

    def __init__(
        self,
        n_basis: int = 25,
        basis_sigma_fraction: float = 0.1,
        alpha: float = 1.0,
        beta: float = 100.0,
    ):
        self.n_basis = n_basis
        self.basis_sigma_fraction = basis_sigma_fraction
        self.alpha = alpha
        self.beta = beta

        self.basis_centers_ = None
        self.basis_sigma_ = None

        self.posterior_mean_ = None
        self.posterior_cov_ = None

    def _gaussian_basis(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        diff = X - self.basis_centers_.T

        phi = np.exp(
            -(diff ** 2)
            / (2.0 * self.basis_sigma_ ** 2)
        )

        return np.column_stack(
            (
                np.ones(X.shape[0]),
                phi,
            )
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        X = np.asarray(
            X,
            dtype=np.float64,
        ).reshape(-1, 1)

        y = np.asarray(
            y,
            dtype=np.float64,
        ).ravel()

        if self.n_basis > 1:

            self.basis_centers_ = np.linspace(
                X.min(),
                X.max(),
                self.n_basis - 1,
            ).reshape(-1, 1)

            spacing = (
                (X.max() - X.min())
                / max(self.n_basis - 2, 1)
            )

            self.basis_sigma_ = (
                self.basis_sigma_fraction
                * spacing
            )

        else:

            self.basis_centers_ = np.empty((0, 1))
            self.basis_sigma_ = 1.0

        Phi = self._gaussian_basis(X)

        S_inv = (
            self.alpha * np.eye(self.n_basis)
            + self.beta * Phi.T @ Phi
        )

        self.posterior_cov_ = np.linalg.pinv(
            S_inv
        )

        self.posterior_mean_ = (
            self.beta
            * self.posterior_cov_
            @ Phi.T
            @ y
        )

        return self

    def predict_dist(
        self,
        X: np.ndarray,
    ):
        if self.posterior_mean_ is None:
            raise RuntimeError(
                "Model has not been fitted."
            )

        X = np.asarray(
            X,
            dtype=np.float64,
        ).reshape(-1, 1)

        Phi = self._gaussian_basis(X)

        mean = Phi @ self.posterior_mean_

        variance = (
            1.0 / self.beta
            + np.sum(
                (Phi @ self.posterior_cov_)
                * Phi,
                axis=1,
            )
        )

        return mean, variance

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        mean, _ = self.predict_dist(X)

        return mean

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        R² score.
        """
        y = np.asarray(y)

        y_pred = self.predict(X)

        ss_res = np.sum(
            (y - y_pred) ** 2
        )

        ss_tot = np.sum(
            (y - np.mean(y)) ** 2
        )

        if ss_tot == 0:
            return 0.0

        return 1.0 - ss_res / ss_tot
