import numpy as np


class RidgeRegression:
    """
    Ridge Regression using the closed-form solution.

    Minimizes:

        ||y - Xw||² + alpha ||w||²
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        return np.column_stack((np.ones(X.shape[0]), X))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples = X.shape[0]

        if self.fit_intercept:
            X_train = self._add_bias(X)
        else:
            X_train = X

        n_params = X_train.shape[1]

        reg = np.eye(n_params)

        if self.fit_intercept:
            reg[0, 0] = 0.0

        A = X_train.T @ X_train + self.alpha * reg
        b = X_train.T @ y

        w = np.linalg.solve(A, b)

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError(
                "Model has not been fitted."
            )

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        return X @ self.coef_ + self.intercept_

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        R² coefficient of determination.
        """
        y = np.asarray(y)

        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1.0 - (ss_res / ss_tot)
