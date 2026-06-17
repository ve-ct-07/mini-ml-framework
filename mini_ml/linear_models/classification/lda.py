import numpy as np


class LinearDiscriminantAnalysis:
    """
    Binary discriminant classifier using
    least-squares target encoding.

    Note:
    This is not Fisher's classical LDA.
    """

    def __init__(self):
        self.w_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        return np.column_stack((np.ones(X.shape[0]), X))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError(
                "Only binary classification is supported."
            )

        n_samples = X.shape[0]

        n0 = np.sum(y == self.classes_[0])
        n1 = np.sum(y == self.classes_[1])

        T = np.where(
            y == self.classes_[1],
            n_samples / n1,
            -n_samples / n0,
        )

        X_aug = self._add_bias(X)

        self.w_ = np.linalg.pinv(X_aug) @ T

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Model has not been fitted.")

        scores = self._add_bias(X) @ self.w_

        return np.where(
            scores > 0,
            self.classes_[1],
            self.classes_[0],
        )

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
