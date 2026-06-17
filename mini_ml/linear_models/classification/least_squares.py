import numpy as np


class LeastSquaresClassifier:
    """
    Least-squares classifier supporting
    binary and multiclass classification.
    """

    def __init__(self):
        self.w_ = None
        self.classes_ = None
        self.w_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _add_bias(X):
        return np.column_stack((np.ones(X.shape[0]), X))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)

        X_aug = self._add_bias(X)

        if n_classes == 2:
            T = np.where(
                y == self.classes_[0],
                -1.0,
                1.0,
            ).reshape(-1, 1)
        else:
            T = np.eye(n_classes)[
                np.searchsorted(self.classes_, y)
            ]

        self.w_ = np.linalg.pinv(X_aug) @ T

        return self

    def predict(self, X):
        if self.w_ is None:
            raise RuntimeError("Model has not been fitted.")

        scores = self._add_bias(X) @ self.w_

        if scores.ndim == 1 or scores.shape[1] == 1:
            idx = (scores.ravel() > 0).astype(int)
        else:
            idx = np.argmax(scores, axis=1)

        return self.classes_[idx]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
