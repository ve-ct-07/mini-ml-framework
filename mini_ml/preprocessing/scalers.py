import numpy as np


class StandardScaler:
    """
    Standardize features by removing the mean
    and scaling to unit variance.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        self.mean_ = np.mean(X, axis=0)

        self.scale_ = np.std(X, axis=0)

        self.scale_[self.scale_ == 0.0] = 1.0

        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError(
                "StandardScaler has not been fitted."
            )

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )
        
        if self.mean_ is None:
            raise RuntimeError(
                "StandardScaler has not been fitted."
            )

        return X * self.scale_ + self.mean_
