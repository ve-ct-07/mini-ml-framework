import numpy as np


class PCA:
    """
    Performs dimensionality reduction by projecting
    data onto the directions of maximum variance.
    """

    def __init__(self, n_components):
        self.n_components = n_components

        if n_components <= 0:
            raise ValueError(
                "n_components must be positive."
            )

        self.components_ = None
        self.mean_ = None

        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X, y=None):
        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        n_features = X.shape[1]

        if self.n_components > n_features:
            raise ValueError(
                "n_components cannot exceed "
                "the number of features."
            )

        self.mean_ = np.mean(
            X,
            axis=0,
        )

        X_centered = X - self.mean_

        covariance = np.cov(
            X_centered,
            rowvar=False,
        )

        eigenvalues, eigenvectors = (
            np.linalg.eigh(covariance)
        )

        order = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.components_ = eigenvectors[
            :, : self.n_components
        ]

        self.explained_variance_ = (
            eigenvalues[
                : self.n_components
            ]
        )

        total_variance = np.sum(
            eigenvalues
        )

        self.explained_variance_ratio_ = (
            self.explained_variance_
            / total_variance
        )

        return self

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError(
                "PCA has not been fitted."
            )

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        return np.dot(
            X - self.mean_,
            self.components_,
        )

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self):
        """
        Return estimator parameters.
        """
        return {
            "n_components": (
                self.n_components
            ),
        }