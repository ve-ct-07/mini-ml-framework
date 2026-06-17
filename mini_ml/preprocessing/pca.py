import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        # Compute covariance matrix
        cov = np.cov(X_centered, rowvar=False)
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort eigenvectors by descending eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        self.components_ = eigenvectors[:, idx[:self.n_components]]
        return self

    def transform(self, X):
        return np.dot(X - self.mean_, self.components_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
