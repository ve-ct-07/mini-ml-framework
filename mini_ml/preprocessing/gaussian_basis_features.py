import numpy as np
from sklearn.cluster import KMeans


class GaussianBasisFeatures:
    """
    Gaussian radial basis function (RBF) feature expansion.

    Features are generated using:

        phi_j(x) = exp(
            -||x - c_j||² / (2 * sigma²)
        )

    where c_j denotes the j-th basis center.

    Parameters
    ----------
    n_centers : int, default=100
        Number of RBF centers.

    sigma : float or {"auto"}, default=50.0
        Bandwidth of the Gaussian basis functions.

        If "auto", sigma is estimated from the median
        pairwise distance between centers.

    init : {"random", "kmeans"}, default="kmeans"
        Strategy used to choose basis centers.

    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_centers=100,
        sigma=50.0,
        init="kmeans",
        random_state=None,
    ):
        self.n_centers = n_centers
        self.sigma = sigma
        self.init = init
        self.random_state = random_state

        if init not in {"random", "kmeans"}:
            raise ValueError(
                "init must be 'random' or 'kmeans'."
            )

        if sigma != "auto" and sigma <= 0:
            raise ValueError(
                "sigma must be positive or 'auto'."
            )

        self.centers_ = None
        self.sigma_ = None

    def fit(self, X, y=None):
        """
        Fit basis centers.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        n_samples = X.shape[0]

        if self.n_centers > n_samples:
            raise ValueError(
                "n_centers cannot exceed "
                "the number of samples."
            )

        n_centers = min(
            self.n_centers,
            n_samples,
        )

        if self.init == "random":

            rng = np.random.default_rng(
                self.random_state
            )

            indices = rng.choice(
                n_samples,
                size=n_centers,
                replace=False,
            )

            self.centers_ = X[indices]

        else:
            # K-Means center initialization
            kmeans = KMeans(
                n_clusters=n_centers,
                random_state=self.random_state,
                n_init="auto",
            )

            kmeans.fit(X)

            self.centers_ = (
                kmeans.cluster_centers_
                .astype(np.float64, copy=False)
            )

        if self.sigma == "auto":

            diff = (
                self.centers_[None, :, :]
                - self.centers_[:, None, :]
            )

            pairwise = np.sqrt(
                np.maximum(
                    np.sum(diff**2, axis=2),
                    0.0,
                )
            )

            nonzero = pairwise[pairwise > 0]

            if len(nonzero) == 0:
                self.sigma_ = 1.0
            else:
                self.sigma_ = np.median(nonzero)

        else:
            self.sigma_ = float(self.sigma)

        return self

    def transform(self, X):
        """
        Transform data into Gaussian basis space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        phi : ndarray of shape
            (n_samples, n_centers)
        """
        if self.centers_ is None:
            raise RuntimeError(
                "GaussianBasisFeatures "
                "has not been fitted."
            )

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        # Squared norms of input samples
        x_norms = np.sum(
            X**2,
            axis=1,
            keepdims=True,
        )

        # Squared norms of centers
        c_norms = np.sum(
            self.centers_**2,
            axis=1,
            keepdims=True,
        ).T

        # Memory-efficient squared Euclidean distance:
        #
        # ||x - c||² =
        # ||x||² - 2 x·c + ||c||²
        #
        sq_dist = (
            x_norms
            - 2.0 * np.dot(X, self.centers_.T)
            + c_norms
        )

        sq_dist = np.maximum(
            sq_dist,
            0.0,
        )

        sq_dist = np.nan_to_num(
            sq_dist,
            nan=0.0,
            posinf=np.finfo(np.float64).max,
        )

        return np.exp(
            -sq_dist /
            (2.0 * self.sigma_**2)
        )

    def fit_transform(self, X, y=None):
        """
        Fit and transform in a single call.
        """
        return self.fit(X, y).transform(X)

    def get_params(self):
        """
        Return estimator parameters.
        """
        return {
            "n_centers": self.n_centers,
            "sigma": self.sigma,
            "init": self.init,
            "random_state": self.random_state,
        }
