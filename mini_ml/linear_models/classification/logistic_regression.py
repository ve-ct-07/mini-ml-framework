import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression using mini-batch SGD with L2 regularization.

    Parameters
    ----------
    alpha : float, default=0.01
        L2 regularization strength.

    epochs : int, default=50
        Number of training epochs.

    lr : float, default=0.01
        Learning rate.

    batch_size : int, default=64
        Mini-batch size.

    fit_intercept : bool, default=True
        Whether to include a bias term.

    random_state : int or None, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 64,
        fit_intercept: bool = True,
        random_state: int | None = None,
    ):
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        self.w_: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid.
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        """
        Add intercept column to feature matrix.
        """
        return np.column_stack((np.ones(X.shape[0]), X))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train logistic regression using mini-batch SGD.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        y = np.asarray(y, dtype=np.float64)

        unique_labels = np.unique(y)

        if len(unique_labels) != 2:
            raise ValueError(
                "Binary classification requires exactly two classes."
            )

        if not np.array_equal(unique_labels, [0, 1]):
            raise ValueError(
                "Labels must be encoded as {0,1}."
            )

        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_train = self._add_bias(X)
            self.w_ = np.zeros(n_features + 1)
        else:
            X_train = X
            self.w_ = np.zeros(n_features)

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.epochs):

            # Shuffle examples at each epoch
            indices = rng.permutation(n_samples)

            X_shuffled = X_train[indices]
            y_shuffled = y[indices]

            # Mini-batch SGD updates
            for start in range(
                0,
                n_samples,
                self.batch_size,
            ):
                stop = start + self.batch_size

                X_batch = X_shuffled[start:stop]
                y_batch = y_shuffled[start:stop]

                # Forward pass
                probs = self._sigmoid(
                    X_batch @ self.w_
                )

                # Binary cross-entropy gradient
                gradient = (
                    X_batch.T @ (probs - y_batch)
                ) / len(y_batch)

                # L2 regularization (excluding intercept)
                if self.alpha > 0:

                    if self.fit_intercept:
                        gradient[1:] += (
                            self.alpha
                            * self.w_[1:]
                            / len(y_batch)
                        )
                    else:
                        gradient += (
                            self.alpha
                            * self.w_
                            / len(y_batch)
                        )

                self.w_ -= self.lr * gradient

        return self

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Returns
        -------
        ndarray of shape (n_samples, 2)

        Columns correspond to:
        [P(class=0), P(class=1)]
        """
        if self.w_ is None:
            raise RuntimeError(
                "Model has not been fitted."
            )

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        if self.fit_intercept:
            X = self._add_bias(X)

        probs_1 = self._sigmoid(X @ self.w_)
        probs_0 = 1.0 - probs_1

        return np.column_stack(
            (probs_0, probs_1)
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict binary class labels.
        """
        return (
            self.predict_proba(X)[:, 1] >= 0.5
        ).astype(np.int64)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Compute classification accuracy.
        """
        return np.mean(
            self.predict(X) == y
        )
