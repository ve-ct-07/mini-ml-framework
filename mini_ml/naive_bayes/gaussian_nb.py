import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.
    """

    def __init__(
        self,
        epsilon: float = 1e-9,
    ):
        self.epsilon = epsilon

        self.classes_ = None
        self.class_priors_ = None
        self.theta_ = None
        self.var_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )
        
        y = np.asarray(y)

        self.classes_, counts = np.unique(
            y,
            return_counts=True,
        )

        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_priors_ = (
            counts / len(y)
        )

        self.theta_ = np.zeros(
            (n_classes, n_features)
        )

        self.var_ = np.zeros(
            (n_classes, n_features)
        )

        for idx, cls in enumerate(
            self.classes_
        ):
            X_cls = X[y == cls]

            self.theta_[idx] = np.mean(
                X_cls,
                axis=0,
            )

            self.var_[idx] = (
                np.var(X_cls, axis=0)
                + self.epsilon
            )

        return self

    def _gaussian_log_pdf(
        self,
        X: np.ndarray,
        class_idx: int,
    ) -> np.ndarray:

        mean = self.theta_[class_idx]
        var = self.var_[class_idx]

        return -0.5 * np.sum(
            np.log(
                2.0 * np.pi * var
            )
            + ((X - mean) ** 2) / var,
            axis=1,
        )

    def predict_log_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        if self.classes_ is None:
            raise RuntimeError(
                "Model has not been fitted."
            )

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        joint = []

        for idx in range(
            len(self.classes_)
        ):

            log_prob = (
                np.log(
                    self.class_priors_[idx]
                )
                + self._gaussian_log_pdf(
                    X,
                    idx,
                )
            )

            joint.append(log_prob)

        return np.column_stack(joint)

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        log_prob = self.predict_log_proba(X)

        log_prob -= np.max(
            log_prob,
            axis=1,
            keepdims=True,
        )

        prob = np.exp(log_prob)

        return prob / prob.sum(
            axis=1,
            keepdims=True,
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        idx = np.argmax(
            self.predict_log_proba(X),
            axis=1,
        )

        return self.classes_[idx]

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:

        return np.mean(
            self.predict(X) == y
        )
