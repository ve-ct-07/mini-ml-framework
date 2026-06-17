import numpy as np


class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes classifier.

    Assumes binary-valued features.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        if not np.all(
            np.logical_or(X == 0, X == 1)
        ):
            raise ValueError(
                "BernoulliNB requires binary features."
            )

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

        self.class_log_prior_ = np.log(
            counts / len(y)
        )

        self.feature_log_prob_ = np.zeros(
            (n_classes, n_features)
        )

        for idx, cls in enumerate(self.classes_):

            X_cls = X[y == cls]

            n_cls = len(X_cls)

            feature_counts = np.sum(
                X_cls,
                axis=0,
            )

            probs = (
                feature_counts + self.alpha
            ) / (
                n_cls + 2 * self.alpha
            )

            self.feature_log_prob_[idx] = np.log(
                probs
            )

        return self

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

        log_p1 = self.feature_log_prob_

        log_p0 = np.log(
            np.clip(
                1.0 - np.exp(log_p1),
                1e-12,
                None,
            )
        )

        joint = []

        for cls_idx in range(
            len(self.classes_)
        ):

            log_likelihood = (
                X * log_p1[cls_idx]
                + (1 - X) * log_p0[cls_idx]
            ).sum(axis=1)

            joint.append(
                self.class_log_prior_[cls_idx]
                + log_likelihood
            )

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
