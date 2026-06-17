import numpy as np


class Perceptron:

    def __init__(
        self,
        learning_rate=0.01,
        max_iters=1000,
        random_state=None,
    ):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.random_state = random_state

        self.w_ = None
        self.errors_ = []

    @staticmethod
    def _add_bias(X):
        return np.column_stack((np.ones(X.shape[0]), X))

    def fit(self, X, y):

        rng = np.random.default_rng(
            self.random_state
        )

        X_aug = self._add_bias(X)

        self.w_ = rng.normal(
            0.0,
            0.01,
            X_aug.shape[1],
        )

        classes = np.unique(y)

        if set(classes) == {0, 1}:
            y_mod = np.where(y == 0, -1, 1)
        elif set(classes) == {-1, 1}:
            y_mod = y
        else:
            raise ValueError(
                "Labels must be {0,1} or {-1,1}"
            )

        best_w = self.w_.copy()
        best_err = np.inf

        for _ in range(self.max_iters):

            errors = 0

            for xi, target in zip(X_aug, y_mod):

                pred = 1 if xi @ self.w_ >= 0 else -1

                if pred != target:

                    self.w_ += (
                        self.learning_rate
                        * (target - pred)
                        * xi
                    )

                    errors += 1

            self.errors_.append(errors)

            if errors < best_err:
                best_err = errors
                best_w = self.w_.copy()

            if errors == 0:
                break

        self.w_ = best_w
        self.classes_ = classes

        return self

    def predict(self, X):
        scores = self._add_bias(X) @ self.w_

        preds = np.where(scores >= 0, 1, -1)

        if np.array_equal(self.classes_, [0, 1]):
            return np.where(preds == -1, 0, 1)

        return preds

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
