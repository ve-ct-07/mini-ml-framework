import numpy as np
from itertools import combinations_with_replacement


class PolynomialFeatures:
    """
    Generate polynomial and interaction features.
    Supports a memory-optimized 'pure_squares_only' mode to bypass 
    combinatorial explosions on high-dimensional datasets.
    """

    def __init__(
        self,
        degree=2,
        include_bias=True,
        pure_squares_only=False,
    ):
        self.degree = degree
        self.include_bias = include_bias
        self.pure_squares_only = pure_squares_only

        if degree < 0:
            raise ValueError(
                "degree must be non-negative."
            )

        self.n_input_features_ = None
        self.n_output_features_ = None
        self.powers_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)

        n_features = X.shape[1]

        self.n_input_features_ = n_features

        combinations = []

        start_degree = (
            0 if self.include_bias else 1
        )

        for degree in range(
            start_degree,
            self.degree + 1,
        ):
            raw_combinations = combinations_with_replacement(
                range(n_features),
                degree,
            )
            
            if self.pure_squares_only:
                for comb in raw_combinations:
                    if len(set(comb)) <= 1:
                        combinations.append(comb)
            else:
                combinations.extend(raw_combinations)

        self.powers_ = combinations

        self.n_output_features_ = len(
            self.powers_
        )

        return self

    def transform(self, X):
        if self.powers_ is None:
            raise RuntimeError(
                "PolynomialFeatures has not been fitted."
            )

        X = np.asarray(X)

        n_samples = X.shape[0]

        X_poly = np.empty(
            (
                n_samples,
                self.n_output_features_,
            ),
            dtype=np.float64,
        )

        for i, comb in enumerate(
            self.powers_
        ):

            if len(comb) == 0:
                X_poly[:, i] = 1.0
            else:
                X_poly[:, i] = np.prod(
                    X[:, comb],
                    axis=1,
                )

        return X_poly

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
