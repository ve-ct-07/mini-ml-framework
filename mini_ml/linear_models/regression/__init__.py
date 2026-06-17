"""
Regression models.
"""

from .bayesian_regression import BayesianRegression
from .ridge import RidgeRegression

__all__ = [
    "BayesianRegression",
    "RidgeRegression",
]
