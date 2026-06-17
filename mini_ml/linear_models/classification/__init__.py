"""
Classification models.
"""

from .lda import LinearDiscriminantAnalysis
from .least_squares import LeastSquaresClassifier
from .logistic_regression import LogisticRegression
from .perceptron import Perceptron

__all__ = [
    "LinearDiscriminantAnalysis",
    "LeastSquaresClassifier",
    "LogisticRegression",
    "Perceptron",
]
