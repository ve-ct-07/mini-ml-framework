import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_ml.linear_models.regression import (
    RidgeRegression
)


def test_linear_fit():

    rng = np.random.default_rng(42)

    X = rng.normal(size=(200,1))

    y = 3 * X[:,0] + 2

    model = RidgeRegression(alpha=0)

    model.fit(X,y)

    preds = model.predict(X)

    mse = np.mean((preds-y)**2)

    assert mse < 1e-10
