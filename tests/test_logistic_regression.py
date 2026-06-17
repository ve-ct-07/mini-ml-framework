import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_ml.linear_models.classification import (
    LogisticRegression
)


def test_or_gate():

    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])

    y = np.array([0,1,1,1])

    model = LogisticRegression(
        lr=0.1,
        epochs=500,
        alpha=0
    )

    model.fit(X,y)

    preds = model.predict(X)

    assert np.all(preds == y)
