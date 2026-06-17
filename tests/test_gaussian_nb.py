import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_ml.naive_bayes import (
    GaussianNaiveBayes
)


def test_training_accuracy():

    rng = np.random.default_rng(42)

    X0 = rng.normal(0,1,(100,2))
    X1 = rng.normal(3,1,(100,2))

    X = np.vstack([X0,X1])

    y = np.array(
        [0]*100 + [1]*100
    )

    model = GaussianNaiveBayes()

    model.fit(X,y)

    assert model.score(X,y) > 0.9
