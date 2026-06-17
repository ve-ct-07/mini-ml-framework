import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_ml.preprocessing import (
    StandardScaler
)


def test_scaler():

    rng = np.random.default_rng(42)

    X = rng.normal(size=(1000,3))

    scaler = StandardScaler()

    Xs = scaler.fit_transform(X)

    assert np.allclose(
        np.mean(Xs,axis=0),
        0,
        atol=1e-10
    )

    assert np.allclose(
        np.std(Xs,axis=0),
        1,
        atol=1e-10
    )
