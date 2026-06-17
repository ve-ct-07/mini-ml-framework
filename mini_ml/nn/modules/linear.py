import numpy as np

from .base import Module
from mini_ml.nn.autograd import Value


class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(2.0 / max(1, in_features))

        self.weight = Value(
            np.random.randn(in_features, out_features) * scale
        )

        if bias:
            self.bias = Value(
                np.zeros((1, out_features)),
                label='bias'
            )
        else:
            self.register_parameter("bias", None)

    def __call__(self, x):

        out = x @ self.weight

        if self._parameters["bias"] is not None:
            out = out + self.bias

        return out

    def __repr__(self):

        bias = self._parameters["bias"] is not None

        return (
            f"Linear("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={bias})"
        )
