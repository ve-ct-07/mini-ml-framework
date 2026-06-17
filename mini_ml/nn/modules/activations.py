from .base import Module


class ReLU(Module):

    def __call__(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):

    def __call__(self, x):
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"


class Softmax(Module):

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def __call__(self, x):
        return x.softmax(axis=self.axis)

    def __repr__(self):
        return f"Softmax(axis={self.axis})"
