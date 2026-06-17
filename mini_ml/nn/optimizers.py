import numpy as np


class SGD:

    def __init__(self, params, lr=0.01):

        self.params = list(params)
        self.lr = lr

    def step(self):

        for p in self.params:

            if p.grad is None:
                continue

            p.data -= self.lr * p.grad

    def zero_grad(self):

        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def __repr__(self):
        return f"SGD(lr={self.lr})"
