import numpy as np


class Value:
    """
    Tensor object supporting automatic differentiation.
    Stores data, gradient, and computation graph information.
    """

    def __init__(self, data, _parents=(), _op="", label=""):

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)

        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float64)

        self.data = data
        self.grad = np.zeros_like(data)

        self._prev = tuple(_parents)
        self._op = _op
        self.label = label

        self._backward = lambda: None

    def __repr__(self):
        return f"Value(shape={self.data.shape}, op='{self._op}')"

    @staticmethod
    def _unbroadcast(grad, shape):
        """
        Reduce broadcasted gradients back to original shape.
        """

        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for axis, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad

    # ==========================================================
    # Basic Arithmetic
    # ==========================================================

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():

            grad_self = out.grad
            grad_other = out.grad

            if grad_self.shape != self.data.shape:
                grad_self = Value._unbroadcast(
                    grad_self,
                    self.data.shape
                )

            if grad_other.shape != other.data.shape:
                grad_other = Value._unbroadcast(
                    grad_other,
                    other.data.shape
                )

            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():

            grad_self = out.grad * other.data
            grad_other = out.grad * self.data

            if grad_self.shape != self.data.shape:
                grad_self = Value._unbroadcast(
                    grad_self,
                    self.data.shape
                )

            if grad_other.shape != other.data.shape:
                grad_other = Value._unbroadcast(
                    grad_other,
                    other.data.shape
                )

            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __pow__(self, power):

        if not isinstance(power, (int, float)):
            raise TypeError(
                "Power only supports scalar exponents."
            )

        out = Value(
            self.data ** power,
            (self,),
            f"**{power}"
        )

        def _backward():
            self.grad += (
                out.grad
                * power
                * (self.data ** (power - 1))
            )

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        other = (
            other
            if isinstance(other, Value)
            else Value(other)
        )
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = (
            other
            if isinstance(other, Value)
            else Value(other)
        )
        return other * (self ** -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # ==========================================================
    # Matrix Multiplication
    # ==========================================================

    def __matmul__(self, other):

        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data @ other.data,
            (self, other),
            "@"
        )

        def _backward():

            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    # ==========================================================
    # Activations
    # ==========================================================

    def relu(self):

        out = Value(
            np.maximum(0.0, self.data),
            (self,),
            "relu"
        )

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def sigmoid(self):

        s = 1.0 / (
            1.0 + np.exp(-np.clip(self.data, -500, 500))
        )

        out = Value(
            s,
            (self,),
            "sigmoid"
        )

        def _backward():
            self.grad += out.grad * s * (1.0 - s)

        out._backward = _backward
        return out

    def softmax(self, axis=-1):

        shifted = self.data - np.max(
            self.data,
            axis=axis,
            keepdims=True
        )

        exp_vals = np.exp(shifted)

        probs = exp_vals / np.sum(
            exp_vals,
            axis=axis,
            keepdims=True
        )

        out = Value(
            probs,
            (self,),
            "softmax"
        )

        def _backward():

            grad = (
                out.grad
                - np.sum(
                    out.grad * probs,
                    axis=axis,
                    keepdims=True
                )
            )

            self.grad += grad * probs

        out._backward = _backward
        return out

    # ==========================================================
    # Elementary Functions
    # ==========================================================

    def exp(self):

        e = np.exp(
            np.clip(self.data, -500, 500)
        )

        out = Value(
            e,
            (self,),
            "exp"
        )

        def _backward():
            self.grad += out.grad * e

        out._backward = _backward
        return out

    def log(self):

        eps = 1e-12

        safe = np.maximum(self.data, eps)

        out = Value(
            np.log(safe),
            (self,),
            "log"
        )

        def _backward():
            self.grad += out.grad / safe

        out._backward = _backward
        return out

    def clip(self, min_val, max_val):

        out = Value(
            np.clip(self.data, min_val, max_val),
            (self,),
            "clip"
        )

        def _backward():

            mask = (
                (self.data >= min_val)
                & (self.data <= max_val)
            )

            self.grad += out.grad * mask

        out._backward = _backward
        return out

    # ==========================================================
    # Reductions
    # ==========================================================

    def sum(self, axis=None, keepdims=False):

        out = Value(
            np.sum(
                self.data,
                axis=axis,
                keepdims=keepdims
            ),
            (self,),
            "sum"
        )

        def _backward():

            grad = out.grad

            if axis is not None and not keepdims:

                shape = list(self.data.shape)

                if isinstance(axis, int):
                    shape[axis] = 1
                else:
                    for ax in axis:
                        shape[ax] = 1

                grad = grad.reshape(shape)

            self.grad += np.broadcast_to(
                grad,
                self.data.shape
            )

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):

        if axis is None:
            divisor = self.data.size
        else:
            if isinstance(axis, int):
                divisor = self.data.shape[axis]
            else:
                divisor = np.prod(
                    [self.data.shape[a] for a in axis]
                )

        return self.sum(
            axis=axis,
            keepdims=keepdims
        ) / divisor

    # ==========================================================
    # Backpropagation
    # ==========================================================

    def backward(self):

        # Build topological ordering of graph
        topo = []
        visited = set()

        def build(v):

            if id(v) not in visited:

                visited.add(id(v))

                for child in v._prev:
                    build(child)

                topo.append(v)

        build(self)

        # Seed gradient of output node
        self.grad = np.ones_like(
            self.data,
            dtype=np.float64
        )

        # Reverse-mode backpropagation
        for node in reversed(topo):
            node._backward()
