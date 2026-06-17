from collections import OrderedDict
import numpy as np

from mini_ml.nn.autograd import Value


class Module:
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def register_parameter(self, name, param):
        if param is not None and not isinstance(param, Value):
            raise TypeError(
                f"cannot assign {type(param)} as parameter '{name}'"
            )

        if "." in name:
            raise KeyError("parameter name can't contain '.'")

        if name == "":
            raise KeyError("parameter name can't be empty")

        self._parameters[name] = param

    def add_module(self, name, module):
        if module is not None and not isinstance(module, Module):
            raise TypeError(
                f"{module} is not a Module subclass"
            )

        if "." in name:
            raise KeyError("module name can't contain '.'")

        if name == "":
            raise KeyError("module name can't be empty")

        self._modules[name] = module

    def _get_named_parameters(self, prefix="", memo=None):
        """
        Recursively yields all named parameters.
        """

        if memo is None:
            memo = set()

        for name, param in self._parameters.items():

            if param is None:
                continue

            if id(param) in memo:
                continue

            memo.add(id(param))

            full_name = prefix + ("." if prefix else "") + name

            yield full_name, param

        for name, module in self._modules.items():

            if module is None:
                continue

            child_prefix = prefix + ("." if prefix else "") + name

            yield from module._get_named_parameters(
                child_prefix,
                memo
            )

    def parameters(self):
        for _, param in self._get_named_parameters():
            yield param

    def zero_grad(self):
        for p in self.parameters():

            if p.grad is None:
                p.grad = np.zeros_like(
                    p.data,
                    dtype=np.float64
                )
            else:
                p.grad.fill(0.0)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def state_dict(self):
        return {
            name: param.data
            for name, param in self._get_named_parameters()
        }

    def save_state_dict(self, filepath):
        np.savez_compressed(
            filepath,
            **self.state_dict()
        )

    def load_state_dict(self, filepath):

        loaded = np.load(filepath)

        current_params = dict(
            self._get_named_parameters()
        )

        for name, param in current_params.items():

            if name not in loaded:
                continue

            if param.data.shape != loaded[name].shape:
                raise ValueError(
                    f"Shape mismatch for '{name}'"
                )

            param.data[:] = loaded[name]

    def __setattr__(self, name, value):

        if (
            name.startswith("_")
            or not isinstance(value, (Value, Module))
        ):
            super().__setattr__(name, value)
            return

        if isinstance(value, Value):

            if "_parameters" not in self.__dict__:
                raise AttributeError(
                    "cannot assign parameter before Module.__init__()"
                )

            self._modules.pop(name, None)
            self.register_parameter(name, value)

        elif isinstance(value, Module):

            if "_modules" not in self.__dict__:
                raise AttributeError(
                    "cannot assign module before Module.__init__()"
                )

            self._parameters.pop(name, None)
            self.add_module(name, value)

        super().__setattr__(name, value)
