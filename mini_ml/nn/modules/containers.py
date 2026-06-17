from collections import OrderedDict

from .base import Module


class Sequential(Module):

    def __init__(self, *layers):

        super().__init__()

        if len(layers) == 1 and isinstance(layers[0], OrderedDict):

            for name, layer in layers[0].items():
                self.add_module(name, layer)

        else:

            for i, layer in enumerate(layers):
                self.add_module(str(i), layer)

    def __call__(self, x):

        for module in self._modules.values():
            x = module(x)

        return x

    def __repr__(self):

        lines = []

        for name, module in self._modules.items():
            lines.append(f"  ({name}): {module}")

        return "Sequential(\n" + "\n".join(lines) + "\n)"
