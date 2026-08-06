from library.tensor.Tensor import Tensor
import numpy as np


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, xs):
        out = Tensor(xs)
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)
