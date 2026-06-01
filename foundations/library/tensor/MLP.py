from library.tensor.Layer import Layer
from library.tensor.Tensor import Tensor
import numpy as np


class MLP:
    def __init__(self, nin, layers):
        self.sizes = [nin] + layers
        self.layers = [
            Layer(self.sizes[i], self.sizes[i+1])
            for i in range(len(layers))
        ]

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
