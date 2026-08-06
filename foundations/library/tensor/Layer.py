from library.tensor.Tensor import Tensor
import numpy as np


class Layer:
    def __init__(self, nin, nout, activation=True):
        self.shape = (nin, nout)
        self.activation = activation

        # Initialize the weights and the bias
        self.weights = Tensor(np.random.randn(*self.shape) * np.sqrt(2 / nin))
        self.bias = Tensor(np.zeros(nout))

    def __call__(self, x):
        out = x @ self.weights + self.bias
        return out.tanh() if self.activation else out

    def parameters(self):
        return [self.weights, self.bias]
