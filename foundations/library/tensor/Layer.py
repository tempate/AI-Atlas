from library.tensor.Tensor import Tensor
import numpy as np


class Layer:
    def __init__(self, nin, nout):
        self.shape = (nin, nout)
        self.weights = Tensor(np.random.uniform(-1,1,size=self.shape))
        self.bias = Tensor(np.random.uniform(-1,1,size=(nout,)))

    def __call__(self, xs):
        out = xs @ self.weights + self.bias
        return out.tanh()

    def parameters(self):
        return [self.weights, self.bias]
