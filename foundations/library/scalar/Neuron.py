from library.scalar.Value import Value

import random


class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self, xs):
        output = self.bias + sum(x * w for x, w in zip(xs, self.weights))
        return output.tanh()

    def parameters(self):
        return self.weights + [self.bias]
