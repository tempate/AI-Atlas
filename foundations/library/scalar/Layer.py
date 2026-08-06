from library.scalar.Neuron import Neuron


class Layer:
    def __init__(self, nin, size):
        self.neurons = [Neuron(nin) for _ in range(size)]

    def __call__(self, xs):
        return [neuron(xs) for neuron in self.neurons]

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
