from library.Neuron import Neuron


class Layer:
    def __init__(self, nin, size):
        self.neurons = [Neuron(nin) for _ in range(size)]

    def __call__(self, xs):
        output = []

        for neuron in self.neurons:
            output.append(neuron(xs))

        return output

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
