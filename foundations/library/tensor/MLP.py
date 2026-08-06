from library.tensor.Layer import Layer
from library.tensor.Sequential import Sequential


class MLP(Sequential):
    def __init__(self, nin, nout):
        sizes = [nin] + nout

        # Create the hidden layers
        hidden_layers = [
            Layer(sizes[i], sizes[i+1])
            for i in range(len(nout)-1)
        ]

        # Create the output layer
        output_layer = Layer(sizes[-2], sizes[-1],
                             activation=False)

        layers = hidden_layers + [output_layer]
        super().__init__(layers)
