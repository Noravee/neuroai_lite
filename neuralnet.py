'''Neural network for using in NeuralNetworkPrescriptor'''

import torch
from torch import nn


class ForwardNeuralNet(nn.Module):
    '''A neural network that only do feedforward.'''
    def __init__(
            self,
            layers: list[int],
            activation: nn.Module | list[nn.Module] | None = None,
            output_activation: nn.Module | None = None,
            random_state: int | None = None):

        super().__init__()

        if random_state is not None:
            torch.manual_seed(random_state)

        self.layers = layers

        # Handle activation functions
        if isinstance(activation, list):
            if len(activation) != len(layers) - 2:
                raise ValueError("""Activation list length must be equal to
                                 the number of hidden layers.""")
            self.activation = activation
        else:
            self.activation = [activation] * (len(layers) - 2)

        # Construct the layers
        multilayers = []
        inputs = layers[0]
        for i, outputs in enumerate(layers[1:-1]):
            multilayers.append(nn.Linear(inputs, outputs))
            if self.activation[i] is not None:
                multilayers.append(self.activation[i])
            inputs = outputs

        multilayers.append(nn.Linear(layers[-2], layers[-1]))
        if output_activation is not None:
            multilayers.append(output_activation)
        self.network = nn.Sequential(*multilayers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Input data into network'''
        with torch.no_grad():
            y = self.network(x)
        return y
