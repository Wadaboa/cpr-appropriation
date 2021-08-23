import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron to be used with columnar-type
    fixed-size input vectors
    """

    def __init__(self, input_size, hidden_dims, output_size, non_linearity=nn.ReLU):
        assert isinstance(
            input_size, int
        ), "Input dimensions should be given as an integer"
        assert isinstance(
            hidden_dims, list
        ), "Hidden dimensions should be given as a list of integers"
        assert isinstance(
            output_size, int
        ), "Output dimensions should be given as an integer"
        super(MLP, self).__init__()

        hidden_dims += [output_size]
        linear_layers = [nn.Linear(input_size, hidden_dims[0]), non_linearity()]
        for h in range(1, len(hidden_dims)):
            linear_layers += [
                nn.Linear(hidden_dims[h - 1], hidden_dims[h]),
                non_linearity(),
            ]
        self.mlp = nn.Sequential(*linear_layers)
        self.out = nn.LogSoftmax()

    def forward(self, x):
        """
        Perform a single example or batch forward pass
        """
        start_dim = 0 if not self.training else 2
        x = self.mlp(torch.flatten(x, start_dim=start_dim))
        return self.out(x)
