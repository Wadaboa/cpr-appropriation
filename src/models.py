import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork


class FCNetwork(TorchModelV2, nn.Module):
    """
    Custom RLlib network that delegates to fully-connected layers
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.torch_sub_model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


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
        ), "Input dimensions should be given as an integer"
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

    def reset_parameters(self):
        """
        Set network's parameters to their initial values
        """
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x):
        """
        Perform a single example or batch forward pass
        """
        start_dim = 0 if not self.training else 2
        x = self.mlp(torch.flatten(x, start_dim=start_dim))
        return self.out(x)
