import torch
from torch import nn


class IdentityNet(nn.Module):
    """
    A dummy network that learns to output its inputs.
    """

    def __init__(self, num_channels: int):
        """
        Initialize the model.
        """

        super().__init__()

        self.model = nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform a single forward pass through the dummy network.

        :param inputs: The input tensor.
        :type inputs: torch.Tensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """

        return self.model(inputs)
