import torch
from torch import Tensor


class Constant(torch.nn.Module):
    """
    Module that returns a constant tensor.
    """

    def __init__(self, value: Tensor, trainable: bool) -> None:
        super().__init__()
        value = value.clone().detach()

        if trainable:
            self.value = torch.nn.Parameter(value)
        else:
            self.register_buffer("value", value)

        assert isinstance(self.value, Tensor)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.value
