from typing import Any, Callable

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


class Lambda(torch.nn.Module):
    """
    Module that wraps an arbitrary function.
    """

    def __init__(
        self,
        fn: Callable,
        kwargs: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.name = name
        if kwargs is None:
            kwargs = {}
        self.addictional_kwargs = kwargs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs, **self.addictional_kwargs)

    def extra_repr(self) -> str:
        if self.name is not None:
            name = self.name
        else:
            name = self.fn.__name__

        if len(self.addictional_kwargs) > 0:
            additional = ", ".join(
                f"{k}={v}" for k, v in self.addictional_kwargs.items()
            )
            return f"fn={name}(..., {additional})"
        else:
            return f"fn={name}"
