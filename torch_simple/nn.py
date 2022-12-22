import inspect
import math
import numbers
from typing import Any, Callable, Iterable, Type

import torch
from torch import Tensor

from torch_simple.exceptions import IncompatibleShapesError
from torch_simple.functional import pad_to_shape, truncate_to_shape
from torch_simple.typedefs import Side
from torch_simple.utils import deep_copy_with_pickle_fallback


def get_activation(
    activation: str | torch.nn.Module | Callable[[Tensor], Tensor] | Type | None,
    kwargs: dict[str, Any] | None = None,
    return_identity_if_none: bool = False,
) -> torch.nn.Module | None:
    """
    Returns the activation function as a torch.nn.Module.
    It will create a copy if a module is passed, so
    that the parameters are not accidentally shared.
    """

    if kwargs is None:
        kwargs = {}

    if activation is None or activation == "none":
        if return_identity_if_none:
            return torch.nn.Identity()
        else:
            return None

    if inspect.isclass(activation):
        if issubclass(activation, torch.nn.Module):
            return activation(**kwargs)
        elif callable(activation):
            return Lambda(activation, kwargs)
        else:
            raise ValueError(f"Invalid activation: {activation}")

    if isinstance(activation, torch.nn.Module):
        # some loss functions can have parameters,
        # so we need to return a copy of the module
        # so the parameters are not shared
        return deep_copy_with_pickle_fallback(activation)

    if callable(activation):
        return Lambda(activation)

    if isinstance(activation, str):
        for scope in [torch, torch.nn, torch.nn.modules, torch.nn.functional]:
            fn = getattr(scope, activation, None)
            if fn is not None:
                break

        if fn is None:
            raise ValueError(f"Invalid activation: {activation}")

        if inspect.isclass(fn) or callable(fn):
            return get_activation(fn, kwargs)

    raise ValueError(f"Invalid activation: {activation}")


def get_dropout(
    drop: float | torch.nn.Module | Callable[[Tensor], Tensor] | None,
    return_dropout_if_none: bool = False,
) -> torch.nn.Module | None:

    if drop is None or (isinstance(drop, numbers.Real) and math.isclose(drop, 0.0)):
        if return_dropout_if_none:
            return torch.nn.Dropout(0.0)
        else:
            return None

    if isinstance(drop, torch.nn.Module):
        return deep_copy_with_pickle_fallback(drop)

    if isinstance(drop, float):
        return torch.nn.Dropout(drop)

    if callable(drop):
        return Lambda(drop)

    raise ValueError(f"Invalid dropout: {drop}")


def get_norm_layer(
    norm_layer: str | torch.nn.Module | Callable[[Tensor], Tensor] | Type | None,
    kwargs: dict[str, Any] | None = None,
    prefer_lazy: bool = True,
    return_identity_if_none: bool = False,
    num_features: int | None = None,
) -> torch.nn.Module | None:

    if kwargs is None:
        kwargs = {}

    if norm_layer is None or norm_layer == "none":
        return torch.nn.Identity() if return_identity_if_none else None

    if inspect.isclass(norm_layer):
        if inspect.signature(norm_layer.__init__).parameters.get("num_features") is not None:
            if num_features is None:
                raise ValueError("num_features must be specified if norm_layer is a class")
            kwargs["num_features"] = num_features
        return norm_layer(**kwargs)

    if isinstance(norm_layer, torch.nn.Module):
        return deep_copy_with_pickle_fallback(norm_layer)

    if isinstance(norm_layer, str):
        for scope in [torch, torch.nn, torch.nn.modules, torch.nn.functional]:
            fn = getattr(scope, norm_layer, None)
            fn_lazy = getattr(scope, f"Lazy{norm_layer}", None)
            if prefer_lazy and fn_lazy is not None:
                fn = fn_lazy

            if fn is not None:
                break

        if fn is None:
            raise ValueError(f"Invalid norm_layer: {norm_layer}")

        if inspect.isclass(fn):
            return get_norm_layer(fn, kwargs, prefer_lazy, return_identity_if_none, num_features)

        if callable(fn):
            return Lambda(fn, kwargs)

    if callable(norm_layer):
        return Lambda(norm_layer)

    raise ValueError(f"Invalid norm_layer: {norm_layer}")


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
            additional = ", ".join(f"{k}={v}" for k, v in self.addictional_kwargs.items())
            return f"fn={name}(..., {additional})"
        else:
            return f"fn={name}"


class Residual(torch.nn.Module):
    """
    Simple residual wrapper that adds a residual connection around a module.

    If the input and residual (output) shapes are the same, the input is added to the output.
    It can also pad/truncate the one to match the output shape of the other.


    >>> import torch
    >>> x = torch.randn(2, 10)
    >>> block = torch.nn.Linear(10, 10)
    >>> model = Residual(block)
    >>> torch.allclose(model(x), x + block(x))
    True

    >>> x = torch.randn(4, 5, 10, 10)
    >>> block = torch.nn.Conv2d(5, 5, (3, 3), padding=1)
    >>> model = Residual(block)
    >>> torch.allclose(model(x), x + block(x))
    True

    """

    def __init__(
        self,
        around: torch.nn.Module,
        enabled: bool = True,
        learn_skipping_weight: bool = False,
        learn_residual_weight: bool = False,
        pad_if_needed: bool = False,
        truncate_if_needed: bool = False,
        align_on_pad: Side | Iterable[Side] | None = "middle",
        keep_on_truncate: Side | Iterable[Side] | None = "middle",
        match_input_shape: bool = False,
    ) -> None:
        """
        Args:
            around:
                The module to skip around with residual connection

            enabled:
                Whether to enable the residual connection
                if False, it will just return the output of the inner module

            learn_skipping_weight:
                Whether to learn a weight for the skipping connection.
                By default, the weight is 1.0

            learn_residual_weight:
                Whether to learn a weight for the residual connection
                By default, the weight is 1.0

            pad_if_needed:
                Whether to pad the input to match the output shape
                if ouput is larger than input

            truncate_if_needed:
                Whether to truncate the input to match the output shape
                if output is smaller than input

            align_on_pad:
                If pad_if_needed is True, how to align the input on the output shape.

            keep_on_truncate:
                If truncate_if_needed is True, which part of the input to keep when truncating.

            match_input_shape:
                If False, output shape will be the same as residual shape
                    -> input shape will be padded/truncated to match.
                If True, output shape will be the same as input shape
                    -> residual shape will be padded/truncated to match.
        """

        super().__init__()

        if align_on_pad is None:
            align_on_pad = "middle"

        if keep_on_truncate is None:
            keep_on_truncate = "middle"

        self.around = around
        self.enabled = enabled
        self.pad_if_needed = pad_if_needed
        self.truncate_if_needed = truncate_if_needed
        self.align_on_pad = align_on_pad
        self.keep_on_truncate = keep_on_truncate
        self.match_input_shape = match_input_shape

        self.skipping_weight: torch.nn.Parameter | None = None
        self.residual_weight: torch.nn.Parameter | None = None

        if learn_skipping_weight:
            self.skipping_weight = torch.nn.Parameter(torch.tensor(1.0))

        if learn_residual_weight:
            self.residual_weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = self.around(x)
        if not self.enabled:
            return y

        if len(x.shape) != len(y.shape):
            raise IncompatibleShapesError(
                "Input and output shapes must have same number of dimensions, "
                f"got {x.shape} and {y.shape}",
                [x, y],
            )

        if not self.truncate_if_needed and any(
            dim_x > dim_y for dim_x, dim_y in zip(x.shape, y.shape)
        ):
            raise IncompatibleShapesError(
                f"Input shape {x.shape} is larger "
                f"than output shape {y.shape} and truncate_if_needed=False",
                [x, y],
            )

        if not self.pad_if_needed and any(dim_x < dim_y for dim_x, dim_y in zip(x.shape, y.shape)):
            raise IncompatibleShapesError(
                f"Input shape {x.shape} is smaller "
                f"than output shape {y.shape} and pad_if_needed=False",
                [x, y],
            )

        if self.truncate_if_needed:
            if self.match_input_shape:
                y = truncate_to_shape(y, x.shape, self.keep_on_truncate)
            else:
                x = truncate_to_shape(x, y.shape, self.keep_on_truncate)

        if self.pad_if_needed:
            if self.match_input_shape:
                y = pad_to_shape(y, x.shape, self.align_on_pad)
            else:
                x = pad_to_shape(x, y.shape, self.align_on_pad)

        if self.skipping_weight is not None:
            x = x * self.skipping_weight

        if self.residual_weight is not None:
            y = y * self.residual_weight

        return x + y

    def extra_repr(self) -> str:
        extra = []
        extra.append(f"learn_skipping_weight={self.skipping_weight is not None}")
        extra.append(f"learn_residual_weight={self.residual_weight is not None}")
        return ", ".join(extra)
