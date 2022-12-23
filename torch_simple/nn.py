from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import Any, Callable, Iterable

import pydantic
import torch
from pydantic import PositiveInt
from torch import Tensor

from torch_simple.exceptions import IncompatibleShapesError
from torch_simple.functional import pad_to_shape, truncate_to_shape
from torch_simple.typedefs import ActivationLike, DropoutLike, NormLayerLike, Side
from torch_simple.utils import deep_copy_with_pickle_fallback


def get_activation(
    activation: ActivationLike,
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
        return Lambda(activation, kwargs)

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
    drop: DropoutLike,
    return_dropout_if_none: bool = False,
    kwargs: dict[str, Any] | None = None,
) -> torch.nn.Module | None:

    if kwargs is None:
        kwargs = {}

    if drop is None or drop == "none":
        if return_dropout_if_none:
            return torch.nn.Dropout(0.0, **kwargs)
        else:
            return None

    if isinstance(drop, torch.nn.Module):
        return deep_copy_with_pickle_fallback(drop)

    if isinstance(drop, float):
        return torch.nn.Dropout(drop, **kwargs)

    if callable(drop):
        return Lambda(drop, kwargs)

    raise ValueError(f"Invalid dropout: {drop}")


def get_norm_layer(
    norm_layer: NormLayerLike,
    kwargs: dict[str, Any] | None = None,
    prefer_lazy: bool = False,
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


class FeedForwardBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim: int | None,
        hidden_dim: int | None,
        output_dim: int,
        hidden_activation: ActivationLike,
        is_residual: bool,
        num_layers: int,
        use_bias: bool = True,
        output_activation: ActivationLike = None,
        norm_layer: NormLayerLike = None,
        norm_layer_kwargs: dict[str, Any] | None = None,
        dropout: DropoutLike = None,
        residual_kwargs: dict[str, Any] | None = None,
        dropout_kwargs: dict[str, Any] | None = None,
        hidden_activation_kwargs: dict[str, Any] | None = None,
        output_activation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation
        self.is_residual = is_residual
        self.num_layers = num_layers
        self.output_activation = output_activation

        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}
        if output_activation_kwargs is None:
            output_activation_kwargs = {}
        if norm_layer_kwargs is None:
            norm_layer_kwargs = {}
        if residual_kwargs is None:
            residual_kwargs = {}
        if dropout_kwargs is None:
            dropout_kwargs = {}

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        if hidden_dim is None and num_layers != 1:
            raise ValueError("hidden_dim must be specified if num_layers > 1")

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        layers: list[torch.nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            assert out_dim is not None
            if in_dim is None:
                layers.append(torch.nn.LazyLinear(out_dim, use_bias))
            else:
                layers.append(torch.nn.Linear(in_dim, out_dim, use_bias))

            if i == num_layers - 1:
                activation = get_activation(output_activation, output_activation_kwargs)
            else:
                activation = get_activation(hidden_activation, hidden_activation_kwargs)

            if activation is not None:
                layers.append(activation)

        linears = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
        assert len(linears) == num_layers

        norm = get_norm_layer(
            norm_layer,
            kwargs=norm_layer_kwargs,
            num_features=output_dim,
        )
        if norm is not None:
            layers.append(norm)

        self.nn: torch.nn.Module = torch.nn.Sequential(*layers)

        if is_residual:
            self.nn = Residual(self.nn, **residual_kwargs)

        dropout_layer = get_dropout(dropout, kwargs=dropout_kwargs)
        if dropout_layer is not None:
            if isinstance(self.nn, Residual):
                self.nn = torch.nn.Sequential(self.nn, dropout_layer)
            elif isinstance(self.nn, torch.nn.Sequential):
                self.nn.append(dropout_layer)

    def forward(self, x: Tensor) -> Tensor:
        return self.nn(x)


class FeedForwardConfig(pydantic.BaseModel):
    input_dim: PositiveInt | None = None
    hidden_dim: PositiveInt
    squeeze_dim: PositiveInt | None = None
    output_dim: PositiveInt | None = None
    num_blocks: PositiveInt
    blocks_are_residual = True
    hidden_activation = "ReLU"
    output_activation: str | None = None
    num_layers_per_block = 2
    dropout: float | None = None
    norm_layer: str | None = None
    hidden_activation_kwargs: dict[str, Any] | None = None
    output_activation_kwargs: dict[str, Any] | None = None
    residual_kwargs: dict[str, Any] | None = None
    norm_layer_kwargs: dict[str, Any] | None = None
    dropout_kwargs: dict[str, Any] | None = None
    use_bias = True
    block_name_template: str | None = None


class FeedForward(torch.nn.Module):
    @classmethod
    def from_config(cls, config: FeedForwardConfig) -> FeedForward:
        return cls(**config.dict())

    def __init__(
        self,
        input_dim: int | None,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int | None = None,
        squeeze_dim: int | None = None,
        blocks_are_residual: bool = True,
        residual_kwargs: dict[str, Any] | None = None,
        hidden_activation: ActivationLike = torch.nn.ReLU,
        hidden_activation_kwargs: dict[str, Any] | None = None,
        output_activation: ActivationLike = None,
        output_activation_kwargs: dict[str, Any] | None = None,
        num_layers_per_block: int = 2,
        dropout: DropoutLike | None = None,
        dropout_kwargs: dict[str, Any] | None = None,
        norm_layer: NormLayerLike = None,
        norm_layer_kwargs: dict[str, Any] | None = None,
        use_bias: bool = True,
        use_linear_in: bool | None = None,
        use_linear_out: bool | None = None,
        block_name_template: str | None = None,
    ) -> None:
        super().__init__()

        if squeeze_dim is None:
            squeeze_dim = hidden_dim

        if output_dim is None:
            output_dim = hidden_dim

        for arg in ["input_dim", "hidden_dim", "squeeze_dim", "output_dim"]:
            if locals()[arg] is not None and locals()[arg] < 1:
                raise ValueError(f"{arg} must be at least 1")

        if use_linear_in is None:
            use_linear_in = input_dim != hidden_dim
        if use_linear_out is None:
            use_linear_out = output_dim != hidden_dim

        blocks: OrderedDict[str, torch.nn.Module] = OrderedDict()

        if use_linear_in:
            if input_dim is None:
                blocks["linear_in"] = torch.nn.LazyLinear(hidden_dim, use_bias)
            else:
                blocks["linear_in"] = torch.nn.Linear(input_dim, hidden_dim, use_bias)

        if block_name_template is None:
            num_digits = str(max(3, len(str(num_blocks))))
            block_name_template = "ff_block_{:0" + num_digits + "d}"

        for b in range(num_blocks):
            block = FeedForwardBlock(
                input_dim=hidden_dim,
                hidden_dim=squeeze_dim,
                output_dim=hidden_dim,
                num_layers=num_layers_per_block,
                is_residual=blocks_are_residual,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                output_activation=hidden_activation,
                output_activation_kwargs=hidden_activation_kwargs,
                dropout=dropout,
                dropout_kwargs=dropout_kwargs,
                norm_layer=norm_layer,
                norm_layer_kwargs=norm_layer_kwargs,
                residual_kwargs=residual_kwargs,
                use_bias=use_bias,
            )
            block_name = block_name_template.format(b)
            blocks[block_name] = block

        if use_linear_out:
            blocks["linear_out"] = torch.nn.Linear(hidden_dim, output_dim, use_bias)

        activation_out = get_activation(output_activation, output_activation_kwargs)
        if activation_out is not None:
            blocks["activation_out"] = activation_out

        self.nn = torch.nn.Sequential(blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.nn(x)
