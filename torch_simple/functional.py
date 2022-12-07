import itertools
import typing
from typing import Any, Iterable, Literal

import torch
from torch import Tensor

from torch_simple.exceptions import IncompatibleShapesError

Side = Literal["start", "end", "middle"]


def _check_pad_truncate_args_valid(
    x: Tensor,
    shape: tuple[int] | torch.Size,
    side: Side | Iterable[Side],
) -> None:
    sides = typing.get_args(Side)

    if isinstance(side, str):
        if side not in sides:
            raise ValueError(f"side must be one of {sides}, got {side}")
    else:
        side = tuple(side)
        if any(s not in sides for s in side):
            raise ValueError(f"all values in side must be one of {sides}, got {side}")

        if len(side) != len(shape):
            raise ValueError(f"side and shape must have same length, got {side} and {shape}")

    if len(x.shape) != len(shape):
        raise ValueError(
            "input and output shapes must have same number of dimensions, "
            f"got {x.shape} and {shape}"
        )


def pad_to_shape(
    x: Tensor,
    shape: tuple[int] | torch.Size,
    align: Side | Iterable[Side] = "start",
    raise_if_larger: bool = True,
    **kwargs: Any,
) -> Tensor:
    """
    Pads x to match a desired shape.
    """

    _check_pad_truncate_args_valid(x, shape, align)

    if raise_if_larger and any(dim_x > dim_y for dim_x, dim_y in zip(x.shape, shape)):
        raise IncompatibleShapesError(
            f"Input shape {x.shape} has a dimension larger than "
            f"output shape {shape} and raise_if_larger=True",
            [x],
        )

    if isinstance(align, str):
        align = (align,) * len(shape)

    padding = []
    for dim_x, dim_y, side in zip(x.shape, shape, align):
        if dim_y > dim_x:
            padding_size = dim_y - dim_x
            if side == "start":
                padding.append((0, padding_size))
            elif side == "end":
                padding.append((padding_size, 0))
            elif side == "middle":
                padding.append((padding_size // 2, padding_size - padding_size // 2))
        else:
            padding.append((0, 0))

    output_shape = tuple(max(dim_x, dim_y) for dim_x, dim_y in zip(x.shape, shape))
    padding_spec = tuple(itertools.chain(*reversed(padding)))

    padded = torch.nn.functional.pad(x, pad=padding_spec, **kwargs)
    assert padded.shape == output_shape
    return padded


def truncate_to_shape(
    x: Tensor,
    shape: tuple[int] | torch.Size,
    keep_side: Side | Iterable[Side] = "start",
    raise_if_smaller: bool = True,
) -> Tensor:
    """
    Truncates x to match a desired shape.
    """

    _check_pad_truncate_args_valid(x, shape, keep_side)

    if raise_if_smaller and any(dim_x < dim_y for dim_x, dim_y in zip(x.shape, shape)):
        raise IncompatibleShapesError(
            f"Input shape {x.shape} has a dimension smaller than "
            f"output shape {shape} and raise_if_smaller=True",
            [x],
        )

    if isinstance(keep_side, str):
        keep_side = (keep_side,) * len(shape)

    output_shape = tuple(min(dim_x, dim_y) for dim_x, dim_y in zip(x.shape, shape))
    slices = []
    for dim_x, dim_y, side in zip(x.shape, shape, keep_side):
        if dim_y < dim_x:
            if side == "start":
                slices.append(slice(0, dim_y))
            elif side == "end":
                slices.append(slice(dim_x - dim_y, dim_x))
            elif side == "middle":
                slices.append(slice((dim_x - dim_y) // 2, (dim_x + dim_y) // 2))
        else:
            slices.append(slice(0, dim_y))

    if len(x.shape) == 0:
        truncated = x
    else:
        truncated = x[slices]
    assert truncated.shape == output_shape
    return truncated
