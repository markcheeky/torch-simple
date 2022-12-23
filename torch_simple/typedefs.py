from __future__ import annotations

from typing import Callable, Literal, Type, TypeAlias

import torch
from torch import Tensor

Side = Literal["start", "end", "middle"]

ActivationLike: TypeAlias = (
    str | torch.nn.Module | Callable[[Tensor], Tensor] | Type | Literal["none"] | None
)

DropoutLike: TypeAlias = (
    float | torch.nn.Module | Callable[[Tensor], Tensor] | Type | Literal["none"] | None
)

NormLayerLike: TypeAlias = (
    str | torch.nn.Module | Callable[[Tensor], Tensor] | Type | Literal["none"] | None
)
