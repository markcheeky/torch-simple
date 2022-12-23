from __future__ import annotations

from torch import Tensor


class IncompatibleShapesError(ValueError):
    def __init__(self, message: str, tensors: list[Tensor]) -> None:
        self.tensors = tensors
        super().__init__(message)
