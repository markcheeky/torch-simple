import itertools

import torch
from torch import Tensor

from torch_simple.nn import Constant


class TestConstant:
    def test_layer_outputs_correctly(self):
        values = [
            torch.zeros(0),
            torch.zeros(1),
            torch.ones(1, 2, 3),
            torch.randn(4, 5, 6, 7),
        ]

        argss = [
            (),
            (1, 2),
            (torch.ones(23) * 5,),
        ]

        kwargss = [
            {},
            {"hello": "world"},
            {"a": torch.zeros(0), "b": [1, 2]},
            {"a": torch.ones(1, 2, 3), "b": torch.randn(4, 5, 6, 7)},
        ]

        for value, args, kwargs in itertools.product(values, argss, kwargss):
            assert isinstance(kwargs, dict)
            for trainable in [True, False]:
                layer = Constant(value, trainable=trainable)
                assert torch.allclose(layer(*args, **kwargs), value)

    def test_layer_is_trainable(self):
        orig_value = torch.ones(1, 2, 3)
        layer = Constant(orig_value, trainable=True)
        assert sum(param.requires_grad for param in layer.parameters()) == 1

        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        output: Tensor = layer()
        output.sum().backward()
        assert layer.value.grad is not None
        optimizer.step()
        assert not torch.allclose(layer.value, orig_value)

    def test_layer_is_not_trainable(self):
        layer = Constant(torch.ones(1, 2, 3), trainable=False)
        assert sum(param.requires_grad for param in layer.parameters()) == 0
