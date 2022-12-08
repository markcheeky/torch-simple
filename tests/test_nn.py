import itertools
import typing

import pytest
import torch
from torch import Tensor

from torch_simple.exceptions import IncompatibleShapesError
from torch_simple.functional import Side, truncate_to_shape
from torch_simple.nn import Constant, Lambda, Residual


class TestConstant:
    def test_layer_outputs_correctly(self) -> None:
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

    def test_layer_is_trainable(self) -> None:
        orig_value = torch.ones(1, 2, 3)
        layer = Constant(orig_value, trainable=True)
        assert sum(param.requires_grad for param in layer.parameters()) == 1

        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        output: Tensor = layer()
        output.sum().backward()
        assert layer.value.grad is not None
        optimizer.step()
        assert not torch.allclose(layer.value, orig_value)

    def test_layer_is_not_trainable(self) -> None:
        layer = Constant(torch.ones(1, 2, 3), trainable=False)
        assert sum(param.requires_grad for param in layer.parameters()) == 0


class TestResidual:
    def test_layer_outputs_correctly(self) -> None:

        model_lin_input = (2, 10)
        model_lin = torch.nn.Linear(10, 10)

        model_ff_input = (4, 10)
        model_ff = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.Tanh(),
        )

        model_conv_input = (4, 5, 10, 10)
        model_conv = torch.nn.Sequential(
            torch.nn.Conv2d(5, 5, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(5, 5, (3, 3), padding=1),
            torch.nn.ReLU(),
        )

        model_gru_input = (15, 4, 8)
        model_gru = torch.nn.Sequential(
            torch.nn.GRU(
                input_size=8,
                hidden_size=8,
                num_layers=4,
                batch_first=False,
            ),
            Lambda(lambda out_and_h: out_and_h[0]),
        )

        model_transformer_input = (15, 4, 64)
        model_transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
            ),
            num_layers=3,
        )

        for model_input, model in [
            (model_lin_input, model_lin),
            (model_ff_input, model_ff),
            (model_conv_input, model_conv),
            (model_gru_input, model_gru),
            (model_transformer_input, model_transformer),
        ]:
            x = torch.randn(model_input)
            res_model = Residual(model).eval()
            assert torch.allclose(res_model(x), x + model(x))

            res_model = Residual(model, enabled=False).eval()
            assert torch.allclose(res_model(x), model(x))

    def test_layer_with_padding(self) -> None:
        model_lin_input = (2, 10)
        model_lin = torch.nn.Linear(10, 20)

        model_ff_input = (4, 10)
        model_ff = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 20),
            torch.nn.Tanh(),
        )

        model_conv_input = (4, 5, 10, 10)
        model_conv = torch.nn.Sequential(
            torch.nn.Conv2d(5, 5, (3, 3), padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(5, 5, (3, 3), padding=4),
            torch.nn.ReLU(),
        )

        model_gru_input = (15, 4, 8)
        model_gru = torch.nn.Sequential(
            torch.nn.GRU(
                input_size=8,
                hidden_size=20,
                num_layers=4,
                batch_first=False,
            ),
            Lambda(lambda out_and_h: out_and_h[0]),
        )

        model_transformer_input = (15, 4, 32)
        model_transformer = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=64,
                    nhead=4,
                    dim_feedforward=128,
                ),
                num_layers=3,
            ),
        )

        for model_input, model in [
            (model_lin_input, model_lin),
            (model_ff_input, model_ff),
            (model_conv_input, model_conv),
            (model_gru_input, model_gru),
            (model_transformer_input, model_transformer),
        ]:
            assert isinstance(model, torch.nn.Module)
            model = model.eval()
            x = torch.randn(model_input)

            with pytest.raises(IncompatibleShapesError):
                res_model = Residual(model).eval()
                res_model(x)

            for side in typing.get_args(Side):
                res_model = Residual(model, pad_if_needed=True, align_on_pad=side).eval()
                assert torch.allclose(
                    truncate_to_shape(res_model(x), x.shape, keep_side=side),
                    x + truncate_to_shape(model(x), x.shape, keep_side=side),
                )

    def test_layer_with_truncation(self) -> None:

        model_lin_input = (2, 10)
        model_lin = torch.nn.Linear(10, 5)

        model_ff_input = (4, 10)
        model_ff = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 5),
            torch.nn.Tanh(),
        )

        model_conv_input = (4, 5, 10, 10)
        model_conv = torch.nn.Sequential(
            torch.nn.Conv2d(5, 5, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(5, 5, (3, 3)),
            torch.nn.ReLU(),
        )

        model_gru_input = (15, 4, 20)
        model_gru = torch.nn.Sequential(
            torch.nn.GRU(
                input_size=20,
                hidden_size=8,
                num_layers=4,
                batch_first=False,
            ),
            Lambda(lambda out_and_h: out_and_h[0]),
        )

        model_transformer_input = (15, 4, 64)
        model_transformer = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=32,
                    nhead=4,
                    dim_feedforward=128,
                ),
                num_layers=3,
            ),
        )

        for model_input, model in [
            (model_lin_input, model_lin),
            (model_ff_input, model_ff),
            (model_conv_input, model_conv),
            (model_gru_input, model_gru),
            (model_transformer_input, model_transformer),
        ]:
            assert isinstance(model, torch.nn.Module)
            model = model.eval()
            x = torch.randn(model_input)

            with pytest.raises(IncompatibleShapesError):
                res_model = Residual(model).eval()
                res_model(x)

            for side in typing.get_args(Side):
                res_model = Residual(model, truncate_if_needed=True, keep_on_truncate=side).eval()
                res_pred = res_model(x)
                expected = model(x) + truncate_to_shape(x, res_pred.shape, keep_side=side)

                assert torch.allclose(res_pred, expected)
