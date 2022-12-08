import random
import typing

import pytest
import torch

from torch_simple.exceptions import IncompatibleShapesError
from torch_simple.functional import Side, pad_to_shape, truncate_to_shape

TEST_SHAPES = [
    (1,),
    (2,),
    (1, 2),
    (4, 5),
    (1, 2, 3),
    (2, 5, 4),
    (3, 4, 5),
    (1, 2, 3, 4),
    (2, 3, 4, 5),
    (3, 4, 5, 6),
]


def _get_incompatible_shapes(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    """
    Returns a list of shapes that are incompatible with the given shape
    (i.e. have a different number of dimensions)
    """

    return [
        shape + (1,),
        shape + (2, 1),
        shape + (2, 1, 3),
        (1,) + shape,
        (2, 1) + shape,
        (2, 1, 3) + shape,
        shape[1:],
        shape[2:],
        shape[3:],
        shape[:-1],
    ]


class TestPadToShape:
    def test_pad_to_shape_basic(self):
        x = torch.rand(2, 3)
        y_shape = (5, 6)

        y = pad_to_shape(x, y_shape, "start")
        assert y.shape == y_shape
        assert torch.allclose(x, y[:2, :3])

        y = pad_to_shape(x, y_shape, "end")
        assert y.shape == y_shape
        assert torch.allclose(x, y[3:, 3:])

        y = pad_to_shape(x, y_shape, "middle")
        assert y.shape == y_shape
        assert torch.allclose(x, y[1:3, 1:4])

    def test_pad_to_shape_random(self):
        random_gen_torch = torch.Generator().manual_seed(1)
        random_gen = random.Random(2)

        for x_shape in TEST_SHAPES:
            for _ in range(10):
                x = torch.rand(x_shape, generator=random_gen_torch)
                padding_value = random_gen.random()

                # padding tensor to the same shape should not change anything
                for align in typing.get_args(Side):
                    y = pad_to_shape(x, x.shape, align=align, value=padding_value)
                    assert x.shape == y.shape
                    assert torch.allclose(x, y)

                # randomized y shape
                y_shape = tuple(x_dim + random_gen.randint(0, 3) for x_dim in x_shape)

                slices_start = tuple(slice(0, x_dim) for x_dim in x_shape)
                slices_end = tuple(
                    slice(y_dim - x_dim, y_dim) for x_dim, y_dim in zip(x_shape, y_shape)
                )
                slices_middle = tuple(
                    slice((y_dim - x_dim) // 2, (y_dim + x_dim) // 2)
                    for x_dim, y_dim in zip(x_shape, y_shape)
                )

                slicing = {
                    "start": slices_start,
                    "end": slices_end,
                    "middle": slices_middle,
                }

                for align in typing.get_args(Side):
                    slices = slicing[align]
                    y = pad_to_shape(x, y_shape, align=align, value=padding_value)

                    # Check that the output shape is correct
                    assert y.shape == y_shape

                    # Check that the input is still there
                    assert torch.allclose(x, y[slices])

                    # Check that the rest is filled with the padding value
                    y[slices] = padding_value
                    assert torch.allclose(y, torch.ones_like(y) * padding_value)

    def test_pad_to_shape_backward(self):
        for align in typing.get_args(Side):
            random_gen_torch = torch.Generator().manual_seed(2)
            x = torch.rand(2, 3, requires_grad=True, generator=random_gen_torch)
            y_shape = (5, 6)

            assert x.grad is None
            y = pad_to_shape(x, y_shape, align=align)
            y.sum().backward()
            assert x.grad is not None
            assert torch.isclose(x.grad, torch.ones_like(x)).all()

    def test_pad_to_shape_incompatible_shape_size(self):
        for shape in TEST_SHAPES:
            x = torch.ones(shape)
            bad_y_shapes = _get_incompatible_shapes(shape)

            for bad_y_shape in bad_y_shapes:
                with pytest.raises(IncompatibleShapesError):
                    pad_to_shape(x, bad_y_shape)

    def test_pad_to_shape_strict(self):
        random_gen = random.Random(3)

        for shape in TEST_SHAPES:
            x = torch.ones(shape)

            for _ in range(10):
                y_shape = tuple(random_gen.randint(1, x_dim) for x_dim in shape)
                if y_shape == shape:
                    continue

                with pytest.raises(IncompatibleShapesError):
                    pad_to_shape(x, y_shape, strict=True)

                y = pad_to_shape(x, y_shape, strict=False)
                assert y.shape == tuple(max(x_dim, y_dim) for x_dim, y_dim in zip(shape, y_shape))


class TestTruncateToShape:
    def test_truncate_to_shape_basic(self):
        x = torch.rand(5, 6)
        y_shape = (2, 3)

        y = truncate_to_shape(x, y_shape, "start")
        assert y.shape == y_shape
        assert torch.allclose(x[:2, :3], y)

        y = truncate_to_shape(x, y_shape, "end")
        assert y.shape == y_shape
        assert torch.allclose(x[3:, 3:], y)

        y = truncate_to_shape(x, y_shape, "middle")
        assert y.shape == y_shape
        assert torch.allclose(x[1:3, 1:4], y)

    def test_truncate_to_shape_random(self):
        random_gen_torch = torch.Generator().manual_seed(4)
        random_gen = random.Random(5)

        for x_shape in TEST_SHAPES:
            for _ in range(10):
                x = torch.rand(x_shape, generator=random_gen_torch)

                # truncating tensor to the same shape should not change anything
                for keep_side in typing.get_args(Side):
                    y = truncate_to_shape(x, x.shape, keep_side=keep_side)
                    assert x.shape == y.shape
                    assert torch.allclose(x, y)

                # randomized y shape
                y_shape = tuple(max(x_dim - random_gen.randint(0, 3), 1) for x_dim in x_shape)

                slices_start = tuple(slice(0, y_dim) for y_dim in y_shape)
                slices_end = tuple(
                    slice(x_dim - y_dim, x_dim) for x_dim, y_dim in zip(x_shape, y_shape)
                )
                slices_middle = tuple(
                    slice((x_dim - y_dim) // 2, (x_dim + y_dim) // 2)
                    for x_dim, y_dim in zip(x_shape, y_shape)
                )

                slicing = {
                    "start": slices_start,
                    "end": slices_end,
                    "middle": slices_middle,
                }

                for keep_side in typing.get_args(Side):
                    slices = slicing[keep_side]
                    y = truncate_to_shape(x, y_shape, keep_side=keep_side)

                    # Check that the output shape is correct
                    assert y.shape == y_shape

                    # Check that the input is still there
                    assert torch.allclose(x[slices], y)

    def test_truncate_to_shape_backward(self):
        for keep_side in typing.get_args(Side):
            random_gen_torch = torch.Generator().manual_seed(6)
            x = torch.rand(3, 5, requires_grad=True, generator=random_gen_torch)
            y_shape = (2, 2)
            if keep_side == "start":
                slices = (slice(0, 2), slice(0, 2))
            elif keep_side == "end":
                slices = (slice(1, 3), slice(3, 5))
            elif keep_side == "middle":
                slices = (slice(0, 2), slice(1, 3))
            else:
                raise ValueError(f"untested keep_side: {keep_side}")

            assert x.grad is None
            y = truncate_to_shape(x, y_shape, keep_side=keep_side)
            y.sum().backward()
            assert x.grad is not None
            assert torch.isclose(x.grad[slices], torch.ones_like(x.grad[slices])).all()
            x.grad[slices] = 0
            assert torch.isclose(x.grad, torch.zeros_like(x)).all()

    def test_truncate_to_shape_incompatible_shape_size(self):
        for shape in TEST_SHAPES:
            x = torch.ones(shape)
            bad_y_shapes = _get_incompatible_shapes(shape)

            for bad_y_shape in bad_y_shapes:
                with pytest.raises(IncompatibleShapesError):
                    truncate_to_shape(x, bad_y_shape)

    def test_truncate_to_shape_strict(self):
        random_gen = random.Random(7)

        for shape in TEST_SHAPES:
            x = torch.ones(shape)

            for _ in range(10):
                y_shape = tuple(random_gen.randint(x_dim, x_dim + 4) for x_dim in shape)
                if y_shape == shape:
                    continue

                with pytest.raises(IncompatibleShapesError):
                    truncate_to_shape(x, y_shape, strict=True)

                y = truncate_to_shape(x, y_shape, strict=False)
                assert y.shape == tuple(min(x_dim, y_dim) for x_dim, y_dim in zip(shape, y_shape))
