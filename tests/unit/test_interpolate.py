import cupynumeric as np
import numpy as num
import pytest
from scipy.signal import convolve2d

from pyminiweather.__main__ import get_parser
from pyminiweather.data import Fields, initialize_fields
from pyminiweather.solve import interpolate_x, interpolate_z


def test_interpolate_x():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    params = vars(args)
    nz = params["nz"]

    fields = initialize_fields(params)

    shape = fields.state.shape
    nelements = num.prod(shape)

    state_num = num.arange(nelements).astype(num.float64).reshape(shape)
    kernel_num = num.array(
        [-1.0 / 12, 7.0 / 12, 7.0 / 12, -1.0 / 12], dtype=num.float64
    )

    fields.state = np.array(state_num)
    fields.fourth_order_kernel = np.array(kernel_num)

    interpolate_x(params, fields)
    for variable in range(fields.nvariables):
        out_scipy = convolve2d(
            state_num[variable, 2 : nz + 2, :], kernel_num[np.newaxis, :], mode="same"
        )[:, 2:-1]
        assert np.allclose(out_scipy, fields.vals_x[variable])


def test_interpolate_z():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    params = vars(args)

    nx = params["nx"]

    fields = initialize_fields(params)

    shape = fields.state.shape
    nelements = num.prod(shape)

    state_num = num.arange(nelements).astype(num.float64).reshape(shape)
    kernel_num = num.array(
        [-1.0 / 12, 7.0 / 12, 7.0 / 12, -1.0 / 12], dtype=num.float64
    )

    fields.state = np.array(state_num)
    fields.fourth_order_kernel = np.array(kernel_num)

    interpolate_z(params, fields)
    for variable in range(fields.nvariables):
        out_scipy = convolve2d(
            state_num[variable, :, 2 : nx + 2], kernel_num[:, np.newaxis], mode="same"
        )[2:-1, :]
        assert np.allclose(out_scipy, fields.vals_z[variable])


def test_convolution():
    """
    Make sure cuPyNumeric convolution and scipy's convolve2D give
    the same result
    """

    nx = 12
    nz = 7
    hs = 2

    total = (nx + 2 * hs) * (nz + 2 * hs)

    state = np.arange(total).astype(np.float64).reshape(nz + 2 * hs, nx + 2 * hs)
    kernel = np.array([-1.0, 3.0, -3.0, 1.0])
    kernel_2d = kernel[:, np.newaxis]

    # Direction: z
    out_np = np.convolve(state[:, 2 : nx + 2], kernel_2d, mode="same")
    out_scipy = convolve2d(state[:, 2 : nx + 2], kernel_2d, mode="same")

    assert np.allclose(out_np, out_scipy)

    # Direction: x
    kernel_2d = kernel[np.newaxis, :]
    out_np = np.convolve(state[2 : nz + 2, :], kernel_2d, mode="same")
    out_scipy = convolve2d(state[2 : nz + 2, :], kernel_2d, mode="same")

    assert np.allclose(out_np, out_scipy)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
