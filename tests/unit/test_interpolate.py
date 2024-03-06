import cunumeric as np
import numpy as num
import pytest
from scipy.signal import convolve2d as convolve

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
        out_scipy = convolve(
            state_num[variable, 2 : nz + 2, :], kernel_num[np.newaxis, :], mode="same"
        )[:, 2:-1]
        print(fields.vals_x[variable])
        print(out_scipy)
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
        out_scipy = convolve(
            state_num[variable, :, 2 : nx + 2], kernel_num[:, np.newaxis], mode="same"
        )[2:-1, :]
        assert np.allclose(out_scipy, fields.vals_z[variable])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
