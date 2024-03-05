import cunumeric as np
import numpy as num
import pytest

from pyminiweather.__main__ import get_parser
from pyminiweather.data import Fields, initialize_fields
from pyminiweather.solve import interpolate_x, interpolate_z


def test_interpolate_z():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    params = vars(args)

    fields = initialize_fields(params)

    shape = fields.state.shape
    nelements = np.prod(shape)

    # initialize data array and kernel
    fields.state = np.arange(nelements).astype(np.float64).reshape(shape)
    fields.fourth_order_kernel = np.array(
        [-1.0 / 12, 7.0 / 12, 7.0 / 12, -1.0 / 12], dtype=np.float64
    )

    # TODO
    print(f"Test invalid", flush=True)
    interpolate_z(params, fields, fields.state)

    assert True


def test_interpolate_x():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    params = vars(args)

    fields = initialize_fields(params)

    shape = fields.state.shape
    nelements = np.prod(shape)

    # initialize data array and kernel
    fields.state = np.arange(nelements).astype(np.float64).reshape(shape)
    fields.fourth_order_kernel = np.array(
        [-1.0 / 12, 7.0 / 12, 7.0 / 12, -1.0 / 12], dtype=np.float64
    )

    print(f"Test invalid", flush=True)
    interpolate_x(params, fields, fields.state)

    # TODO
    assert True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
