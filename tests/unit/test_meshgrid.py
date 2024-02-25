import cunumeric as np
import numpy as num
import pytest

from pyminiweather import meshgrid


def test_meshgrid():
    hs = 2
    dx = 1.0
    dz = 1.0
    nx = 12
    nz = 8

    x = num.linspace(-hs * dx, (nx + hs) * dx, nx + 2 * hs, endpoint=False)
    z = num.linspace(-hs * dz, (nz + hs) * dz, nz + 2 * hs, endpoint=False)

    x_np = np.array(x)
    z_np = np.array(z)

    x_np, z_np = meshgrid(x_np, z_np)
    x_num, z_num = num.meshgrid(x, z)

    assert num.allclose(x_np, x_num)
    assert num.allclose(z_np, z_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
