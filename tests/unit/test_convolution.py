import cunumeric as np
import pytest
from scipy.signal import convolve2d


def test_convolution():
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
