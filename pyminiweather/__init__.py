import os
from typing import Tuple

if "LEGATE_MAX_DIM" in os.environ and "LEGATE_MAX_FIELDS" in os.environ:
    import cunumeric as numpy
    from cunumeric import convolve as convolve  # noqa: F401
else:
    import numpy as numpy
    from scipy.signal import convolve2d as convolve  # noqa: F401

print(f"Imported numpy backend: {numpy.__name__}")

# use an enum
ID_DENS = 0
ID_UMOM = 1
ID_WMOM = 2
ID_RHOT = 3


def meshgrid(x: numpy.ndarray, y: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """This is equivalent to numpy's meshgrid for two one-dimensional
    input arrays without all the bells and whistles.

    Parameters:
    ----------
    x: numpy.ndarray
        1D array denoting x-coordinates
    y: numpy.ndarray
        1D array denoting y-coordinates

    Returns:
    -------
    A tuple of two-dimensional arrays that correspond to (x, y) pairs
    in a cartesian coordinate system
    """
    if numpy.__name__ == "numpy":
        return numpy.meshgrid(x, y)
    elif numpy.__name__ == "cunumeric":
        assert x.ndim == y.ndim == 1

        return (numpy.tile(x, (y.size, 1)), numpy.tile(y, (x.size, 1)).T)
    else:
        raise ValueError("Unknown backend")
