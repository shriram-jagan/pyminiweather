import os
from enum import IntEnum
from typing import Tuple

if "LEGATE_MAX_DIM" in os.environ and "LEGATE_MAX_FIELDS" in os.environ:
    import cunumeric as numpy
    from cunumeric import convolve as convolve
else:
    import numpy as numpy
    from scipy.signal import convolve as convolve

print(f"Imported numpy backend: {numpy.__name__}")


class IDS(IntEnum):
    DENS = 0
    UMOM = 1
    WMOM = 2
    RHOT = 3
