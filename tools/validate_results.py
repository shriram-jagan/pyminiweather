#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import signal 
import sys
from typing import List, Any
from pathlib import Path

np.set_printoptions(threshold=sys.maxsize)
cwd = Path("./")

nvariables = 4
nz = 200
nx = 400
hs = 2
variable_idx = 3 # rhoT
save_fig: bool = False
kernel = np.array([-1.0 / 12, 7.0 / 12, 7.0 / 12, -1.0 / 12], dtype=np.float64)

# This is the input to convolution
infile = "00.z.state_forcing.txt"
shape_state = (nvariables, nz + 2*hs, nx + 2*hs)
state_forcing = np.loadtxt(infile).reshape(shape_state)


# This is the output of the convolution from legate (on 4 CPUs)
infile = "00.z.vals.txt"
shape_z = (nvariables, nz + 1, nx)
vals_z = np.loadtxt(infile).reshape(shape_z)

# Do a scipy convolve
output = signal.convolve(
    state_forcing[:, :, 2 : nx + 2], 
    kernel[np.newaxis, :, np.newaxis], 
    mode="same",)[:, 2:-1, :]


assert np.allclose(output[variable_idx], vals_z[variable_idx]) 

