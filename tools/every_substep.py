#!/usr/bin/env python
# coding: utf-8

# # Compare results between scipy and legate

# When we run pyminiweather on multiple CPUs or GPUs, we get NaNs in the simulation. This creeps up due to result difference between scipy's convolution and legate's convolution implementation (specifically, cuNumeric).
# 
# Steps to reproduce:
# 1. Execute run_cpu.sh script as is.
# 2. Run this notebook after the simulation completes.
# 3. Look at the plots below that compare the output from scipy and legate after a convolution operation. In most of the plots below, notice the sharp gradient at x ~ 150:200 and y ~ 100 (indices). This may or may not coincide with the partition interface This difference in results, which was visualized earlier, happens right after the first call to the function `interpolate_z` in pyminiweather/solve.interpolate.py. If we dump the data and read it back and try to reproduce it, we don't see the failure.
# 4. (optional) Change N_CPUS to 1 in `run_cpus.sh` and run the script, and then run this notebook, to observe that there are no artifacts in and around the partition interface
# 
# Possible causes of the bug:
# - The inputs to the convolution API are a slice of 3D array and another transformed array (kernel).
# - The GPU variant of the convolution implementation could be buggy for these type of input shapes, esp., the kernel

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
import sys
from typing import List, Any
from pathlib import Path


# In[2]:


np.set_printoptions(threshold=sys.maxsize)
cwd = Path("./")


# In[3]:


nvariables = 4
nz = 200
nx = 400
hs = 2


# In[4]:


save_fig: bool = False


# In[5]:


kernel = np.array([-1.0 / 12, 7.0 / 12, 7.0 / 12, -1.0 / 12], dtype=np.float64)


# In[6]:


# step_number -> direction
directions = {}


# In[7]:


def count_directions(tmp_files: List[str], in_direction: str):
    return sum([in_direction in f.as_posix() for f in tmp_files])


# ### Read state_forcing files

# In[8]:


tmp_files = sorted([file_path for file_path in cwd.glob("*state_forcing.txt")])

n_substeps = len(tmp_files)
state_forcing = np.zeros((n_substeps, nvariables, nz + 2*hs, nx + 2*hs))

for file in tmp_files:
    step, direction, _, _ = file.as_posix().split(".")
    
    step = int(step)
    directions[step] = direction

    # read all files
    state_forcing[step,...] = np.loadtxt(file).reshape(nvariables, nz + 2*hs, nx + 2*hs)


# ### Read vals files

# In[9]:


tmp_files = sorted([file_path for file_path in cwd.glob("*vals.txt")])

# note that the step numbers for vals is local while that of state is global.
# compute offsets to handle the difference in how they are dumped.
# since the bug is reproduced right after the first discrete step, this is 
# not essential for this work

n_substeps = len(tmp_files)

nvals_x = count_directions(tmp_files, ".x.")
nvals_z = count_directions(tmp_files, ".z.")

shape_z = (nvariables, nz + 1, nx)
shape_x = (nvariables, nz, nx + 1)

assert nvals_x + nvals_z == n_substeps

vals_z = np.zeros((nvals_z, *shape_z))
vals_x = np.zeros((nvals_x, *shape_x))

step_z = 0
step_x = 0

for file in tmp_files:
    _, direction, _, _ = file.as_posix().split(".")
    
    if direction == "z":
        vals_z[step_z,...] = np.loadtxt(file).reshape(shape_z)
        step_z += 1
    elif direction == "x":
        vals_x[step_x,...] = np.loadtxt(file).reshape(shape_x)
        step_x += 1
    

step_number = 0
variable_idx = 3 # rhoT


# we know that the first substep causes differences
assert step_number == 0

output = convolve(
    state_forcing[step_number][:, :, 2 : nx + 2], 
    kernel[np.newaxis, :, np.newaxis], 
    mode="same",)[:, 2:-1, :]


assert np.allclose(output[variable_idx], vals_z[step_number][variable_idx]) 

