from dataclasses import dataclass
from typing import Tuple

from pyminiweather import numpy as np


@dataclass
class Fields:
    # sizes
    nx: int
    nz: int
    hs: int
    nvariables: int
    s: int

    # persistent variables
    state: np.ndarray
    state_tmp: np.ndarray
    flux: np.ndarray
    tend: np.ndarray

    # Intermediate variables that are stored
    vals_x: np.ndarray
    d3_vals_x: np.ndarray
    vals_z: np.ndarray
    d3_vals_z: np.ndarray

    # NOTE that the hydrostatic fields below don't change with x,
    # so we use 1D arrays to store them and promote the 1D array as needed
    # to account for the entire spatial domain.

    # Hydrostatic density computed at vertical cell averages of shape (1-hs: nz+hs)
    hy_dens_cell: np.ndarray

    # Hydrostatic potential temperature computed at vertical cell averages of shape (1-hs: nz+hs)
    hy_dens_theta_cell: np.ndarray

    # Hydrostatic density computed at vertical cell interface of shape (1s: nz+1)
    hy_dens_int: np.ndarray

    # Hydrostatic pressure computed at vertical cell interface of shape (1s: nz+1)
    hy_pressure_int: np.ndarray

    # Hydrostatic potential temperature computed at vertical cell interface of shape (1s: nz+1)
    hy_dens_theta_int: np.ndarray

    # kernel related
    fourth_order_kernel: np.ndarray
    first_order_kernel: np.ndarray

    shape: Tuple


def initialize_fields(params):
    nx = params["nx"]
    nz = params["nz"]
    hs = params["hs"]
    s = params["s"]
    nvariables = 4

    assert hs * 2 - s == 0

    shape = (nvariables, nz + 2 * hs, nx + 2 * hs)

    # flux, tendency and state
    flux = np.zeros((nvariables, nz + 1, nx + 1))
    tend = np.zeros((nvariables, nz, nx))
    state = np.zeros(shape).astype(np.float64)

    # temp arrays in x and z for storing the interpolated values
    vals_x = np.zeros((nvariables, nz, nx + 1))
    d3_vals_x = np.zeros((nvariables, nz, nx + 1))
    vals_z = np.zeros((nvariables, nz + 1, nx))
    d3_vals_z = np.zeros((nvariables, nz + 1, nx))

    # All hydrostatic quantities go here
    hy_dens_cell = np.zeros(nz + 2 * hs)
    hy_dens_theta_cell = np.zeros(nz + 2 * hs)
    hy_dens_int = np.zeros(nz + 1)
    hy_dens_theta_int = np.zeros(nz + 1)
    hy_pressure_int = np.zeros(nz + 1)

    state_tmp = state.copy()

    # interpolating kernels
    # NOTE: The interpolation is done by means of a convolution operation
    # which means that the weights in the kernel have to be reversed.
    # Recall (f * g) (t): integral_{f(x) * g(t-x) dx}
    # tand note that g(t-x) part is where the kerel gets "flipped"
    fourth_order_kernel = np.array(
        [-1.0 / 12, 7.0 / 12, 7.0 / 12, -1.0 / 12], dtype=np.float64
    )
    first_order_kernel = np.array([1.0, -3.0, 3.0, -1.0], dtype=np.float64)

    initialized_fields = Fields(
        nx=nx,
        nz=nz,
        hs=hs,
        nvariables=nvariables,
        s=s,
        state=state,
        state_tmp=state_tmp,
        hy_dens_cell=hy_dens_cell,
        hy_dens_theta_cell=hy_dens_theta_cell,
        hy_dens_int=hy_dens_int,
        hy_pressure_int=hy_pressure_int,
        hy_dens_theta_int=hy_dens_theta_int,
        vals_x=vals_x,
        vals_z=vals_z,
        d3_vals_x=d3_vals_x,
        d3_vals_z=d3_vals_z,
        flux=flux,
        tend=tend,
        fourth_order_kernel=fourth_order_kernel,
        first_order_kernel=first_order_kernel,
        shape=shape,
    )

    return initialized_fields
