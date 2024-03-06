from typing import Dict

from pyminiweather import IDS
from pyminiweather import numpy as np
from pyminiweather.data import Constants, Fields
from pyminiweather.ics import Directions
from pyminiweather import convolve


def interpolate_x(params: Dict, fields: Fields, state: np.ndarray = None):
    """
    Computes the intermediate quantities that are used in the
    computation of flux. These quantities represent the third
    and first order interpolation of conserved quantities and
    the stabilization term.

    Parameters:
    -----------
    params: Dict
        Dictionary with simulation parameters

    fields: Fields
        A dataclass that contains the simulation variables

    state: np.ndarray
        An array with conservative variables at the current timesteps

    """
    nz = params["nz"]

    state = fields.state if state is None else state

    for ivar in range(fields.nvariables):
        fields.vals_x[ivar, ...] = convolve(
            state[ivar, 2 : nz + 2, :],
            fields.fourth_order_kernel[np.newaxis, :],
            mode="same",
        )[:, 2:-1]

        fields.d3_vals_x[ivar, ...] = convolve(
            state[ivar, 2 : nz + 2, :],
            fields.first_order_kernel[np.newaxis, :],
            mode="same",
        )[:, 2:-1]


def interpolate_z(params: Dict, fields: Fields, state: np.ndarray = None):
    """
    Computes the intermediate quantities that are used in the
    computation of flux. These quantities represent the third
    and first order interpolation of conserved quantities and
    the stabilization term.

    Parameters:
    -----------
    params: Dict
        Dictionary with simulation parameters

    fields: Fields
        A dataclass that contains the simulation variables

    state: np.ndarray
        An array with conservative variables at the current timesteps

    """
    nx = params["nx"]

    state = fields.state if state is None else state

    for ivar in range(fields.nvariables):
        fields.vals_z[ivar, ...] = convolve(
            state[ivar, :, 2 : nx + 2],
            fields.fourth_order_kernel[:, np.newaxis],
            mode="same",
        )[2:-1, :]

        fields.d3_vals_z[ivar, ...] = convolve(
            state[ivar, :, 2 : nx + 2],
            fields.first_order_kernel[:, np.newaxis],
            mode="same",
        )[2:-1, :]


def compute_flux_x(params: Dict, fields: Fields):
    """
    Computes the intermediate quantities that are used in the
    computation of flux. These quantities represent the third
    and first order interpolation of conserved quantities and
    the stabilization term.

    Parameters:
    -----------
    params: Dict
        Dictionary with simulation parameters

    fields: Fields
        A dataclass that contains the simulation variables
    """
    nx = params["nx"]
    nz = params["nz"]
    hs = params["hs"]
    dx = params["dx"]
    dt = params["dt"]

    hv_coeff = -Constants.hv_beta.value * dx / (16 * dt)
    gamma = Constants.gamma.value
    C0 = Constants.C0.value

    rho = (
        fields.vals_x[IDS.DENS, 0:nz, 0 : nx + 1]
        + fields.hy_dens_cell[hs : nz + hs, np.newaxis]
    )
    u = fields.vals_x[IDS.UMOM, 0:nz, 0 : nx + 1] / rho
    w = fields.vals_x[IDS.WMOM, 0:nz, 0 : nx + 1] / rho
    t = (
        fields.vals_x[IDS.RHOT, 0:nz, 0 : nx + 1]
        + fields.hy_dens_theta_cell[hs : nz + hs, np.newaxis]
    ) / rho

    pressure = C0 * np.power(rho * t, gamma)

    fields.flux[IDS.DENS, 0:nz, 0 : nx + 1] = (
        rho * u - hv_coeff * fields.d3_vals_x[IDS.DENS]
    )
    fields.flux[IDS.UMOM, 0:nz, 0 : nx + 1] = (
        rho * u**2 + pressure - hv_coeff * fields.d3_vals_x[IDS.UMOM]
    )
    fields.flux[IDS.WMOM, 0:nz, 0 : nx + 1] = (
        rho * u * w - hv_coeff * fields.d3_vals_x[IDS.WMOM]
    )
    fields.flux[IDS.RHOT, 0:nz, 0 : nx + 1] = (
        rho * u * t - hv_coeff * fields.d3_vals_x[IDS.RHOT]
    )


def compute_flux_z(params: Dict, fields: Fields):
    """
    Computes the intermediate quantities that are used in the
    computation of flux. These quantities represent the third
    and first order interpolation of conserved quantities and
    the stabilization term.

    Parameters:
    -----------
    params: Dict
        Dictionary with simulation parameters

    fields: Fields
        A dataclass that contains the simulation variables
    """
    nx = params["nx"]
    nz = params["nz"]
    dz = params["dz"]
    dt = params["dt"]

    hv_coeff = -Constants.hv_beta.value * dz / (16 * dt)
    gamma = Constants.gamma.value
    C0 = Constants.C0.value

    rho = (
        fields.vals_z[IDS.DENS, 0 : nz + 1, 0:nx]
        + fields.hy_dens_int[0 : nz + 1, np.newaxis]
    )
    u = fields.vals_z[IDS.UMOM, 0 : nz + 1, 0:nx] / rho
    w = fields.vals_z[IDS.WMOM, 0 : nz + 1, 0:nx] / rho
    t = (
        fields.vals_z[IDS.RHOT, 0 : nz + 1, 0:nx]
        + fields.hy_dens_theta_int[0 : nz + 1, np.newaxis]
    ) / rho
    pressure = (
        C0 * np.power(rho * t, gamma) - fields.hy_pressure_int[0 : nz + 1, np.newaxis]
    )

    # TODO: this is a weird place to update w; THIS needs to be done somewhere else
    w[0, :] = 0.0
    w[nz, :] = 0.0

    # update density
    fields.d3_vals_z[IDS.DENS, 0, :] = 0.0
    fields.d3_vals_z[IDS.DENS, nz, :] = 0.0

    fields.flux[IDS.DENS, 0 : nz + 1, 0:nx] = (
        rho * w - hv_coeff * fields.d3_vals_z[IDS.DENS]
    )
    fields.flux[IDS.UMOM, 0 : nz + 1, 0:nx] = (
        rho * w * u - hv_coeff * fields.d3_vals_z[IDS.UMOM]
    )
    fields.flux[IDS.WMOM, 0 : nz + 1, 0:nx] = (
        rho * w**2 + pressure - hv_coeff * fields.d3_vals_z[IDS.WMOM]
    )
    fields.flux[IDS.RHOT, 0 : nz + 1, 0:nx] = (
        rho * w * t - hv_coeff * fields.d3_vals_z[IDS.RHOT]
    )


def compute_tend_x(params: Dict, fields: Fields, state: np.ndarray):
    """
    Computes tendencies in x.

    Parameters:
    -----------
    params: Dict
        Dictionary with simulation parameters

    fields: Fields
        A dataclass that contains the simulation variables

    state: np.ndarray
        An array with conservative variables at the current timesteps
    """
    nx = params["nx"]
    nz = params["nz"]
    dx = params["dx"]
    nvariables = 4

    for ivar in range(nvariables):
        fields.tend[ivar, 0:nz, 0:nx] = (
            -(fields.flux[ivar, 0:nz, 1 : nx + 1] - fields.flux[ivar, 0:nz, 0:nx]) / dx
        )


def compute_tend_z(params: Dict, fields: Fields, state: np.ndarray):
    """
    Computes tendencies in z.

    Parameters:
    -----------
    params: Dict
        Dictionary with simulation parameters

    fields: Fields
        A dataclass that contains the simulation variables

    state: np.ndarray
        An array with conservative variables at the current timesteps
    """
    nx = params["nx"]
    nz = params["nz"]
    hs = params["hs"]
    dz = params["dz"]
    nvariables = 4

    for ivar in range(nvariables):
        fields.tend[ivar, 0:nz, 0:nx] = (
            -(fields.flux[ivar, 1 : nz + 1, 0:nx] - fields.flux[ivar, 0:nz, 0:nx]) / dz
        )

    # add source term
    fields.tend[IDS.WMOM, 0:nz, 0:nx] -= (
        state[IDS.DENS, hs : nz + hs, hs : nx + hs] * Constants.grav.value
    )
