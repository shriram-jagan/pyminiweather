from typing import Dict

from pyminiweather import IDS
from pyminiweather import numpy as np
from pyminiweather.data import Constants, Fields


def compute_stats(params: Dict, fields: Fields) -> float:
    """Return the total mass and kinetic energy"""
    nx = params["nx"]
    nz = params["nz"]
    dx = params["dx"]
    dz = params["dz"]
    hs = params["hs"]

    rho = (
        fields.state[IDS.DENS][hs : nz + hs, hs : nx + hs]
        + fields.hy_dens_cell[hs : nz + hs, np.newaxis]
    )
    u = fields.state[IDS.UMOM][hs : nz + hs, hs : nx + hs] / rho
    w = fields.state[IDS.WMOM][hs : nz + hs, hs : nx + hs] / rho
    th = (
        fields.state[IDS.RHOT][hs : nz + hs, hs : nx + hs]
        + fields.hy_dens_theta_cell[hs : nz + hs, np.newaxis]
    ) / rho
    p = Constants.C0.value * np.power(rho * th, Constants.gamma.value)
    t = th / np.power(Constants.p0.value / p, Constants.rd.value / Constants.cp.value)

    ke = rho * (u * u + w * w)
    ie = rho * Constants.cv.value * t

    total_mass = rho.sum() * dx * dz
    total_energy = (ke + ie).sum() * dx * dz

    return total_mass, total_energy


def compute_solution_variables(params: Dict, fields: Fields) -> np.ndarray:
    """Return the solution variables (Add more info)"""
    nx = params["nx"]
    nz = params["nz"]
    hs = params["hs"]

    variables = np.zeros((fields.nvariables, nz, nx), dtype=np.float64)

    variables[IDS.DENS, ...] = fields.state[IDS.DENS, hs : nz + hs, hs : nx + hs]

    variables[IDS.UMOM, ...] = fields.state[IDS.UMOM, hs : nz + hs, hs : nx + hs] / (
        fields.hy_dens_cell[hs : nz + hs, np.newaxis]
        + fields.state[IDS.DENS, hs : nz + hs, hs : nx + hs]
    )

    variables[IDS.WMOM, ...] = fields.state[IDS.WMOM, hs : nz + hs, hs : nx + hs] / (
        fields.hy_dens_cell[hs : nz + hs, np.newaxis]
        + fields.state[IDS.DENS, hs : nz + hs, hs : nx + hs]
    )

    variables[IDS.RHOT, ...] = (
        fields.state[IDS.RHOT, hs : nz + hs, hs : nx + hs]
        + fields.hy_dens_theta_cell[hs : nz + hs, np.newaxis]
    ) / (
        fields.hy_dens_cell[hs : nz + hs, np.newaxis]
        + fields.state[IDS.DENS, hs : nz + hs, hs : nx + hs]
    ) - (
        fields.hy_dens_theta_cell[hs : nz + hs, np.newaxis]
        / fields.hy_dens_cell[hs : nz + hs, np.newaxis]
    )

    return variables
