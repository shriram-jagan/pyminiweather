from typing import Dict

from pyminiweather import ID_DENS, ID_RHOT, ID_UMOM, ID_WMOM
from pyminiweather import numpy as np
from pyminiweather.data import Fields 


def set_bc_x(params: Dict, fields: Fields, state_forcing: np.ndarray, ic_type: str):
    dz = params["dz"]
    nx = params["nx"]
    nz = params["nz"]
    hs = params["hs"]
    zlen = params["zlen"]

    # update all variables
    state_forcing[:, hs : nz + hs, 0] = state_forcing[:, hs : nz + hs, nx]
    state_forcing[:, hs : nz + hs, 1] = state_forcing[:, hs : nz + hs, nx + 1]

    state_forcing[:, hs : nz + hs, nx + hs] = state_forcing[:, hs : nz + hs, hs]
    state_forcing[:, hs : nz + hs, nx + hs + 1] = state_forcing[:, hs : nz + hs, hs + 1]

    if ic_type == "injection":
        # for injections, update u and t
        z = np.linspace(start=0, stop=nz * dz, num=nz, endpoint=False) + 0.5

        # we need to get the indices where the condition is satisfied
        # since we are only looking at the interior points in the array
        cond = np.fabs(z - 3.0 * zlen / 4.0) <= zlen / 16.0
        indices = np.nonzero(cond)[0] + hs

        # i = 0, 1 for umom
        state_forcing[ID_UMOM, indices, 0] = (
            state_forcing[ID_DENS, indices, 0] + fields.hy_dens_cell[indices]
        ) * 50.0
        state_forcing[ID_UMOM, indices, 1] = (
            state_forcing[ID_DENS, indices, 1] + fields.hy_dens_cell[indices]
        ) * 50.0

        # i = 0, 1 for rho*t
        state_forcing[ID_RHOT, indices, 0] = (
            state_forcing[ID_DENS, indices, 0] + fields.hy_dens_cell[indices]
        ) * 298.0 - fields.hy_dens_theta_cell[indices]
        state_forcing[ID_RHOT, indices, 1] = (
            state_forcing[ID_DENS, indices, 1] + fields.hy_dens_cell[indices]
        ) * 298.0 - fields.hy_dens_theta_cell[indices]


def set_bc_z(params: Dict, fields: Fields, state_forcing: np.ndarray, ic_type: str):
    nx = params["nx"]
    nz = params["nz"]
    hs = params["hs"]

    # W mom
    state_forcing[ID_WMOM, 0, 0 : nx + 2 * hs] = 0.0
    state_forcing[ID_WMOM, 1, 0 : nx + 2 * hs] = 0.0
    state_forcing[ID_WMOM, nz + hs, 0 : nx + 2 * hs] = 0.0
    state_forcing[ID_WMOM, nz + hs + 1, 0 : nx + 2 * hs] = 0.0

    # U mom
    state_forcing[ID_UMOM, 0, 0 : nx + 2 * hs] = (
        state_forcing[ID_UMOM, hs, 0 : nx + 2 * hs]
        / fields.hy_dens_cell[hs]
        * fields.hy_dens_cell[0]
    )
    state_forcing[ID_UMOM, 1, 0 : nx + 2 * hs] = (
        state_forcing[ID_UMOM, hs, 0 : nx + 2 * hs]
        / fields.hy_dens_cell[hs]
        * fields.hy_dens_cell[1]
    )

    state_forcing[ID_UMOM, nz + hs, 0 : nx + 2 * hs] = (
        state_forcing[ID_UMOM, nz + hs - 1, 0 : nx + 2 * hs]
        / fields.hy_dens_cell[nz + hs - 1]
        * fields.hy_dens_cell[nz + hs]
    )
    state_forcing[ID_UMOM, nz + hs + 1, 0 : nx + 2 * hs] = (
        state_forcing[ID_UMOM, nz + hs - 1, 0 : nx + 2 * hs]
        / fields.hy_dens_cell[nz + hs - 1]
        * fields.hy_dens_cell[nz + hs + 1]
    )

    # Density
    state_forcing[ID_DENS, 0, 0 : nx + 2 * hs] = state_forcing[
        ID_DENS, hs, 0 : nx + 2 * hs
    ]
    state_forcing[ID_DENS, 1, 0 : nx + 2 * hs] = state_forcing[
        ID_DENS, hs, 0 : nx + 2 * hs
    ]

    state_forcing[ID_DENS, nz + hs, 0 : nx + 2 * hs] = state_forcing[
        ID_DENS, nz + hs - 1, 0 : nx + 2 * hs
    ]
    state_forcing[ID_DENS, nz + hs + 1, 0 : nx + 2 * hs] = state_forcing[
        ID_DENS, nz + hs - 1, 0 : nx + 2 * hs
    ]

    # Temperature
    state_forcing[ID_RHOT, 0, 0 : nx + 2 * hs] = state_forcing[
        ID_RHOT, hs, 0 : nx + 2 * hs
    ]
    state_forcing[ID_RHOT, 1, 0 : nx + 2 * hs] = state_forcing[
        ID_RHOT, hs, 0 : nx + 2 * hs
    ]

    state_forcing[ID_RHOT, nz + hs, 0 : nx + 2 * hs] = state_forcing[
        ID_RHOT, nz + hs - 1, 0 : nx + 2 * hs
    ]
    state_forcing[ID_RHOT, nz + hs + 1, 0 : nx + 2 * hs] = state_forcing[
        ID_RHOT, nz + hs - 1, 0 : nx + 2 * hs
    ]
