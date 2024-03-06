from typing import Dict

from pyminiweather import numpy as np
from pyminiweather.data import Fields
from pyminiweather.ics import Directions, set_bc_x, set_bc_z
from pyminiweather.mesh import MeshData

from .interpolate import (
    compute_flux_x,
    compute_flux_z,
    compute_tend_x,
    compute_tend_z,
    interpolate_x,
    interpolate_z,
)
from .source import add_source_terms

reverse_direction: bool = False


def discrete_step(
    params: Dict,
    fields: Fields,
    mesh: MeshData,
    state_init: np.ndarray,
    state_forcing: np.ndarray,
    state_out: np.ndarray,
    dt: float,
    direction: Directions,
) -> None:
    """
    Perform one discrete timestep for the ODE. A discrete step
    consists of a halo exchange, an interpolation, and computation
    of fluxes and tendencies. The conservative variables are then
    updated.

    Parameters:
    ----------
    params: Dict
        Dictionary with simulation parameters

    fields: Fields
        A dataclass that contains the simulation variables

    state_init: np.ndarray
        An array with conservative variables at the beginning of
        the time step

    state_forcing: np.ndarray
        An array with conservative variables at the current timestep
        that is used for the RHS computation

    state_out: np.ndarray
        An array with conservative variables that gets updated

    dt: float
        Timestep

    direction: Directions
        Direction for this substep
    """

    nz = params["nz"]
    nx = params["nx"]

    ic_type = params["ic_type"]
    if direction == Directions.X:
        set_bc_x(params, fields, state_forcing, ic_type)
        interpolate_x(params, fields, state_forcing)
        compute_flux_x(params, fields)
        compute_tend_x(params, fields, state_forcing)
    if direction == Directions.Z:
        set_bc_z(params, fields, state_forcing, ic_type)
        interpolate_z(params, fields, state_forcing)
        compute_flux_z(params, fields)
        compute_tend_z(params, fields, state_forcing)

    add_source_terms(params, mesh, fields)

    state_out[:, 2 : nz + 2, 2 : nx + 2] = (
        state_init[:, 2 : nz + 2, 2 : nx + 2] + dt * fields.tend[:]
    )


def evolve(params, fields, mesh, dt: float = 1e-4) -> None:
    """
    Step through in time by one timestep of dt. This consists of
    three discrete timesteps with the direction getting
    switched automatically (between x and z)

    Parameters:
    -----------
    params: Dict
        Dictionary with simulation parameters

    fields: Fields
        A dataclass that contains the simulation variables

    mesh: MeshData
        Class that stores coordinates at different points in the domain
    """

    global reverse_direction

    directions = (
        [Directions.X, Directions.Z]
        if reverse_direction
        else [Directions.Z, Directions.X]
    )

    for direction in directions:
        discrete_step(
            params,
            fields,
            mesh,
            fields.state,
            fields.state,
            fields.state_tmp,
            dt / 3,
            direction,
        )
        discrete_step(
            params,
            fields,
            mesh,
            fields.state,
            fields.state_tmp,
            fields.state_tmp,
            dt / 2,
            direction,
        )
        discrete_step(
            params,
            fields,
            mesh,
            fields.state,
            fields.state_tmp,
            fields.state,
            dt / 1,
            direction,
        )

    reverse_direction = False if reverse_direction else True
