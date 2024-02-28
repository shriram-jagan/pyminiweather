from typing import Dict

from pyminiweather import numpy as np
from pyminiweather.data import Constants, Fields, Quadrature 
from .initial_conditions import CCQInitFactory, VCEQInitFactory
from pyminiweather.mesh import MeshData


def init(fields: Fields, params: Dict, Mesh: MeshData):
    xlen = params["xlen"]
    dx = params["dx"]
    dz = params["dz"]
    ic_type = params["ic_type"]

    C0 = Constants.C0.value
    gamma = Constants.gamma.value

    assert ic_type in [
        "thermal",
        "collision",
        "gravity",
        "density-current",
        "injection",
    ]

    init_full_domain = CCQInitFactory(ic_type)
    init_interface = VCEQInitFactory(ic_type)

    ## Do the whole domain
    # x = np.linspace(-hs * dx, (nx + hs) * dx, nx + 2 * hs, endpoint=False)
    # z = np.linspace(-hs * dz, (nz + hs) * dz, nz + 2 * hs, endpoint=False)
    # x, z = meshgrid(x, z)

    x, z = Mesh.get_mesh_int_ext()

    x = x[:, :, np.newaxis, np.newaxis] + Quadrature.qpoints_grid_x * dx
    z = z[:, :, np.newaxis, np.newaxis] + Quadrature.qpoints_grid_z * dz

    r, u, w, t, hr, ht = init_full_domain(x, z, xlen)

    fields.state[0, ...] = (
        np.multiply(r, Quadrature.qweights_outer).sum(axis=-1).sum(axis=-1)
    )
    fields.state[1, ...] = (
        np.multiply((r + hr) * u, Quadrature.qweights_outer).sum(axis=-1).sum(axis=-1)
    )
    fields.state[2, ...] = (
        np.multiply((r + hr) * w, Quadrature.qweights_outer).sum(axis=-1).sum(axis=-1)
    )
    fields.state[3, ...] = (
        np.multiply((r + hr) * (t + ht) - hr * ht, Quadrature.qweights_outer)
        .sum(axis=-1)
        .sum(axis=-1)
    )

    # Update state_tmp
    fields.state_tmp[:] = fields.state[:]

    # Do boundaries
    # z = np.linspace((-hs + 0.5) * dz, (nz + hs + 0.5) * dz, nz + 2 * hs, endpoint=False)
    z = Mesh.get_mesh_vertical_cell_centers_int_ext()
    hr, ht = init_interface(z)  # ht should be a scalar

    # Hydrostatic density and Hydrostatic potential temperature at cell-centers
    # on the vertical interface. Note that the actual computation is
    # hr * sigma_weights but is done differently in the cpp version
    sum_qweights = Quadrature.qweights.sum()
    fields.hy_dens_cell[:] = hr * sum_qweights

    fields.hy_dens_theta_cell[:] = np.multiply(
        (ht * hr)[:, np.newaxis], Quadrature.qweights
    ).sum(axis=-1)

    # Hydrostatic density, pressure, and potential temperature evaluated at
    # cell-edges on the vertical interface (hy_dens_int, hy_pressure_int, hy_dens_theta_int)
    # z = np.linspace(0.0, (nz + 1) * dz, nz + 1, endpoint=False)
    z = Mesh.get_mesh_vertical_cell_edges()

    hr, ht = init_interface(z)
    fields.hy_dens_int[:] = hr[:]
    fields.hy_dens_theta_int[:] = hr * ht
    fields.hy_pressure_int[:] = C0 * ((hr * ht) ** gamma)
