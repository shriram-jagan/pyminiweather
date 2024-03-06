import numpy as np
import pytest

from pyminiweather import IDS
from pyminiweather.__main__ import get_parser
from pyminiweather.data import Fields, initialize_fields
from pyminiweather.ics.initial_conditions import CCQInitFactory
from pyminiweather.ics.initial_conditions import CollisionInterior
from pyminiweather.ics import init
from pyminiweather.mesh import MeshData


def test_thermal():
    """Make sure that the thermal initial condition is correct"""
    ic_type: str = "thermal"
    init_interior = CCQInitFactory(ic_type)

    parser = get_parser()
    args, _ = parser.parse_known_args()
    params = vars(args)

    params["dx"] = params["xlen"] / params["nx"]
    params["dz"] = params["zlen"] / params["nz"]

    fields = initialize_fields(params)
    mesh = MeshData(params)

    init(fields, params, mesh)

    state = fields.state
    shape = state[IDS.DENS].shape

    # density, u-vel, w-vel should be zeros
    assert np.all(state[IDS.DENS] == np.zeros(shape))
    assert np.all(state[IDS.UMOM] == np.zeros(shape))
    assert np.all(state[IDS.WMOM] == np.zeros(shape))

    # ideally, we should check this against a cosine profile,
    # but for now, making sure that it is not zero is fine
    count = np.count_nonzero(state[IDS.RHOT])
    assert count > 0


def test_hydro_const_theta():
    """
    Make sure hydrostatic density and potential temperature
    are computed correctly.
    """

    c = CollisionInterior()

    # no variation of potential temperature
    assert c.hydro_const_theta(1.0)[1] == c.hydro_const_theta(100.0)[1]

    # no variation with z; gradient(hydrostatic dens, z) = 0
    nz = 25000
    lz = 100.0
    z = np.linspace(0, lz, nz)
    out = c.hydro_const_theta(z)[0]
    mean_gradient = np.gradient(out).mean()

    assert np.isclose(mean_gradient, 0.0, atol=1e-4)


def test_hydro_const_bvfreq():
    """
    Check hydrostatic density and potential temperature are
    computed correctly for a given bvfreq.
    """

    bv_freq0 = 0.02
    nz = 25000
    lz = 100.0

    z = np.linspace(0, lz, nz)

    c = CollisionInterior()
    out = c.hydro_const_bvfreq(z, bv_freq0)

    # Check that the mean gradient is close to zero
    assert np.isclose(np.gradient(out[0]).mean(), 0.0, atol=1e-4)
    assert np.isclose(np.gradient(out[1]).mean(), 0.0, atol=1e-4)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
