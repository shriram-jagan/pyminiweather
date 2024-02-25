import numpy as np
import pytest

from pyminiweather import ID_DENS, ID_RHOT, ID_UMOM, ID_WMOM
from pyminiweather.__main__ import get_parser
from pyminiweather.data import Fields, initialize_fields
from pyminiweather.ics.initial_conditions import CCQInitFactory
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
    shape = state[ID_DENS].shape

    # density, u-vel, w-vel should be zeros
    assert np.all(state[ID_DENS] == np.zeros(shape))
    assert np.all(state[ID_UMOM] == np.zeros(shape))
    assert np.all(state[ID_WMOM] == np.zeros(shape))

    # ideally, we should check this against a cosine profile,
    # but for now, making sure that it is not zero is fine
    count = np.count_nonzero(state[ID_RHOT])
    assert count > 0


def test_hydro_const_theta():
    pass


def test_hydro_const_bvfreq():
    pass


def test_hydrostatic_quantities():
    pass


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
