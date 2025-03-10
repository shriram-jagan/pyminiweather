import cupynumeric as np
import numpy as num
import pytest

from pyminiweather.mesh import MeshData


@pytest.fixture
def params():
    parameters = {}
    parameters = {"nx": 64, "nz": 32, "xlen": 2e4, "zlen": 1e4, "hs": 2}
    parameters["dx"] = parameters["xlen"] / parameters["nx"]
    parameters["dz"] = parameters["zlen"] / parameters["nz"]

    return parameters


def test_mesh_int_ext(params):
    m = MeshData(params)
    xx, zz = m.get_mesh_int_ext()

    x = np.linspace(
        -params["hs"] * params["dx"],
        (params["nx"] + params["hs"]) * params["dx"],
        params["nx"] + 2 * params["hs"],
        endpoint=False,
    )

    z = np.linspace(
        -params["hs"] * params["dz"],
        (params["nz"] + params["hs"]) * params["dz"],
        params["nz"] + 2 * params["hs"],
        endpoint=False,
    )

    assert np.allclose(x, xx[0, :])
    assert np.allclose(z, zz[:, 0])


def test_mesh_cell_centers(params):
    m = MeshData(params)
    xx, zz = m.get_mesh_cell_centers()

    x = np.linspace(
        params["dx"] / 2.0,
        params["xlen"] + params["dx"] / 2.0,
        params["nx"],
        endpoint=False,
    )
    z = np.linspace(
        params["dz"] / 2.0,
        params["zlen"] + params["dz"] / 2.0,
        params["nz"],
        endpoint=False,
    )

    assert np.allclose(x, xx[0, :])
    assert np.allclose(z, zz[:, 0])


def test_mesh_vertical_cell_edges(params):
    m = MeshData(params)
    zz = m.get_mesh_vertical_cell_edges()
    z = np.linspace(
        0.0, (params["nz"] + 1) * params["dz"], params["nz"] + 1, endpoint=False
    )

    assert np.allclose(z, zz)


def test_mesh_vertical_cell_centers_int_ext(params):
    m = MeshData(params)
    zz = m.get_mesh_vertical_cell_centers_int_ext()

    z = np.linspace(
        (-params["hs"] + 0.5) * params["dz"],
        (params["nz"] + params["hs"] + 0.5) * params["dz"],
        params["nz"] + 2 * params["hs"],
        endpoint=False,
    )

    assert np.allclose(z, zz)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
