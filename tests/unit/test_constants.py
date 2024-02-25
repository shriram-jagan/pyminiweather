import numpy as np
import pytest

from pyminiweather.data import Constants, Quadrature


def test_constants():
    """Make sure the thermodynamic properties are correct"""
    assert Constants.hv_beta == 0.05
    assert Constants.p0 == 1.0e5
    assert Constants.C0 == 27.5629410929725921310572974482
    assert Constants.gamma == 1.40027894002789400278940027894
    assert Constants.grav == 9.8
    assert Constants.cp == 1004.0
    assert Constants.cv == 717.0
    assert Constants.rd == 287
    assert Constants.pi == 3.14159265358979323846264338327
    assert Constants.theta0 == 300.0
    assert Constants.exner0 == 1.0


def test_quadrature_values():
    """Make sure the quadrature weights and points are correct"""
    assert Quadrature.npoints == 3

    qpoints: np.ndarray = np.array(
        [
            0.112701665379258311482073460022,
            0.500000000000000000000000000000,
            0.887298334620741688517926539980,
        ],
        dtype=np.float64,
    )

    qweights: np.ndarray = np.array(
        [
            0.277777777777777777777777777779,
            0.444444444444444444444444444444,
            0.277777777777777777777777777779,
        ],
        dtype=np.float64,
    )

    assert np.allclose(qpoints, Quadrature.qpoints)
    assert np.allclose(qweights, Quadrature.qweights)


def test_quadrature_immutable():
    """Make sure quadrature class is immutable"""

    exception = AttributeError
    with pytest.raises(exception):
        Quadrature.npoints = 4

    with pytest.raises(exception):
        Quadrature.qweights = 1
    with pytest.raises(exception):
        Quadrature.qpoints_grid_x = 3
    with pytest.raises(exception):
        Quadrature.qpoints_grid_z = 1
    with pytest.raises(exception):
        Quadrature.qweights_grid_x = 3
    with pytest.raises(exception):
        Quadrature.qweights_grid_z = 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
