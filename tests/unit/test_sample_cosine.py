import cupynumeric as np
import numpy as num
from pyminiweather.__main__ import get_parser
from pyminiweather.data import Constants, initialize_fields
from pyminiweather.mesh import MeshData 
from pyminiweather.utils import sample_ellipse_cosine


def test_sample_ellipse_cosine():
    """
    Choose the ellipse such that the distance criteria is not
    satisified and we get zeros for all but one element.
    """

    parser = get_parser()
    args, _ = parser.parse_known_args()
    params = vars(args)

    params["dx"] = params["xlen"] / params["nx"]
    params["dz"] = params["zlen"] / params["nz"]

    fields = initialize_fields(params)
    mesh = MeshData(params)

    x, z = mesh.get_mesh_int_ext()
    amplitude = 1.0
    x0 = params["xlen"] / 2.0
    z0 = params["zlen"] / 2.0
    xrad = 1e-4
    zrad = 1e-4

    data = sample_ellipse_cosine(x, z, amplitude, x0, z0, xrad, zrad)
    cond = data == 0.0
    how_many = np.count_nonzero(cond)

    assert how_many == data.size - 1


def test_sample_ellipse_cosine_cosine_dist():
    """
    Choose the ellipse such that the distance criteria is
    satisifed for all elements and we get a cosine profile.
    """

    parser = get_parser()
    args, _ = parser.parse_known_args()
    params = vars(args)

    params["dx"] = params["xlen"] / params["nx"]
    params["dz"] = params["zlen"] / params["nz"]

    fields = initialize_fields(params)
    mesh = MeshData(params)

    x, z = mesh.get_mesh_int_ext()
    amplitude = 1.0
    x0 = params["xlen"] / 2.0
    z0 = params["zlen"] / 2.0
    xrad = 1e4
    zrad = 1e4

    data_exact = sample_ellipse_cosine(x, z, amplitude, x0, z0, xrad, zrad)

    dist = (
        np.sqrt(((x - x0) / xrad) ** 2 + ((z - z0) / zrad) ** 2)
        * Constants.pi.value
        / 2.0
    )
    data_computed = np.zeros(z.shape, dtype=x.dtype)
    condition = dist <= Constants.pi.value / 2.0
    np.putmask(data_computed, condition, amplitude * (np.cos(dist) ** 2.0))

    assert np.allclose(data_computed, data_exact)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
