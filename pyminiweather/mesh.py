from typing import Dict, Tuple

from pyminiweather import numpy as np


def meshgrid(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """This is equivalent to numpy's meshgrid for two one-dimensional
    input arrays without all the bells and whistles.

    Parameters:
    ----------
    x: np.ndarray
        1D array denoting x-coordinates
    y: np.ndarray
        1D array denoting y-coordinates

    Returns:
    -------
    A tuple of two-dimensional arrays that correspond to (x, y) pairs
    in a cartesian coordinate system
    """
    if np.__name__ == "numpy":
        return np.meshgrid(x, y)
    elif np.__name__ == "cunumeric":
        assert x.ndim == y.ndim == 1

        return (np.tile(x, (y.size, 1)), np.tile(y, (x.size, 1)).T)
    else:
        raise ValueError("Unknown backend")


class MeshData:
    def __init__(self, params: Dict):
        self.params = params

        self.xlen = self.params["xlen"]
        self.zlen = self.params["zlen"]

        self.dx = self.params["dx"]
        self.dz = self.params["dz"]

        self.nx = self.params["nx"]
        self.nz = self.params["nz"]

        self.hs = self.params["hs"]

        self.mesh_int_ext = None
        self.mesh_cell_centers = None
        self.mesh_vertical_cell_edges = None
        self.mesh_vertical_cell_centers_int_ext = None

    def get_mesh_int_ext(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinate of the mesh that includes both the interior and
        ghost points.

        Returns:
        --------
        A tuple of 2D arrays that represent the x- and z- coordinates
        """

        if self.mesh_int_ext is not None:
            return self.mesh_int_ext

        x = np.linspace(
            -self.hs * self.dx,
            (self.nx + self.hs) * self.dx,
            self.nx + 2 * self.hs,
            endpoint=False,
        )
        z = np.linspace(
            -self.hs * self.dz,
            (self.nz + self.hs) * self.dz,
            self.nz + 2 * self.hs,
            endpoint=False,
        )
        self.mesh_int_ext = meshgrid(x, z)

        return self.mesh_int_ext

    def get_mesh_cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinates of the mesh on the cell centers that includes
        only the interior of the domain

        Returns:
        --------
        A tuple of 2D arrays that represent the x- and z- coordinates
        """

        if self.mesh_cell_centers is not None:
            return self.mesh_cell_centers

        x = np.linspace(
            self.dx / 2.0, self.xlen + self.dx / 2.0, self.nx, endpoint=False
        )
        z = np.linspace(
            self.dz / 2.0, self.zlen + self.dz / 2.0, self.nz, endpoint=False
        )

        self.mesh_cell_centers = meshgrid(x, z)

        return self.mesh_cell_centers

    def get_mesh_vertical_cell_edges(self) -> np.ndarray:
        if self.mesh_vertical_cell_edges is not None:
            return self.mesh_vertical_cell_edges

        self.mesh_vertical_cell_edges = np.linspace(
            0.0, (self.nz + 1) * self.dz, self.nz + 1, endpoint=False
        )

        return self.mesh_vertical_cell_edges

    def get_mesh_vertical_cell_centers_int_ext(self) -> np.ndarray:
        if self.mesh_vertical_cell_centers_int_ext is not None:
            return self.mesh_vertical_cell_centers_int_ext

        self.mesh_vertical_cell_centers_int_ext = np.linspace(
            (-self.hs + 0.5) * self.dz,
            (self.nz + self.hs + 0.5) * self.dz,
            self.nz + 2 * self.hs,
            endpoint=False,
        )

        return self.mesh_vertical_cell_centers_int_ext
