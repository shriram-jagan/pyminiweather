from abc import ABC, abstractmethod
from typing import Dict

from pyminiweather import IDS
from pyminiweather import numpy as np
from pyminiweather.data import Fields
from pyminiweather.mesh import MeshData
from pyminiweather.utils import sample_ellipse_cosine


class SourceTerm(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self):
        pass


class GravityBCSourceTerm(SourceTerm):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, params: Dict, mesh: MeshData, fields: Fields):
        """
        Updates tendency with the source term associated
        with Gravity boundary condition.

        Parameters:
        ----------
        params: Dict
            Dictionary with simulation parameters

        mesh: MeshData
            Class that stores coordinates at different points in the domain

        fields: Fields
            A dataclass that contains the simulation variables
        """
        nz = params["nz"]
        x, z = mesh.get_mesh_cell_centers()
        wpert = sample_ellipse_cosine(
            x, z, 0.01, params["xlen"] / 8, 1000.0, 500.0, 500.0
        )

        fields.tend[IDS.WMOM, :, :] += (
            wpert * fields.hy_dens_cell[2 : nz + 2, np.newaxis]
        )


def add_source_terms(params: Dict, mesh: MeshData, fields: Fields):
    """
    Add source terms to tendencies.

    Parameters:
    ----------
    params: Dict
        Dictionary with simulation parameters

    mesh: MeshData
        Class that stores coordinates at different points in the domain

    fields: Fields
        A dataclass that contains the simulation variables

    """

    sources = []
    if params["ic_type"] == "gravity":
        sources.append(GravityBCSourceTerm())

    for source in sources:
        source(params, mesh, fields)
