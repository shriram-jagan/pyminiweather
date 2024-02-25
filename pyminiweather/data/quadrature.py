from enum import Enum

from pyminiweather import meshgrid
from pyminiweather import numpy as np


# TODO: Attributes should be immutable.
class GaussianQuadrature:
    """Contains Gauss-Legendre quadrature points and weights"""

    def __init__(self):
        self.__npoints: int = 3
        self.__qpoints: np.ndarray = np.array(
            [
                0.112701665379258311482073460022,
                0.500000000000000000000000000000,
                0.887298334620741688517926539980,
            ],
            dtype=np.float64,
        )
        self.__qweights: np.ndarray = np.array(
            [
                0.277777777777777777777777777779,
                0.444444444444444444444444444444,
                0.277777777777777777777777777779,
            ],
            dtype=np.float64,
        )

        self.__qweights_outer: np.ndarray = np.outer(self.__qweights, self.__qweights)

        self.__qpoints_grid_x, self.__qpoints_grid_z = meshgrid(
            self.__qpoints, self.__qpoints
        )
        self.__qweights_grid_x, self.__qweights_grid_z = meshgrid(
            self.__qweights, self.__qweights
        )

    @property
    def npoints(self):
        return self.__npoints

    @property
    def qpoints(self):
        return self.__qpoints

    @property
    def qweights(self):
        return self.__qweights

    @property
    def qweights_outer(self):
        return self.__qweights_outer

    @property
    def qpoints_grid_x(self):
        return self.__qpoints_grid_x

    @property
    def qpoints_grid_z(self):
        return self.__qpoints_grid_z

    @property
    def qweights_grid_x(self):
        return self.__qweights_grid_x

    @property
    def qweights_grid_z(self):
        return self.__qweights_grid_z


Quadrature = GaussianQuadrature()
