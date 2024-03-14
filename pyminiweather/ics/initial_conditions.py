from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

from pyminiweather import numpy as np
from pyminiweather.data import Constants, Quadrature
from pyminiweather.utils import sample_ellipse_cosine


class InitBase(ABC):
    def __init__(self):
        """
        Base class with helper functions to be used for all
        types of initial conditions.
        """
        pass

    @abstractmethod
    def __call__(
        self,
    ):
        pass

    # Helper functions from here on

    def hydro_const_theta(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
        -----------
        z: np.ndarray
            Input coordinate

        Returns:
        --------
        Output background hydrostatic density and potential temperature

        """
        ht = Constants.theta0.value
        exner = Constants.exner0.value - Constants.grav.value * z / (
            Constants.cp.value * Constants.theta0.value
        )  # Exner pressure at z
        p = Constants.p0.value * (
            exner ** (Constants.cp.value / Constants.rd.value)
        )  # Pressure at z
        hr = (
            (p / Constants.C0.value) ** (1.0 / Constants.gamma.value)
        ) / ht  # rho*theta at z is numerator; rhs is Density

        # TODO: ht is a constant and not a array yet but we can make
        # it the same shape

        return (hr, ht)

    def hydro_const_bvfreq(
        self, z: np.ndarray, bv_freq0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Establish hydrostatic balance using constant Brunt-Vaisala frequency

        Parameters
        -----------
        z: np.ndarray
            Input coordinate
        bv_freq0: float
            Constant Brunt-Vaisala frequency

        Returns:
        -------
        Output background hydrostatic density and potential temperature

        """
        ht = Constants.theta0.value * np.exp(
            bv_freq0 * bv_freq0 / Constants.grav.value * z
        )
        exner = Constants.exner0.value - Constants.grav.value * Constants.grav.value / (
            Constants.cp.value * bv_freq0 * bv_freq0
        ) * (ht - Constants.theta0.value) / (ht * Constants.theta0.value)
        p = Constants.p0.value * (exner ** (Constants.cp.value / Constants.rd.value))
        hr = ((p / Constants.C0.value) ** (1.0 / Constants.gamma.value)) / ht

        return (hr, ht)

    def sample_ellipse_cosine(
        self,
        x: np.ndarray,
        z: np.ndarray,
        amplitude: float,
        x0: float,
        z0: float,
        xrad: float,
        zrad: float,
    ):
        return sample_ellipse_cosine(x, z, amplitude, x0, z0, xrad, zrad)


class CollisionInterior(InitBase):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, x: np.ndarray, z: np.ndarray, xlen: float):
        """
        Parameters:
        -----------
        x: np.ndarray
            Input x coordinate

        z: np.ndarray
            Input z coordinate

        xlen: float
            x length to be used in the ellipse creation

        Returns:
        --------
        Density, u velocity, w velocity, temperature, potential
        density and potential temperature
        """

        assert x.ndim == z.ndim == 4
        assert x.shape == z.shape

        hr, ht = self.hydro_const_theta(z)

        r = np.zeros(x.shape)
        t = np.zeros(x.shape)
        u = np.zeros(x.shape)
        w = np.zeros(x.shape)

        t = self.sample_ellipse_cosine(x, z, 20.0, xlen / 2, 2000.0, 2000.0, 2000.0)
        t += self.sample_ellipse_cosine(x, z, -20.0, xlen / 2, 8000.0, 2000.0, 2000.0)

        return r, u, w, t, hr, ht


class CollisionVCEQInit(InitBase):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, z) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
        -----------
        z: np.ndarray
            Input z coordinate

        Returns:
        --------
        Potential density and temperature

        Notes:
        ------
        This function is simpler than the original cpp implementation since
        the calls to sample_ellipse_cosine were superfluous in the original
        implementation

        """

        hr, ht = self.hydro_const_theta(z)
        return hr, ht


class ThermalInterior(InitBase):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, x: np.ndarray, z: np.ndarray, xlen: float):
        """
        Parameters:
        -----------
        x: np.ndarray
            Input x coordinate

        z: np.ndarray
            Input z coordinate

        xlen: float
            x length to be used in the ellipse creation

        Returns:
        --------
        Density, u velocity, w velocity, temperature, potential
        density and potential temperature

        """

        hr, ht = self.hydro_const_theta(z)
        r = np.zeros(x.shape)
        t = np.zeros(x.shape)
        u = np.zeros(x.shape)
        w = np.zeros(x.shape)
        t += self.sample_ellipse_cosine(x, z, 3.0, xlen / 2, 2000.0, 2000.0, 2000.0)

        return r, u, w, t, hr, ht


class GravityWavesInterior(InitBase):
    def __init__(self, bv0_freq: float = 0.02):
        super().__init__()
        self.bv0_freq = bv0_freq

    def __call__(self, x: np.ndarray, z: np.ndarray, xlen: float):
        """
        Parameters:
        -----------
        x: np.ndarray
            Input x coordinate

        z: np.ndarray
            Input z coordinate

        xlen: float
            x length to be used in the ellipse creation

        Returns:
        --------
        Density, u velocity, w velocity, temperature, potential
        density and potential temperature

        """

        hr, ht = self.hydro_const_bvfreq(z, self.bv0_freq)
        r = np.zeros(x.shape)
        t = np.zeros(x.shape)
        u = 15.0 * np.ones(x.shape)
        w = np.zeros(x.shape)

        return r, u, w, t, hr, ht


class DensityCurrentInterior(InitBase):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, x: np.ndarray, z: np.ndarray, xlen: float):
        """
        Parameters:
        -----------
        x: np.ndarray
            Input x coordinate

        z: np.ndarray
            Input z coordinate

        xlen: float
            x length to be used in the ellipse creation

        Returns:
        --------
        Density, u velocity, w velocity, temperature, potential
        density and potential temperature

        """
        hr, ht = self.hydro_const_theta(z)
        r = np.zeros(x.shape)
        t = np.zeros(x.shape)
        u = np.zeros(x.shape)
        w = np.zeros(x.shape)
        t += self.sample_ellipse_cosine(x, z, -20.0, xlen / 2, 5000.0, 4000.0, 2000.0)

        return r, u, w, t, hr, ht


class InjectionInterior(InitBase):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, x: np.ndarray, z: np.ndarray, xlen: float):
        """
        Parameters:
        -----------
        x: np.ndarray
            Input x coordinate

        z: np.ndarray
            Input z coordinate

        xlen: float
            x length to be used in the ellipse creation

        Returns:
        --------
        Density, u velocity, w velocity, temperature, potential
        density and potential temperature

        """
        hr, ht = self.hydro_const_theta(z)
        r = np.zeros(x.shape)
        t = np.zeros(x.shape)
        u = np.zeros(x.shape)
        w = np.zeros(x.shape)

        return r, u, w, t, hr, ht


class InjectionVCEQInit(InitBase):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, z: np.ndarray):
        return self.hydro_const_theta(z)


class ThermalVCEQInit(InitBase):
    def __call__(self, z: np.ndarray):
        return self.hydro_const_theta(z)


class DensityCurrentVCEQInit(InitBase):
    def __call__(self, z: np.ndarray):
        return self.hydro_const_theta(z)


class GravityWavesVCEQInit(InitBase):
    def __init__(self, bv0_freq: float = 0.02):
        self.bv0_freq = bv0_freq

    def __call__(self, z: np.ndarray):
        return self.hydro_const_bvfreq(z, self.bv0_freq)


class VCEQInitFactory(ABC):
    def __init__(self, ic_type: str):
        """VCEQ stands for Vertical cell-edge quantities. This is a factory
        class that helps with initialization of quantities that
        are stored at cell-edges along the vertical direction (z).
        The argument to the callable should be coordinates in
        the z-direction."""

        self.ic_type = ic_type
        self.callable = None

        if self.ic_type == "collision":
            self.callable = CollisionVCEQInit()
        if self.ic_type == "thermal":
            self.callable = ThermalVCEQInit()
        if self.ic_type == "gravity":
            self.callable = GravityWavesVCEQInit()
        if self.ic_type == "density-current":
            self.callable = DensityCurrentVCEQInit()
        if self.ic_type == "injection":
            self.callable = InjectionVCEQInit()

    def __call__(self, z):
        """
        Parameters:
        -----------
        z: np.ndarray
            Input z coordinate

        Returns:
        --------
        Potential density and potential temperature
        """

        return self.callable(z)


class CCQInitFactory(ABC):
    def __init__(self, ic_type: str):
        """CCQ stands for cell-centered quantities. This is a factory
        class that helps with initialization of quantities that
        are stored at cell-centers"""

        self.ic_type = ic_type
        self.callable = None

        if self.ic_type == "collision":
            self.callable = CollisionInterior()
        if self.ic_type == "thermal":
            self.callable = ThermalInterior()
        if self.ic_type == "gravity":
            self.callable = GravityWavesInterior()
        if self.ic_type == "density-current":
            self.callable = DensityCurrentInterior()
        if self.ic_type == "injection":
            self.callable = InjectionInterior()

    def is_collision(self) -> bool:
        """Return True if the initial condition if of type Collision"""
        return self.ic_type == "collision"

    def is_thermal(self) -> bool:
        """Return True if the initial condition if of type Thermal"""
        return self.ic_type == "thermal"

    def is_gravity_waves(self) -> bool:
        """Return True if the initial condition if of type Gravity Waves"""
        return self.ic_type == "gravity"

    def is_density_current(self) -> bool:
        """Return True if the initial condition if of type Density Current"""
        return self.ic_type == "density-current"

    def is_injection(self) -> bool:
        """Return True if the initial condition if of type Injection"""
        return self.ic_type == "injection"

    def __call__(self, x: np.ndarray, z: np.ndarray, xlen: float) -> Any:
        """
        Parameters:
        -----------
        x: np.ndarray
            Input x coordinate

        z: np.ndarray
            Input z coordinate

        xlen: float
            x length to be used in the ellipse creation

        Returns:
        --------
        Density, u velocity, w velocity, temperature, potential
        density and potential temperature

        """
        return self.callable(x, z, xlen)
