from pyminiweather import numpy as np
from pyminiweather.data import Constants


def sample_ellipse_cosine(
    x: np.ndarray,
    z: np.ndarray,
    amplitude: float,
    x0: float,
    z0: float,
    xrad: float,
    zrad: float,
) -> np.ndarray:
    """
    Sample from an ellipse of a specified center, radius, and amplitude
    at a specified location

    For e.g., see eqn (35) from here for more details.
    https://journals.ametsoc.org/view/journals/mwre/143/12/mwr-d-15-0134.1.xml

    A squared cosine profile is used to create the profile inside the ellipse


    Parameters:
    ----------
    x: np.ndarray
        x coordinate of the input location
    z: np.ndarray
        z coordinate of the input location
    amplitude: float
        Amplitude of the cosine function
    xrad: float
        x radius of the ellipse
    zrad: float
        z radius of the ellipse

    Returns:
    -------
    A profile that is non-zero inside a bubble with the prescribed radius
    """

    dist = (
        np.sqrt(((x - x0) / xrad) ** 2 + ((z - z0) / zrad) ** 2)
        * Constants.pi.value
        / 2.0
    )
    data = np.zeros(z.shape, dtype=x.dtype)
    condition = dist <= Constants.pi.value / 2.0
    np.putmask(data, condition, amplitude * (np.cos(dist) ** 2.0))

    return data
