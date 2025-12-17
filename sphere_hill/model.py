import numpy as np


def linear_acceleration(g: float, phi: float) -> float:
    """
    Linear acceleration of a solid sphere rolling down an incline.

    a = (5/7) * g * sin(phi)
    """
    return (5.0 / 7.0) * g * np.sin(phi)


def angular_acceleration(g: float, phi: float, R: float) -> float:
    """
    Angular acceleration from rolling constraint.
    """
    return linear_acceleration(g, phi) / R
