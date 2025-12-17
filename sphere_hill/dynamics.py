import numpy as np
from .model import linear_acceleration, angular_acceleration

def equations_of_motion(t, y, g=9.81, phi=np.pi/4, R=1.0):
    """
    Computes the derivatives for a sphere rolling down an incline.

    Parameters:
    - t : float
        Time (not used, but required by solve_ivp)
    - y : array-like
        Current state: [x, theta, xdot, thetadot]
    - g : float
        Gravitational acceleration
    - phi : float
        Incline angle in radians
    - R : float
        Sphere radius

    Returns:
    - dydt : list
        Time derivatives: [xdot, thetadot, xddot, thetaddot]
    """
    x, theta, xdot, thetadot = y

    xddot = linear_acceleration(g, phi)
    thetaddot = angular_acceleration(g, phi, R)

    return [xdot, thetadot, xddot, thetaddot]


def solve_sphere(t_span, y0, g=9.81, phi=np.pi/4, R=1.0, **kwargs):
    """
    Solves the ODE for a rolling sphere.

    Parameters:
    - t_span : tuple
        Start and end times
    - y0 : list
        Initial state: [x0, theta0, xdot0, thetadot0]
    - kwargs : passed to solve_ivp
    """
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        equations_of_motion,
        t_span,
        y0,
        args=(g, phi, R),
        **kwargs
    )
    return sol

