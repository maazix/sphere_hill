import numpy as np
from sphere_hill.dynamics import equations_of_motion

def test_constant_acceleration():
    g = 9.81
    phi = np.pi / 6
    R = 1.0
    y = [0, 0, 0, 0]  # initial x, theta, xdot, thetadot
    dydt = equations_of_motion(0, y, g=g, phi=phi, R=R)
    
    xddot = dydt[2]
    thetaddot = dydt[3]

    # Expected acceleration
    expected_a = (5.0/7.0) * g * np.sin(phi)
    expected_alpha = expected_a / R

    assert np.isclose(xddot, expected_a)
    assert np.isclose(thetaddot, expected_alpha)
