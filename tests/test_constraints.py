import numpy as np
from sphere_hill.dynamics import solve_sphere

def test_rolling_constraint():
    R = 1.0
    y0 = [0, 0, 0, 0]
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)
    
    sol = solve_sphere(t_span, y0, R=R, t_eval=t_eval)
    x = sol.y[0]
    theta = sol.y[1]

    # x - R*theta ≈ 0
    assert np.allclose(x, R*theta, rtol=1e-5)
