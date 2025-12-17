import numpy as np
from sphere_hill.energy import total_energy
from sphere_hill.dynamics import solve_sphere

def test_energy_conservation():
    m, R, g, phi = 1.0, 1.0, 9.81, np.pi/6
    y0 = [0, 0, 0, 0]
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)
    
    sol = solve_sphere(t_span, y0, g=g, phi=phi, R=R, t_eval=t_eval)
    x = sol.y[0]
    theta = sol.y[1]
    v = sol.y[2]
    omega = sol.y[3]

    E0 = total_energy(m, R, g, x[0], v[0], omega[0], phi)
    E_final = total_energy(m, R, g, x[-1], v[-1], omega[-1], phi)
    
    # Allow small numerical tolerance
    assert np.isclose(E0, E_final, rtol=1e-5)
