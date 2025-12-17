import numpy as np

def translational_ke(m: float, v: float) -> float:
    """Translational kinetic energy: (1/2) m v^2"""
    return 0.5 * m * v**2

def rotational_ke(m: float, R: float, omega: float) -> float:
    """Rotational kinetic energy of a solid sphere: (1/2) I omega^2, I = 2/5 m R^2"""
    I = 2/5 * m * R**2
    return 0.5 * I * omega**2

# Modified potential_energy
def potential_energy(m: float, g: float, x: float, phi: float, x0=0.0) -> float:
    """Gravitational potential energy relative to x0"""
    return m * g * (x0 - x) * np.sin(phi)  # height decreases → KE increases


def total_energy(m: float, R: float, g: float, x: float, v: float, omega: float, phi: float) -> float:
    """Total energy"""
    return translational_ke(m, v) + rotational_ke(m, R, omega) + potential_energy(m, g, x, phi)
