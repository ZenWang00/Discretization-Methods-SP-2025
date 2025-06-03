import numpy as np

def burgers_galerkin_rhs(u_hat, k, nu):
    """
    Compute the right-hand side of the Fourier Galerkin semi-discrete Burgers equation in spectral space.
    Parameters:
        u_hat: Fourier coefficients of u
        k: wavenumbers
        nu: viscosity
    Returns:
        du_hat/dt in spectral space
    """
    u = np.fft.ifft(u_hat).real
    u2 = u * u
    u2_hat = np.fft.fft(u2)
    nonlinear = -1j * k * u2_hat / 2
    viscous = -nu * k**2 * u_hat
    return nonlinear + viscous

def rk4_step_galerkin(u_hat, dt, k, nu):
    """
    Advance u_hat one step using RK4 for the Galerkin system.
    """
    k1 = burgers_galerkin_rhs(u_hat, k, nu)
    k2 = burgers_galerkin_rhs(u_hat + dt/2 * k1, k, nu)
    k3 = burgers_galerkin_rhs(u_hat + dt/2 * k2, k, nu)
    k4 = burgers_galerkin_rhs(u_hat + dt * k3, k, nu)
    return u_hat + (dt/6) * (k1 + 2*k2 + 2*k3 + k4) 