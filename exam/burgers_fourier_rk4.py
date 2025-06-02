import numpy as np
import matplotlib.pyplot as plt
import os
from burgers_core import phi, dphi_dx, u_initial, u_exact, F

# Parameters for the Burgers' equation
N  = 129  # Number of grid points (odd)
c  = 4.0  # Wave speed
nu = 0.1  # Viscosity coefficient
L  = 2 * np.pi  # Domain length
x  = np.linspace(0, L, N, endpoint=False)  # Grid points
dx = L / N  # Grid spacing

# Spectral differentiation operators
k   = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Wavenumbers
ik  = 1j * k  # i*k for first derivative
k2  = k**2    # k^2 for second derivative

# Time integration parameters
T   = 1.0  # Final time
CFL = 0.002  # CFL number for stability
max_steps = 5000000  # Maximum number of time steps

# Initial condition
u = u_initial(x, c, nu)

# Time integration using RK4
t = 0.0
steps = 0
while t < T and steps < max_steps:
    # Adaptive time step based on CFL condition
    Umax = np.max(np.abs(u))
    Ueff = max(Umax, 1e-8)  # Avoid division by zero
    dt   = CFL / (Ueff/dx + nu/(dx*dx))
    if t + dt > T:
        dt = T - t
    
    # RK4 time stepping
    u1 = u + dt/2 * F(u, k, ik, k2, nu)
    u2 = u + dt/2 * F(u1, k, ik, k2, nu)
    u3 = u + dt   * F(u2, k, ik, k2, nu)
    u  = (1/3) * (-u + u1 + 2*u2 + u3 + dt/2 * F(u3, k, ik, k2, nu))
    
    t += dt
    steps += 1
    
    # Check for numerical instability
    if not np.isfinite(u).all():
        raise RuntimeError(f"Numerical instability detected at t={t:.6f} (CFL={CFL})")

# Compute exact solution for comparison
u_ex = u_exact(x, T, c, nu)

# Plot results
plt.figure(figsize=(8,6))
plt.plot(x, u, label='Numerical (RK4)')
plt.plot(x, u_ex, '--', label='Exact (Hopf-Cole)')
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Burgers\' Equation: Fourier Collocation + RK4')
plt.legend()
plt.grid(True)

# Save figure
os.makedirs('exam/figure', exist_ok=True)
plt.savefig('exam/figure/burgers_solution_part2a.png', dpi=150, bbox_inches='tight')
plt.close()

# Print numerical details
print(f"Simulation completed:")
print(f"- Final time: {t:.6f}")
print(f"- Total steps: {steps}")
print(f"- Final CFL: {CFL}")
print(f"- L2 error: {np.linalg.norm(u - u_ex) / np.sqrt(N):.2e}")
