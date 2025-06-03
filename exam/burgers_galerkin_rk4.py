import numpy as np
import matplotlib.pyplot as plt
import os
from burgers_core import u_initial, u_exact
from burgers_galerkin_core import burgers_galerkin_rhs, rk4_step_galerkin

# Parameters
N = 128
c = 4.0
nu = 0.1
L = 2 * np.pi
x = np.linspace(0, L, N, endpoint=False)
k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
kmax = N // 2

# Initial condition: project to Fourier space
u0 = u_initial(x, c, nu)
u_hat = np.fft.fft(u0)

# Time stepping parameters
T = 1.0
CFL = 0.002

t = 0.0
steps = 0
max_steps = 5000000

while t < T and steps < max_steps:
    u = np.fft.ifft(u_hat).real
    dt = CFL / (np.max(np.abs(u)) * kmax + nu * kmax**2)
    if t + dt > T:
        dt = T - t
    u_hat = rk4_step_galerkin(u_hat, dt, k, nu)
    t += dt
    steps += 1
    if not np.isfinite(u_hat).all():
        raise RuntimeError(f"Numerical instability at t={t:.6f}")

# Transform back to physical space
u_num = np.fft.ifft(u_hat).real
u_ex = u_exact(x, T, c, nu)

# Plot
os.makedirs('exam/figure', exist_ok=True)
plt.figure(figsize=(8,6))
plt.plot(x, u_num, label='Numerical (Galerkin RK4, t=1)')
plt.plot(x, u_ex, '--', label='Exact (Hopf-Cole, t=1)')
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Burgers Equation: Fourier Galerkin + RK4 (N=128)')
plt.legend()
plt.grid(True)
plt.savefig('exam/figure/burgers_galerkin_solution.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Simulation completed: t={t:.6f}, steps={steps}, L2 error={np.linalg.norm(u_num-u_ex)/np.sqrt(N):.2e}") 