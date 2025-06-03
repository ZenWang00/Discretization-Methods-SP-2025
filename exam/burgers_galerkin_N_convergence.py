import numpy as np
import matplotlib.pyplot as plt
import os
from burgers_core import u_initial, u_exact
from burgers_galerkin_core import rk4_step_galerkin

N_list = [16, 32, 48, 64, 96, 128, 192, 256]
CFL = 2.0  # Use the maximum stable CFL from previous experiment
T = np.pi / 4
c = 4.0
nu = 0.1

errors = []

for N in N_list:
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    kmax = N // 2
    u0 = u_initial(x, c, nu)
    u_hat = np.fft.fft(u0)
    t = 0.0
    steps = 0
    max_steps = 1000000
    while t < T and steps < max_steps:
        u = np.fft.ifft(u_hat).real
        dt = CFL / (np.max(np.abs(u)) * kmax + nu * kmax**2)
        if t + dt > T:
            dt = T - t
        u_hat = rk4_step_galerkin(u_hat, dt, k, nu)
        t += dt
        steps += 1
        if not np.isfinite(u_hat).all():
            print(f'N={N} unstable at step={steps}, t={t:.4f}')
            break
    u_num = np.fft.ifft(u_hat).real
    u_ref = u_exact(x, T, c, nu)
    err = np.max(np.abs(u_num - u_ref))
    errors.append(err)
    print(f'N={N}, Linf error={err:.3e}')

# Compute convergence order
orders = [np.log(errors[i-1]/errors[i])/np.log(N_list[i]/N_list[i-1]) for i in range(1, len(errors))]
print('Convergence orders:', orders)

# Plot
os.makedirs('exam/figure', exist_ok=True)
plt.figure()
plt.loglog(N_list, errors, 'o-', label='$L^\infty$ error')
plt.xlabel('Number of grid points $N$')
plt.ylabel('$L^\infty$ error at $t=\pi/4$')
plt.title('Spatial convergence for Burgers equation (Galerkin)')
plt.grid(True, which='both')
plt.legend()
plt.savefig('exam/figure/burgers_galerkin_N_convergence.png', dpi=150, bbox_inches='tight')
plt.close() 