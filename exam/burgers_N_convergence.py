import numpy as np
import matplotlib.pyplot as plt
import os
from burgers_core import u_initial, u_exact, F

# Parameters
N_list = [16, 32, 48, 64, 96, 128, 192, 256]
dt = 0.0005  # sufficiently small for all N
T = np.pi / 4
c = 4.0
nu = 0.1

errors = []

for N in N_list:
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    dx = L / N
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ik = 1j * k
    k2 = k**2
    u = u_initial(x, c, nu)
    nsteps = int(T / dt)
    for n in range(nsteps):
        u1 = u + dt/2 * F(u, k, ik, k2, nu)
        u2 = u + dt/2 * F(u1, k, ik, k2, nu)
        u3 = u + dt   * F(u2, k, ik, k2, nu)
        u  = (1/3) * (-u + u1 + 2*u2 + u3 + dt/2 * F(u3, k, ik, k2, nu))
    u_ref = u_exact(x, T, c, nu)
    err = np.max(np.abs(u - u_ref))
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
plt.title('Spatial convergence for Burgers equation')
plt.grid(True, which='both')
plt.legend()
plt.savefig('exam/figure/burgers_N_convergence.png', dpi=150, bbox_inches='tight')
plt.close() 