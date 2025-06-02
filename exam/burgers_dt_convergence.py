import numpy as np
import matplotlib.pyplot as plt
import os
from burgers_core import u_initial, u_exact, F

# Parameters
N = 129
c = 4.0
nu = 0.1
L = 2 * np.pi
x = np.linspace(0, L, N, endpoint=False)
dx = L / N
T = 1.0

# Spectral differentiation operators
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ik = 1j * k
k2 = k**2

# Initial condition (Hopf-Cole)
u0 = u_initial(x, c, nu)
print('[DEBUG] u0: min', np.min(u0), 'max', np.max(u0), 'mean', np.mean(u0))

dt_list = [0.01, 0.005, 0.0025, 0.00125, 0.000625]
errors = []

for dt in dt_list:
    u = u0.copy()
    nsteps = int(T / dt)
    print(f'[DEBUG] dt={dt}, nsteps={nsteps}')
    for n in range(nsteps):
        u1 = u + dt/2 * F(u, k, ik, k2, nu)
        u2 = u + dt/2 * F(u1, k, ik, k2, nu)
        u3 = u + dt   * F(u2, k, ik, k2, nu)
        u  = (1/3) * (-u + u1 + 2*u2 + u3 + dt/2 * F(u3, k, ik, k2, nu))
        if n % 10 == 0 or n == nsteps-1:
            print(f'  [DEBUG] step={n}, t={dt*n:.4f}, u: min {np.min(u):.4e}, max {np.max(u):.4e}, mean {np.mean(u):.4e}')
        if not np.isfinite(u).all():
            print(f'  [ERROR] NaN or Inf detected at step={n}, t={dt*n:.4f}')
            break
    u_ref = u_exact(x, T, c, nu)
    print('[DEBUG] u_ref: min', np.min(u_ref), 'max', np.max(u_ref), 'mean', np.mean(u_ref))
    err = np.linalg.norm(u - u_ref) / np.sqrt(N)
    errors.append(err)
    print(f'[DEBUG] dt={dt}, L2 error={err:.3e}')

# Compute convergence order
orders = [np.log(errors[i-1]/errors[i])/np.log(dt_list[i-1]/dt_list[i]) for i in range(1, len(errors))]
print('Convergence orders:', orders)

# Plot
os.makedirs('exam/figure', exist_ok=True)
plt.figure()
plt.loglog(dt_list, errors, 'o-', label='RK4')
plt.xlabel('Time step size $\Delta t$')
plt.ylabel('L2 error at $t=1.0$')
plt.title('Time step convergence for Burgers equation')
plt.grid(True, which='both')
plt.legend()
plt.savefig('exam/figure/burgers_dt_convergence.png', dpi=150, bbox_inches='tight')
plt.close() 