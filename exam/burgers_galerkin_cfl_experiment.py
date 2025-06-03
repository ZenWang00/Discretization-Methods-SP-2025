import numpy as np
import matplotlib.pyplot as plt
import os
from burgers_core import u_initial
from burgers_galerkin_core import rk4_step_galerkin

N_list = [16, 32, 48, 64, 96, 128, 192, 256]
CFL_values = np.arange(0.05, 2.05, 0.05)
T = np.pi / 4
c = 4.0
nu = 0.1

results = {}

for N in N_list:
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    kmax = N // 2
    u0 = u_initial(x, c, nu)
    max_cfl = 0
    for CFL in CFL_values:
        u_hat = np.fft.fft(u0)
        t = 0.0
        steps = 0
        max_steps = 1000000
        stable = True
        while t < T and steps < max_steps:
            u = np.fft.ifft(u_hat).real
            dt = CFL / (np.max(np.abs(u)) * kmax + nu * kmax**2)
            if t + dt > T:
                dt = T - t
            u_hat = rk4_step_galerkin(u_hat, dt, k, nu)
            t += dt
            steps += 1
            if not np.isfinite(u_hat).all():
                stable = False
                break
        if stable:
            max_cfl = CFL
        else:
            break
    results[N] = max_cfl
    print(f'N={N}, max stable CFL={max_cfl}')

os.makedirs('exam/figure', exist_ok=True)
plt.figure()
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.xlabel('N (number of grid points)')
plt.ylabel('Max stable CFL')
plt.title('Max stable CFL vs N for Burgers equation (Galerkin, T=pi/4)')
plt.grid(True)
plt.savefig('exam/figure/burgers_galerkin_cfl_stability.png', dpi=150, bbox_inches='tight')
plt.close() 