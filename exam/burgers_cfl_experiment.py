import numpy as np
import matplotlib.pyplot as plt
import os
from burgers_core import phi, dphi_dx, u_initial, u_exact, F

def is_stable(u):
    return np.isfinite(u).all()

def try_cfl(N, cfl, c=4.0, nu=0.1, T=np.pi/4, max_steps=10000):
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    dx = L / N
    u = u_initial(x, c, nu)
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ik = 1j * k
    k2 = k**2
    t = 0.0
    steps = 0
    print(f'  [DEBUG] N={N}, CFL={cfl}, initial max|u|={np.max(np.abs(u)):.6f}')
    try:
        while t < T and steps < max_steps:
            dt = cfl / (np.max(np.abs(u)) / dx + nu / dx**2)
            if t + dt > T:
                dt = T - t
            if steps % 100 == 0:
                print(f'    [DEBUG] step={steps}, t={t:.5f}, dt={dt:.5e}, max|u|={np.max(np.abs(u)):.6f}')
            u1 = u + dt/2 * F(u, k, ik, k2, nu)
            u2 = u + dt/2 * F(u1, k, ik, k2, nu)
            u3 = u + dt * F(u2, k, ik, k2, nu)
            u = (1/3) * (-u + u1 + 2*u2 + u3 + dt/2 * F(u3, k, ik, k2, nu))
            t += dt
            steps += 1
            if not is_stable(u):
                print(f'    [DEBUG] Unstable at step={steps}, t={t:.5f}, max|u|={np.max(np.abs(u)):.6f}')
                return False
        if steps >= max_steps:
            print(f'Warning: max_steps reached for N={N}, CFL={cfl}')
            return False
    except Exception as e:
        print(f'Exception for N={N}, CFL={cfl}: {e}')
        return False
    return True

N_list = [16, 32, 48, 64, 96, 128, 192, 256]
cfl_values = np.arange(0.05, 2.05, 0.05)
results = {}

for N in N_list:
    max_cfl = 0
    for cfl in cfl_values:
        if try_cfl(N, cfl):
            max_cfl = cfl
        else:
            break
    results[N] = max_cfl
    print(f'N={N}, max stable CFL={max_cfl}')


os.makedirs('figure', exist_ok=True)
plt.figure()
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.xlabel('N (number of grid points)')
plt.ylabel('Max stable CFL')
plt.title('Max stable CFL vs N for Burgers equation (T=π/4)')
plt.grid(True)
plt.savefig('exam/figure/burgers_cfl_stability.png', dpi=150)
plt.close() 