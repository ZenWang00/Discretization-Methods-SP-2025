import numpy as np
import matplotlib.pyplot as plt
import os
from burgers_core import u_initial, u_exact, F

# Parameters
N = 128
c = 4.0
nu = 0.1
L = 2 * np.pi
x = np.linspace(0, L, N, endpoint=False)
dx = L / N
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ik = 1j * k
k2 = k**2

dt = 0.0005
T_list = [0, np.pi/8, np.pi/6, np.pi/4]

snapshots = []

for T in T_list:
    u = u_initial(x, c, nu)
    nsteps = int(T / dt)
    for n in range(nsteps):
        u1 = u + dt/2 * F(u, k, ik, k2, nu)
        u2 = u + dt/2 * F(u1, k, ik, k2, nu)
        u3 = u + dt   * F(u2, k, ik, k2, nu)
        u  = (1/3) * (-u + u1 + 2*u2 + u3 + dt/2 * F(u3, k, ik, k2, nu))
    u_ex = u_exact(x, T, c, nu)
    snapshots.append((T, u.copy(), u_ex.copy()))

# Plot
os.makedirs('exam/figure', exist_ok=True)
plt.figure(figsize=(8,6))
for T, u_num, u_ex in snapshots:
    plt.plot(x, u_num, label=f'Numerical t={T:.3f}')
    plt.plot(x, u_ex, '--', label=f'Exact t={T:.3f}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Burgers equation solution snapshots (N=128)')
plt.legend()
plt.grid(True)
plt.savefig('exam/figure/burgers_snapshots.png', dpi=150, bbox_inches='tight')
plt.close() 