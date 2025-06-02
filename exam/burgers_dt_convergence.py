import numpy as np
import matplotlib.pyplot as plt
import os

def phi(a, b, nu):
    k = np.arange(-50, 51)
    a = np.atleast_1d(a)
    K, A = np.meshgrid(k, a, indexing='ij')
    val = np.exp(-((A - (2*K+1)*np.pi)**2) / (4*nu*b))
    return np.sum(val, axis=0)

def u_exact(x, t, c, nu):
    return c - 2*nu * (phi(x - c*t + 1, t + 1, nu) / phi(x - c*t, t + 1, nu))

def F(u, k, ik, k2, nu):
    u_hat = np.fft.fft(u)
    du_dx = np.fft.ifft(ik * u_hat).real
    d2u_dx2 = np.fft.ifft(-k2 * u_hat).real
    return -u * du_dx + nu * d2u_dx2

N = 129
c = 4.0
nu = 0.1
L = 2 * np.pi
x = np.linspace(0, L, N, endpoint=False)
dx = L / N
T = 1.0
u0 = u_exact(x, 0, c, nu)
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ik = 1j * k
k2 = k**2

dt_list = [0.1, 0.05, 0.025, 0.0125, 0.00625]
errors = []

for dt in dt_list:
    u = u0.copy()
    nsteps = int(T / dt)
    print(f'[DEBUG] dt={dt}, nsteps={nsteps}')
    for n in range(nsteps):
        u1 = u + dt/2 * F(u, k, ik, k2, nu)
        u2 = u + dt/2 * F(u1, k, ik, k2, nu)
        u3 = u + dt * F(u2, k, ik, k2, nu)
        u = (1/3) * (-u + u1 + 2*u2 + u3 + dt/2 * F(u3, k, ik, k2, nu))
        if n % 100 == 0 or n == nsteps-1:
            print(f'  [DEBUG] step={n}, t={dt*n:.4f}')
    u_ref = u_exact(x, T, c, nu)
    err = np.linalg.norm(u - u_ref) / np.sqrt(N)
    errors.append(err)
    print(f'[DEBUG] dt={dt}, L2 error={err:.3e}')

os.makedirs('figure', exist_ok=True)
plt.figure()
plt.loglog(dt_list, errors, 'o-', label='RK4')
plt.xlabel('Time step size $\Delta t$')
plt.ylabel('L2 error at $t=1.0$')
plt.title('Time step convergence for Burgers equation')
plt.grid(True, which='both')
plt.legend()
plt.savefig('figure/burgers_dt_convergence.png', dpi=150)
plt.close() 