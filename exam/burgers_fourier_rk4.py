import numpy as np
import matplotlib.pyplot as plt
import os

N = 129
c = 4.0
nu = 0.1
L = 2 * np.pi
x = np.linspace(0, L, N, endpoint=False)
dx = L / N

def phi(a, b, nu=nu):
    k = np.arange(-50, 51)
    a = np.atleast_1d(a)
    K, A = np.meshgrid(k, a, indexing='ij')
    # K: (101, len(a)), A: (101, len(a))
    val = np.exp(-((A - (2*K+1)*np.pi)**2) / (4*nu*b))
    return np.sum(val, axis=0)  # shape: (len(a),)

def u_exact(x, t, c=c, nu=nu):
    return c - 2*nu * (phi(x - c*t + 1, t + 1) / phi(x - c*t, t + 1))

u = u_exact(x, 0)

k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ik = 1j * k
k2 = k**2

def F(u):
    u_hat = np.fft.fft(u)
    du_dx = np.fft.ifft(ik * u_hat).real
    d2u_dx2 = np.fft.ifft(-k2 * u_hat).real
    return -u * du_dx + nu * d2u_dx2


dt = 0.001
T = 1.0
nsteps = int(T / dt)

for n in range(nsteps):
    u1 = u + dt/2 * F(u)
    u2 = u + dt/2 * F(u1)
    u3 = u + dt * F(u2)
    u = (1/3) * (-u + u1 + 2*u2 + u3 + dt/2 * F(u3))

plt.plot(x, u, label='Numerical')
plt.plot(x, u_exact(x, T), '--', label='Exact')
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.title(f'Burger Equation Solution at t={T}')
os.makedirs('figure', exist_ok=True)
plt.savefig('figure/burgers_solution.png', dpi=150)
plt.show() 