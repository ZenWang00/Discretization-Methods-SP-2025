import numpy as np

_debug_f_counter = 0

def phi(a, b, nu=0.1, M=50):
    k = np.arange(-M, M+1)
    a = np.atleast_1d(a)
    K, A = np.meshgrid(k, a, indexing='ij')
    arg = A - (2*K + 1)*np.pi
    return np.sum(np.exp(- (arg**2) / (4 * nu * b)), axis=0)

def dphi_dx(a, b, nu=0.1, M=50):
    k = np.arange(-M, M+1)
    a = np.atleast_1d(a)
    K, A = np.meshgrid(k, a, indexing='ij')
    arg = A - (2*K + 1)*np.pi
    factor = -arg / (2 * nu * b)
    return np.sum(factor * np.exp(- arg**2 / (4 * nu * b)), axis=0)

def u_initial(x, c, nu):
    phi_x1  = phi(x, 1.0, nu)
    dphi_x1 = dphi_dx(x, 1.0, nu)
    return c - 2 * nu * (dphi_x1 / phi_x1)

def u_exact(x, t, c, nu, M=50):
    if t <= 0:
        return u_initial(x, c, nu)
    a = x - c * t
    b = t + 1.0
    phi_val  = phi(a, b, nu, M)
    dphi_val = dphi_dx(a, b, nu, M)
    return c - 2 * nu * (dphi_val / phi_val)

def F(u, k, ik, k2, nu):
    global _debug_f_counter
    u_hat   = np.fft.fft(u)
    du_dx   = np.fft.ifft(ik  * u_hat ).real
    d2u_dx2 = np.fft.ifft(-k2 * u_hat ).real
    out = -u * du_dx + nu * d2u_dx2
    if _debug_f_counter < 5:
        print(f'[DEBUG F] call {_debug_f_counter}:')
        print(f'  u:      min {np.min(u):.4e}, max {np.max(u):.4e}, mean {np.mean(u):.4e}')
        print(f'  du_dx:  min {np.min(du_dx):.4e}, max {np.max(du_dx):.4e}, mean {np.mean(du_dx):.4e}')
        print(f'  d2u_dx2:min {np.min(d2u_dx2):.4e}, max {np.max(d2u_dx2):.4e}, mean {np.mean(d2u_dx2):.4e}')
        print(f'  F(u):   min {np.min(out):.4e}, max {np.max(out):.4e}, mean {np.mean(out):.4e}')
        _debug_f_counter += 1
    return out 