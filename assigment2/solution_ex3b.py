#!/usr/bin/env python3
"""
Exercise 3(b): Long-time integration comparison of infinite order and second order methods
for the hyperbolic equation u_t = -2π u_x.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Create figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

def infinite_derivative(u, D):
    """
    Global differentiation using infinite order method.
    """
    return -2*np.pi * u @ D

def second_order_derivative(u, dx):
    """
    Second order central difference approximation.
    """
    N = len(u) - 1
    du = np.zeros_like(u)
    
    # Periodic boundary conditions
    du[0] = (u[1] - u[-1]) / (2*dx)
    du[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    du[-1] = (u[0] - u[-2]) / (2*dx)
    
    return -2*np.pi * du

def rk4_step(u, dt, derivative_func, *args):
    """
    Special form of fourth-order Runge-Kutta method.
    """
    k1 = u + dt/2 * derivative_func(u, *args)
    k2 = u + dt/2 * derivative_func(k1, *args)
    k3 = u + dt * derivative_func(k2, *args)
    return (-u + k1 + 2*k2 + k3 + dt/2 * derivative_func(k3, *args)) / 3

def exact_solution(x, t):
    """
    Exact solution u(x,t) = exp(sin(x-2πt))
    """
    return np.exp(np.sin(x - 2*np.pi*t))

def fourier_diff_matrix(N):
    """
    Construct the Fourier differentiation matrix using infinite order method.
    """
    D = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                D[i,j] = (-1)**(i+j) / (2 * np.sin((j-i)*np.pi/(N+1)))
    return D

def long_time_integration():
    """
    Long time integration comparison (part b).
    """
    # Parameters
    timesteps = 10000
    T_end = np.pi
    dt = T_end/timesteps
    plot_times = [0, 100, 200]  # Times at which to plot results
    
    # Fine grid for exact solution
    x_fine = 2 * np.pi * np.arange(301) / (301)
    
    # For infinite order method (N=10)
    N_infinite = 10
    x_infinite = 2 * np.pi * np.arange(N_infinite+1) / (N_infinite+1)
    u_infinite = np.exp(np.sin(x_infinite))
    D_infinite = fourier_diff_matrix(N_infinite)
    
    # For second order method (N=200)
    N_second = 200
    x_second = 2 * np.pi * np.arange(N_second+1) / (N_second+1)
    u_second = np.exp(np.sin(x_second))
    dx_second = 2*np.pi/(N_second+1)
    
    # Time integration and plotting
    for method_idx, (N, x, u, method_name, derivative_func, args) in enumerate([
        (N_infinite, x_infinite, u_infinite.copy(), 'infinite', infinite_derivative, (D_infinite,)),
        (N_second, x_second, u_second.copy(), 'second_order', second_order_derivative, (dx_second,))
    ]):
        t = 0
        steps = 0
        
        # Plot initial condition
        plt.figure(figsize=(10, 6))
        plt.plot(x_fine/(2*np.pi), exact_solution(x_fine, t), 'k-', label='Exact')
        plt.plot(x/(2*np.pi), u, 'r--', label=f'Computed')
        plt.grid(True)
        plt.xlabel('x/(2π)')
        plt.ylabel('u')
        plt.legend()
        plt.title(f't = {int(t)}')
        plt.savefig(f'figures/long_time_{method_name}_{int(t)}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Time stepping with progress bar
        pbar = tqdm(total=plot_times[-1], desc=f"Long-time {method_name}")
        while t < plot_times[-1]:
            u = rk4_step(u, dt, derivative_func, *args)
            t += dt
            steps += 1
            pbar.update(dt)
            
            # Plot at specified times
            if any(abs(t - plot_time) < dt/2 for plot_time in plot_times[1:]):
                plt.figure(figsize=(10, 6))
                plt.plot(x_fine/(2*np.pi), exact_solution(x_fine, t), 'k-', label='Exact')
                plt.plot(x/(2*np.pi), u, 'r--', label=f'Computed')
                plt.grid(True)
                plt.xlabel('x/(2π)')
                plt.ylabel('u')
                plt.legend()
                plt.title(f't = {int(t)}')
                plt.savefig(f'figures/long_time_{method_name}_{int(t)}.png', dpi=300, bbox_inches='tight')
                plt.close()
        pbar.close()

def main():
    """
    Main function to run the long-time integration comparison.
    """
    print("Starting long-time integration analysis...")
    long_time_integration()
    print("Computation completed.")

if __name__ == "__main__":
    main() 