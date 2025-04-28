import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fourier_diff_matrix(N):
    """
    Construct the Fourier differentiation matrix using the infinite order method.
    """
    D = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                D[i,j] = (-1)**(i+j) / (2 * np.sin((j-i)*np.pi/(N+1)))
    return D

def second_order_derivative(u, dx):
    """
    Second-order central difference approximation with periodic boundary conditions.
    """
    N = len(u) - 1
    result = np.zeros_like(u)
    
    # Interior points
    result[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    
    # Boundary points with periodic conditions
    result[0] = (u[1] - u[-1]) / (2*dx)
    result[-1] = (u[0] - u[-2]) / (2*dx)
    
    return -2*np.pi * result

def fourth_order_derivative(u, dx):
    """
    Fourth-order central difference approximation with periodic boundary conditions.
    """
    N = len(u) - 1
    result = np.zeros_like(u)
    
    # Interior points
    for j in range(2, N-1):
        result[j] = (-u[j+2] + 8*u[j+1] - 8*u[j-1] + u[j-2]) / (12*dx)
    
    # Boundary points with periodic conditions
    result[0] = (-u[2] + 8*u[1] - 8*u[-1] + u[-2]) / (12*dx)
    result[1] = (-u[3] + 8*u[2] - 8*u[0] + u[-1]) / (12*dx)
    result[-2] = (-u[0] + 8*u[-1] - 8*u[-3] + u[-4]) / (12*dx)
    result[-1] = (-u[1] + 8*u[0] - 8*u[-2] + u[-3]) / (12*dx)
    
    return -2*np.pi * result

def fourier_derivative(u, D):
    """
    Global spectral differentiation using Fourier transform.
    """
    N = len(u) - 1
    # Compute Fourier coefficients
    u_hat = np.fft.fft(u[:-1])  # Remove last point as it's the same as first point
    # Wavenumbers
    k = np.fft.fftfreq(N, 1/N)
    # Compute derivative in Fourier space
    du_hat = 1j * k * u_hat
    # Transform back to physical space
    du = np.fft.ifft(du_hat)
    # Add periodic point
    return -2*np.pi * np.concatenate([np.real(du), [np.real(du[0])]])

def rk4_step(u, dt, derivative_func, *args):
    """
    Special form of RK4 time stepping.
    """
    k1 = u + dt/2 * derivative_func(u, *args)
    k2 = u + dt/2 * derivative_func(k1, *args)
    k3 = u + dt * derivative_func(k2, *args)
    return (-u + k1 + 2*k2 + k3 + dt/2 * derivative_func(k3, *args)) / 3

def exact_solution(x, t):
    """
    Exact solution of the hyperbolic equation.
    """
    return np.exp(np.sin(x - 2*np.pi*t))

def infinite_derivative(u, D):
    """
    Global differentiation using infinite order method.
    """
    return -2*np.pi * u @ D

def error_analysis():
    """
    Error analysis for different spatial discretization methods (part a).
    """
    N_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    timesteps = 10000
    T_end = np.pi
    dt = T_end/timesteps
    
    errors = np.zeros((3, len(N_values)))
    
    for idx, N in enumerate(N_values):
        logging.info(f"Processing N = {N}")
        
        # Grid setup
        x = 2 * np.pi * np.arange(N+1) / (N+1)
        dx = 2*np.pi/(N+1)
        
        # Initial condition
        u0 = np.exp(np.sin(x))
        
        # Infinite order method
        D = fourier_diff_matrix(N)
        u = u0.copy()
        for _ in tqdm(range(timesteps), desc=f"Infinite Order N={N}", leave=False):
            u = rk4_step(u, dt, infinite_derivative, D)
        u_exact = exact_solution(x, T_end)
        errors[0, idx] = np.max(np.abs(u - u_exact))
        
        # Second order method
        u = u0.copy()
        for _ in tqdm(range(timesteps), desc=f"Second Order N={N}", leave=False):
            u = rk4_step(u, dt, second_order_derivative, dx)
        errors[1, idx] = np.max(np.abs(u - u_exact))
        
        # Fourth order method
        u = u0.copy()
        for _ in tqdm(range(timesteps), desc=f"Fourth Order N={N}", leave=False):
            u = rk4_step(u, dt, fourth_order_derivative, dx)
        errors[2, idx] = np.max(np.abs(u - u_exact))
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    methods = ['Infinite Order', 'Second Order', 'Fourth Order']
    markers = ['o', 's', '^']
    for i in range(3):
        plt.loglog(N_values, errors[i,:], f'-{markers[i]}', label=methods[i])
    plt.grid(True)
    plt.xlabel('N')
    plt.ylabel('L∞ Error')
    plt.legend()
    plt.title('Convergence Analysis')
    plt.savefig('figures/convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return errors

def long_time_integration():
    """
    Long time integration comparison (part b).
    """
    # Parameters for time integration
    dt = 0.001
    plot_times = [0, 100, 200]
    steps_between_plots = int(100/dt)
    
    # Fine grid for exact solution
    x_fine = 2 * np.pi * np.arange(301) / 301
    
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
        (N_infinite, x_infinite, u_infinite.copy(), 'Infinite Order', infinite_derivative, (D_infinite,)),
        (N_second, x_second, u_second.copy(), 'Second Order', second_order_derivative, (dx_second,))
    ]):
        t = 0
        plot_idx = 0
        total_steps = int(plot_times[-1]/dt)
        
        # Plot initial condition
        plt.figure(figsize=(10, 6))
        plt.plot(x_fine/(2*np.pi), exact_solution(x_fine, t), 'k-', label='Exact')
        plt.plot(x/(2*np.pi), u, 'r--', label=f'Computed ({method_name})')
        plt.grid(True)
        plt.xlabel('x/(2π)')
        plt.ylabel('u')
        plt.legend()
        plt.title(f't = {t}')
        plt.savefig(f'figures/long_time_{method_name.lower()}_{int(t)}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Time stepping with progress bar
        pbar = tqdm(total=total_steps, desc=f"Long-time {method_name}")
        while t < plot_times[-1]:
            u = rk4_step(u, dt, derivative_func, *args)
            t += dt
            pbar.update(1)
            
            if abs(t - plot_times[plot_idx+1]) < dt/2:
                plt.figure(figsize=(10, 6))
                plt.plot(x_fine/(2*np.pi), exact_solution(x_fine, t), 'k-', label='Exact')
                plt.plot(x/(2*np.pi), u, 'r--', label=f'Computed ({method_name})')
                plt.grid(True)
                plt.xlabel('x/(2π)')
                plt.ylabel('u')
                plt.legend()
                plt.title(f't = {int(t)}')
                plt.savefig(f'figures/long_time_{method_name.lower()}_{int(t)}.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
                plot_idx += 1
                if plot_idx >= len(plot_times)-1:
                    break
        pbar.close()

def main():
    """
    Main function to run both analyses.
    """
    logging.info("Starting error analysis")
    errors = error_analysis()
    
    # Print error table
    methods = ['Infinite Order', 'Second Order', 'Fourth Order']
    N_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    print("\nError Analysis Results:")
    print("="*60)
    print(f"{'N':>8} | " + " | ".join(f"{method:>15}" for method in methods))
    print("-"*60)
    for i, N in enumerate(N_values):
        print(f"{N:8d} | " + " | ".join(f"{err:15.2e}" for err in errors[:,i]))
    print("="*60)
    
    logging.info("Starting long time integration")
    long_time_integration()
    logging.info("Computation completed")

if __name__ == "__main__":
    main() 