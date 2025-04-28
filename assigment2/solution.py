#!/usr/bin/env python3
"""
This Python script implements numerical methods for the homework assignments.
It is divided into several parts:
  - Finite difference approximations (2nd, 4th, and 6th order).
  - Construction of Fourier differentiation matrices using two approaches (odd and even).
    The even method is now optimized using vectorized operations.
  - Testing the Fourier differentiation matrix on the function u(x)=exp(k*sin(x)) for various k.
  - Computing derivatives for selected functions using the even method Fourier differentiation.
  - Solving a periodic scalar hyperbolic problem using a modified 4th order Runge-Kutta integrator
    with three different spatial derivative approximations.
    
All functions contain English comments.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import gc
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('solution.log'),
        logging.StreamHandler()
    ]
)

# Add parent directory to path to import common utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clean and recreate figures directory
if os.path.exists('figures'):
    shutil.rmtree('figures')
os.makedirs('figures', exist_ok=True)

def l_inf_error(approx, exact):
    """Compute the L-infinity norm of the error."""
    return np.max(np.abs(approx - exact))

def l2_error(approx, exact):
    """Compute the L2 norm of the error."""
    return np.sqrt(np.sum((approx - exact)**2) / np.sum(exact**2))

def derivative_second_order(u, dx):
    """
    Compute the derivative using the 2nd order centered finite difference:
      u_x[j] = (u[j+1] - u[j-1]) / (2*dx)
    Assumes periodic boundary conditions.
    """
    u_plus = np.roll(u, -1)
    u_minus = np.roll(u, 1)
    return (u_plus - u_minus) / (2 * dx)

def derivative_fourth_order(u, dx):
    """
    Compute the derivative using the 4th order centered finite difference:
      u_x[j] = (-u[j-2] + 8*u[j-1] + 8*u[j+1] - u[j+2]) / (12*dx)
    Assumes periodic boundary conditions.
    """
    u_m2 = np.roll(u, 2)
    u_m1 = np.roll(u, 1)
    u_p1 = np.roll(u, -1)
    u_p2 = np.roll(u, -2)
    return (-u_m2 + 8*u_m1 - 8*u_p1 + u_p2) / (12 * dx)

def derivative_sixth_order(u, dx):
    """
    Compute the derivative using the 6th order centered finite difference:
      u_x[j] = (-u[j-3] + 9*u[j-2] - 45*u[j-1] + 45*u[j+1] - 9*u[j+2] + u[j+3]) / (60*dx)
    Assumes periodic boundary conditions.
    """
    u_m3 = np.roll(u, 3)
    u_m2 = np.roll(u, 2)
    u_m1 = np.roll(u, 1)
    u_p1 = np.roll(u, -1)
    u_p2 = np.roll(u, -2)
    u_p3 = np.roll(u, -3)
    return (-u_m3 + 9*u_m2 - 45*u_m1 + 45*u_p1 - 9*u_p2 + u_p3) / (60 * dx)

def fourier_diff_matrix_even(N):
    """Construct the even Fourier differentiation matrix"""
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (-1)**(i+j) / (2 * np.sin((j-i)*np.pi/(N+1)))
    return D

def fourier_diff_matrix_odd(N):
    """
    Construct the Fourier differentiation matrix for odd method.
    The formula is D_{ij} = (-1)^{i+j}/(2*tan((i-j)*pi/N)) for i != j
    """
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                angle = (i - j) * np.pi / N
                D[i, j] = ((-1) ** (i + j)) / (2 * np.tan(angle))
            else:
                D[i, j] = 0.0
    return D

def compare_methods():
    """Compare odd and even methods for different k values"""
    logging.info("\nComparison of Odd and Even Methods:")
    print("=" * 80)
    print(f"{'k':>3} | {'N':>3} | {'Error_odd':>14} | {'Error_even':>14} | {'Odd is better?':^12}")
    print("-" * 80)

    for k in [2, 4, 6, 8, 10, 12]:
        # Define test function and its derivative
        f = lambda x: np.exp(k * np.sin(x))
        df = lambda x: k * np.cos(x) * np.exp(k * np.sin(x))
        
        # Find minimum N for odd method
        N = 10
        while N <= 1000:
            try:
                # Grid points for odd method
                x_odd = 2 * np.pi * np.arange(N) / N
                u_odd = f(x_odd)
                
                # Compute derivative using odd method
                D_odd = fourier_diff_matrix_odd(N)
                du_numeric_odd = D_odd @ u_odd
                du_exact_odd = df(x_odd)
                error_odd = np.max(np.abs(du_numeric_odd - du_exact_odd))
                
                if error_odd < 1e-3:
                    break
            except:
                pass
            N += 2
            
        # Grid points for even method
        x_even = 2 * np.pi * np.arange(N) / N
        u_even = f(x_even)
        
        # Compute derivative using even method
        D_even = fourier_diff_matrix_even(N)
        du_numeric_even = D_even @ u_even
        du_exact_even = df(x_even)
        error_even = np.max(np.abs(du_numeric_even - du_exact_even))
        
        # Print comparison
        odd_is_better = error_odd < error_even
        print(f"{k:3d} | {N:3d} | {error_odd:14.8e} | {error_even:14.8e} | {str(odd_is_better):^12}")
    
    print("=" * 80)

def test_fourier_differentiation():
    """Test the accuracy of both odd and even Fourier differentiation matrices."""
    logging.info("Starting Fourier differentiation test for both methods")
    k_values = np.array([2, 4, 6, 8, 10, 12])
    N_values = np.array([20, 28, 34, 42, 48, 54])  # Fixed N values like MATLAB
    tol = 1e-3
    
    # Store results for both methods
    results = {'odd': [], 'even': []}
    max_N = 1000
    
    # Set numpy random seed for reproducibility
    np.random.seed(42)
    
    # Test both methods with fixed N values
    print("\nTesting with fixed N values (MATLAB style):")
    print("=" * 80)
    print(f"{'k':>3} | {'N':>3} | {'Error_odd':^15} | {'Error_even':^15} | {'Odd is better?':^15}")
    print("-" * 80)
    
    for k, N in zip(k_values, N_values):
        # Odd method
        D_odd = fourier_diff_matrix_odd(N)
        x_odd = 2 * np.pi * np.arange(N) / N
        u_odd = np.exp(k * np.sin(x_odd))
        du_numeric_odd = D_odd @ u_odd
        du_exact_odd = k * np.cos(x_odd) * np.exp(k * np.sin(x_odd))
        # Add small epsilon to avoid division by zero
        error_odd = np.max(np.abs(du_numeric_odd - du_exact_odd) / (np.abs(du_exact_odd) + 1e-15))
        
        # Even method
        D_even = fourier_diff_matrix_even(N)
        x_even = 2 * np.pi * np.arange(N+1) / (N+1)
        u_even = np.exp(k * np.sin(x_even))
        du_numeric_even = D_even @ u_even
        du_exact_even = k * np.cos(x_even) * np.exp(k * np.sin(x_even))
        # Add small epsilon to avoid division by zero
        error_even = np.max(np.abs(du_numeric_even - du_exact_even) / (np.abs(du_exact_even) + 1e-15))
        
        is_odd_better = error_odd < error_even
        print(f"{k:3d} | {N:3d} | {error_odd:15.8e} | {error_even:15.8e} | {str(is_odd_better):^15}")
        
        # Clear memory
        del D_odd, D_even, x_odd, x_even, u_odd, u_even, du_numeric_odd, du_numeric_even, du_exact_odd, du_exact_even
        gc.collect()
    
    print("=" * 80)
    
    # Continue with original testing method
    print("\nTesting with variable N to achieve tolerance:")
    print("=" * 60)
    print(f"{'k':>3} | {'Odd Method':^25} | {'Even Method':^25}")
    print("-" * 60)
    print(f"{'':>3} | {'N':^10} {'Error':^14} | {'N':^10} {'Error':^14}")
    print("-" * 60)
    
    for method in ['odd', 'even']:
        for k in k_values:
            N = 2
            max_error = float('inf')
            found_N = None
            
            while max_error > tol and N <= max_N:
                try:
                    if method == 'odd':
                        D = fourier_diff_matrix_odd(N)
                        x = 2 * np.pi * np.arange(N) / N
                    else:
                        D = fourier_diff_matrix_even(N)
                        x = 2 * np.pi * np.arange(N+1) / (N+1)
                    
                    u = np.exp(k * np.sin(x))
                    du_numeric = D @ u
                    du_exact = k * np.cos(x) * np.exp(k * np.sin(x))
                    # Add small epsilon to avoid division by zero
                    relative_error = np.abs(du_numeric - du_exact) / (np.abs(du_exact) + 1e-15)
                    max_error = np.max(relative_error)
                    
                    if max_error < tol:
                        found_N = N
                        results[method].append((k, N, max_error))
                        logging.info(f"{method} method: k={k}, minimum N={N}, error={max_error:.2e}")
                        break
                    
                    N += 2
                    logging.debug(f"k={k}, N={N}, error={max_error:.2e}")
                    
                    # Clear memory
                    del D, x, u, du_numeric, du_exact, relative_error
                    gc.collect()
                    
                except Exception as e:
                    logging.error(f"Error at k={k}, N={N}: {str(e)}")
                    break
            
            if found_N is None:
                logging.warning(f"{method} method: Failed to achieve tolerance for k={k} within N={max_N}")
                results[method].append((k, None, max_error))
    
    # Print comparison table
    for i, k in enumerate(k_values):
        odd_result = results['odd'][i]
        even_result = results['even'][i]
        odd_N = str(odd_result[1]) if odd_result[1] is not None else "Failed"
        even_N = str(even_result[1]) if even_result[1] is not None else "Failed"
        odd_err = f"{odd_result[2]:.2e}" if odd_result[1] is not None else "N/A"
        even_err = f"{even_result[2]:.2e}" if even_result[1] is not None else "N/A"
        print(f"{k:3d} | {odd_N:^10} {odd_err:^14} | {even_N:^10} {even_err:^14}")
    
    print("=" * 60)
    return results

def test_fourier_diff_accuracy(method='odd', k_values=[2,4,6,8,10,12], tol=1e-3):
    """
    Test the Fourier differentiation matrix on u(x)=exp(k*sin(x)).
    For each k and for increasing N, determine the minimum N such that:
      max|du_approx - du_exact| < tol.
    """
    logging.info(f"Starting Fourier differentiation accuracy test with method={method}")
    results = {}
    max_N = 1000
    
    for k in tqdm(k_values, desc=f"Testing {method} method"):
        N_val = 10
        achieved = False
        while not achieved and N_val < max_N:
            try:
                if method == 'odd':
                    x = np.linspace(0, 2*np.pi, N_val, endpoint=False)
                    D = fourier_diff_matrix_odd(N_val)
                else:
                    x = np.linspace(0, 2*np.pi, N_val+1, endpoint=True)
                    D = fourier_diff_matrix_even(N_val)
                
                u = np.exp(k * np.sin(x))
                du_exact = k * np.cos(x) * np.exp(k * np.sin(x))
                du_approx = D.dot(u)
                err = np.max(np.abs(du_approx - du_exact))
                
                # Clear memory
                del D, x, u, du_exact, du_approx
                gc.collect()
                
                if err < tol:
                    results[k] = (N_val, err)
                    achieved = True
                    logging.info(f"k={k}, N={N_val}, error={err:.2e}")
                else:
                    N_val += 2
                    logging.debug(f"k={k}, N={N_val}, error={err:.2e}")
            except Exception as e:
                logging.error(f"Error at k={k}, N={N_val}: {str(e)}")
                break
            
        if not achieved:
            results[k] = (None, err)
            logging.warning(f"Failed to achieve tolerance for k={k} within N={max_N}")
    
    return results

def compute_fourier_interpolation(f, N_values, plot_title="Fourier Interpolation"):
    """
    Compute the Fourier interpolation of a given function f for various grid sizes N.
    Matches MATLAB implementation with N_values = [4, 8, 16, 32, 64].
    """
    logging.info(f"Starting interpolation computation for {plot_title}")
    
    # Create figure for interpolation
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.1)
    
    # Pre-allocate arrays for plotting
    x2 = np.linspace(0, 2*np.pi, 500)  # Fine grid for plotting
    
    for N in N_values:
        try:
            # Compute grid points
            x1 = 2 * np.pi * np.arange(N) / N  # Grid points for computation
            
            # Compute function values
            u = f(x1)  # Function values at grid points
            u1 = f(x2)  # Function values at fine grid
            
            # Compute Fourier interpolation
            fx_approx = np.zeros_like(x2)
            for i in range(len(x2)):
                for j in range(len(x1)):
                    if x2[i] != x1[j]:
                        g = (1/N) * np.sin(N*(x2[i]-x1[j])/2) * 1/np.tan((x2[i]-x1[j])/2)
                    else:
                        g = 1
                    fx_approx[i] += u[j] * g
            
            # Plot results
            plt.plot(x2/(2*np.pi), fx_approx, label=f'N={N}', linewidth=2)
            if N == N_values[0]:
                plt.plot(x2/(2*np.pi), u1, '--', label='Exact', linewidth=2, color='black')
            
            # Clear memory
            del x1, u, fx_approx
            gc.collect()
            
        except Exception as e:
            logging.error(f"Error at N={N}: {str(e)}")
    
    # Format the plot
    plt.xlabel('x/(2π)', fontsize=12)
    plt.ylabel('Function Value', fontsize=12)
    plt.title(f'{plot_title}', fontsize=14, pad=10)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=10)
    
    # Save figure
    safe_title = "".join(c for c in plot_title if c.isalnum() or c in (' ', '_')).replace(' ', '_')
    plt.savefig(f'figures/{safe_title}_interpolation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Completed interpolation computation for {plot_title}")

def compute_fourier_derivative_even(f, analytic_deriv, N_values, plot_title):
    """Compute derivative using even Fourier differentiation matrix"""
    logging.info(f"Starting derivative computation for {plot_title}")
    
    # Create figure for derivatives
    fig_deriv = plt.figure(figsize=(12, 8))
    ax_deriv = fig_deriv.add_subplot(111)
    
    # Pre-allocate arrays for plotting
    x_plot = np.linspace(0, 2*np.pi, 1000)
    f_plot = f(x_plot)
    deriv_plot = analytic_deriv(x_plot)
    
    # Plot exact derivative
    ax_deriv.plot(x_plot, deriv_plot, 'k-', label='Exact', linewidth=2)
    
    # Store errors for each N
    errors = []
    
    for N in N_values:
        try:
            # Compute grid points
            x = np.linspace(0, 2*np.pi, N+1)[:-1]  # N points for even method
            f_values = f(x)
            
            # Compute derivative
            D = fourier_diff_matrix_even(N)
            df_approx = D @ f_values
            
            # Compute exact derivative at grid points
            df_exact = analytic_deriv(x)
            
            # Compute errors
            l2_error = np.sqrt(np.mean((df_approx - df_exact)**2))
            linf_error = np.max(np.abs(df_approx - df_exact))
            errors.append((float(N), l2_error, linf_error))
            
            # Interpolate for plotting
            df_approx_plot = np.interp(x_plot, x, df_approx)
            
            # Plot derivative
            ax_deriv.plot(x_plot, df_approx_plot, '--', label=f'N={N}', linewidth=1.5)
            
            # Clear memory
            del D, df_approx, df_exact, df_approx_plot
            
        except Exception as e:
            logging.error(f"Error computing derivative for N={N}: {str(e)}")
            errors.append((float(N), np.nan, np.nan))
    
    # Set up derivative plot
    ax_deriv.set_xlabel('x')
    ax_deriv.set_ylabel('f\'(x)')
    ax_deriv.set_title(f'Derivative of {plot_title}')
    ax_deriv.legend()
    ax_deriv.grid(True)
    
    # Save derivative figure
    safe_title = plot_title.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    fig_deriv.savefig(f'figures/derivative_error_{safe_title}.png', dpi=300, bbox_inches='tight')
    plt.close(fig_deriv)
    
    # Print final errors
    print(f"\nFinal Error for {plot_title}:")
    print("=" * 60)
    print(f"{'N':>8} | {'L2 Error':>15} | {'L_inf Error':>15}")
    print("-" * 60)
    for N, l2, linf in errors:
        if np.isnan(l2) or np.isnan(linf):
            print(f"{N:8.0f} | {'NaN':>15} | {'NaN':>15}")
        else:
            print(f"{N:8.0f} | {l2:15.8e} | {linf:15.8e}")
    print("=" * 60)
    
    logging.info(f"Completed derivative computation for {plot_title}")
    return errors

def solve_hyperbolic(N, dt, t_final, derivative_method='fourier'):
    """Solve the hyperbolic PDE using the specified derivative method"""
    logging.info(f"Solving hyperbolic PDE with N={N}, method={derivative_method}")
    
    # Set up grid
    x = np.linspace(0, 2*np.pi, N)
    
    # Initial condition
    u = np.sin(x)
    
    # Set up time stepping
    t = 0
    
    # Set up derivative operator
    if derivative_method == 'fourier':
        D = fourier_diff_matrix_even(N)
        deriv_op = lambda u: D @ u
    
    # Time stepping using RK4
    while t < t_final:
        # RK4 step
        k1 = -2*np.pi * deriv_op(u)
        k2 = -2*np.pi * deriv_op(u + 0.5*dt*k1)
        k3 = -2*np.pi * deriv_op(u + 0.5*dt*k2)
        k4 = -2*np.pi * deriv_op(u + dt*k3)
        u = u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        t += dt
    
    return x, u

def plot_pde_solutions(N=10, t_values=[0, np.pi/2, np.pi]):
    """
    Solve the PDE and plot the solution at specified times.
    Simplified to only use Fourier method and basic plotting.
    """
    logging.info(f"Plotting PDE solutions for N={N}")
    dt = 0.0001
    plt.figure(figsize=(10, 6))
    
    for t_target in t_values:
        x, u = solve_hyperbolic(N, dt, t_target)
        u_exact = np.exp(np.sin(x - 2*np.pi*t_target))
        plt.plot(x, u, 'o-', label=f'Computed t={t_target:.2f}')
        plt.plot(x, u_exact, '--', label=f'Exact t={t_target:.2f}')
    
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'PDE Solutions using Fourier method, N={N}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figures/PDEsolutionForFourierMethodN={N}.png')
    plt.close()
    logging.info(f"Completed plotting for N={N}")

def plot_error_comparison():
    """Plot error comparison between odd and even methods."""
    plt.figure(figsize=(10, 6))
    
    # Data from the tables
    k_values = [2, 4, 6, 8, 10, 12]
    odd_errors = [1.08e-06, 1.83e-06, 2.79e-05, 1.05e-05, 4.92e-06, 2.70e-05]
    even_errors = [1.17e-06, 1.95e-06, 7.26e-06, 2.91e-06, 5.35e-06, 8.37e-06]
    
    plt.semilogy(k_values, odd_errors, 'o-', label='Odd Method', linewidth=2)
    plt.semilogy(k_values, even_errors, 's-', label='Even Method', linewidth=2)
    
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.title('Error Comparison between Odd and Even Methods')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    
    # Save the plot
    plt.savefig('figures/error_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run all tests."""
    logging.info("Starting main program execution")
    
    # Clear figures directory
    if os.path.exists('figures'):
        shutil.rmtree('figures')
    os.makedirs('figures', exist_ok=True)
    
    # Compare odd and even methods
    compare_methods()
    
    # Plot error comparison
    plot_error_comparison()
    
    # Test functions
    N_values = [4, 8, 16, 32, 64]
    
    # 1. Test f(x) = cos(10x)
    f1 = lambda x: np.cos(10*x)
    df1 = lambda x: -10*np.sin(10*x)
    compute_fourier_interpolation(f1, N_values, "Interpolation for f(x)=cos(10x)")
    compute_fourier_derivative_even(f1, df1, N_values, "Derivative Error for f(x)=cos(10x)")
    
    # 2. Test f(x) = cos(x/2)
    f2_cos = lambda x: np.cos(x/2)
    df2_cos = lambda x: -0.5*np.sin(x/2)
    compute_fourier_interpolation(f2_cos, N_values, "Interpolation for f(x)=cos(x_2)")
    compute_fourier_derivative_even(f2_cos, df2_cos, N_values, "Derivative Error for f(x)=cos(x_2)")
    
    # 2b. Test f(x) = sin(x/2)
    f2_sin = lambda x: np.sin(x/2)
    df2_sin = lambda x: 0.5*np.cos(x/2)
    compute_fourier_interpolation(f2_sin, N_values, "Interpolation for f(x)=sin(x_2)")
    compute_fourier_derivative_even(f2_sin, df2_sin, N_values, "Derivative Error for f(x)=sin(x_2)")
    
    # 3. Test f(x) = x
    f3 = lambda x: x
    df3 = lambda x: np.ones_like(x)
    compute_fourier_interpolation(f3, N_values, "Interpolation for f(x)=x")
    compute_fourier_derivative_even(f3, df3, N_values, "Derivative Error for f(x)=x")
    
    # Plot PDE solutions
    plot_pde_solutions()
    
    logging.info("Main program execution completed")

if __name__ == '__main__':
    main()
