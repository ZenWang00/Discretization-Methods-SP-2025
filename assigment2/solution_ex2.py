#!/usr/bin/env python3
"""
This Python script implements numerical methods for the second question of the homework.
It focuses on computing derivatives of selected functions using the even Fourier differentiation method
and analyzing the pointwise error L_inf and global error L_2 for increasing values of N.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import gc
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('solution.log'),
        logging.StreamHandler()
    ]
)

# Clean and recreate figures directory
if os.path.exists('figures'):
    shutil.rmtree('figures')
Path("figures").mkdir(exist_ok=True)

def fourier_diff_matrix_even(N):
    """
    Construct the Fourier differentiation matrix for the even method.
    
    Args:
        N (int): Number of grid points
        
    Returns:
        numpy.ndarray: The Fourier differentiation matrix
    """
    D = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i,j] = (-1)**(i+j) / 2 * np.cos((i-j)*np.pi/N) / np.sin((i-j)*np.pi/N)
    return D

def compute_fourier_derivative_even(f, df_exact, N_values, plot_title):
    """
    Compute and plot the Fourier derivative using the even method.
    """
    # Create figure for derivative comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Colors for different N values
    colors = ['b', 'r', 'g', 'm', 'c']
    
    for N, color in zip(N_values, colors):
        # Compute grid points
        x = np.array([j * 2 * np.pi / N for j in range(N)])
        
        # Compute function values and exact derivative
        f_values = f(x)
        df_exact_values = df_exact(x)
        
        # Compute numerical derivative
        D = fourier_diff_matrix_even(N)
        df_num = D @ f_values
        
        # Fine grid for plotting
        x_fine = np.linspace(0, 2*np.pi, 1000)
        df_num_fine = np.zeros_like(x_fine)
        df_exact_fine = df_exact(x_fine)
        
        # Improved interpolation with boundary handling
        for i in range(len(x_fine)):
            if x_fine[i] < 0.1 or x_fine[i] > 2*np.pi - 0.1:
                # Special handling near boundaries
                # Use linear interpolation from nearest points
                idx = np.argmin(np.abs(x - x_fine[i]))
                if idx == 0:
                    t = (x_fine[i] - x[0]) / (x[1] - x[0])
                    df_num_fine[i] = (1-t)*df_num[0] + t*df_num[1]
                elif idx == N-1:
                    t = (x_fine[i] - x[-2]) / (x[-1] - x[-2])
                    df_num_fine[i] = (1-t)*df_num[-2] + t*df_num[-1]
            else:
                # Improved interpolation for interior points
                sum_weights = 0
                df_num_fine[i] = 0
                for j in range(N):
                    dx = x_fine[i] - x[j]
                    if abs(dx) < 1e-12:  # Increased precision threshold
                        df_num_fine[i] = df_num[j]
                        break
                    else:
                        # Modified interpolation kernel for better stability
                        sin_term = np.sin(N*dx/2)
                        if abs(sin_term) < 1e-12:  # Handle small values
                            continue
                        tan_term = np.tan(dx/2)
                        if abs(tan_term) < 1e-12:  # Handle small values
                            continue
                        weight = sin_term / (N * tan_term)
                        sum_weights += weight
                        df_num_fine[i] += df_num[j] * weight
                
                if sum_weights != 0:  # Normalize if needed
                    df_num_fine[i] /= sum_weights
        
        # Compute errors
        error = np.abs(df_num_fine - df_exact_fine)
        L2_error = np.sqrt(np.mean((df_num_fine - df_exact_fine)**2))
        Linf_error = np.max(np.abs(df_num_fine - df_exact_fine))
        
        # Plot derivative
        ax1.plot(x_fine, df_num_fine, color + '--', label=f'N={N}', linewidth=1.5)
        
        # Plot error on log scale
        ax2.semilogy(x_fine, error, color + '-', label=f'N={N}', linewidth=1)
        
        # Print errors
        print(f"N = {N:2d} | L2 Error = {L2_error:.8e} | L_inf Error = {Linf_error:.8e}")
    
    # Plot exact derivative
    ax1.plot(x_fine, df_exact_fine, 'k-', label='Exact', linewidth=2)
    
    # Configure plots
    ax1.set_title(f"Derivative comparison - {plot_title}")
    ax1.set_xlabel('x')
    ax1.set_ylabel('df/dx')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title(f"Pointwise error (log scale) - {plot_title}")
    ax2.set_xlabel('x')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True)
    
    # Save figure
    plt.tight_layout()
    safe_title = "".join(x if x.isalnum() else "_" for x in plot_title)
    plt.savefig(f'figures/Derivative_Error_for_fx{safe_title}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run all tests."""
    logging.info("Starting main program execution")
    
    # Clear figures directory
    if os.path.exists('figures'):
        shutil.rmtree('figures')
    Path("figures").mkdir(exist_ok=True)
    
    # Test functions
    N_values = [4, 8, 16, 32, 64]
    
    # 1. Test f(x) = cos(10x)
    f1 = lambda x: np.cos(10*x)
    df1 = lambda x: -10*np.sin(10*x)
    compute_fourier_derivative_even(f1, df1, N_values, "Derivative Error for f(x)=cos(10x)")
    
    # 2. Test f(x) = cos(x/2)
    f2 = lambda x: np.cos(x/2)
    df2 = lambda x: -0.5*np.sin(x/2)
    compute_fourier_derivative_even(f2, df2, N_values, "Derivative Error for f(x)=cos(x_2)")
    
    # 3. Test f(x) = x
    f3 = lambda x: x
    df3 = lambda x: np.ones_like(x)
    compute_fourier_derivative_even(f3, df3, N_values, "Derivative Error for f(x)=x")
    
    logging.info("Main program execution completed")

if __name__ == '__main__':
    main() 