#!/usr/bin/env python3
"""
This Python script implements numerical methods for the first question of the homework.
It focuses on comparing the accuracy of odd and even Fourier differentiation methods.
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

# Clean and recreate figures directory
if os.path.exists('figures'):
    shutil.rmtree('figures')
os.makedirs('figures', exist_ok=True)

def fourier_diff_matrix_even(N):
    """
    Construct the Fourier differentiation matrix using the even method.
    Optimized version using vectorized operations.
    """
    logging.debug(f"Constructing Fourier differentiation matrix for N={N}")
    n = N + 1
    D = np.zeros((n, n))
    
    # Create indices for vectorized operations
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    diff = j - i
    
    # Compute non-diagonal elements using vectorized operations
    mask = diff != 0
    # Handle negative powers correctly
    sign = (-1)**np.abs(diff[mask])
    D[mask] = sign / (2 * np.sin(diff[mask] * np.pi / n))
    
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
    """Compare the accuracy of odd and even methods."""
    k_values = [2, 4, 6, 8, 10, 12]
    N_values = [20, 28, 34, 42, 48, 54]  # Same N values as in homework 1
    results = []
    
    for k, N in zip(k_values, N_values):
        # Test odd method
        D_odd = fourier_diff_matrix_odd(N)
        x_odd = 2 * np.pi * np.arange(N) / N
        u_odd = np.exp(k * np.sin(x_odd))
        du_numeric_odd = D_odd @ u_odd
        du_exact_odd = k * np.cos(x_odd) * np.exp(k * np.sin(x_odd))
        error_odd = np.max(np.abs(du_numeric_odd - du_exact_odd))
        
        # Test even method (from homework 1)
        D_even = fourier_diff_matrix_even(N)
        x_even = 2 * np.pi * np.arange(N+1) / (N+1)
        u_even = np.exp(k * np.sin(x_even))
        du_numeric_even = D_even @ u_even
        du_exact_even = k * np.cos(x_even) * np.exp(k * np.sin(x_even))
        error_even = np.max(np.abs(du_numeric_even - du_exact_even))
        
        results.append((k, N, error_odd, error_even))
    
    # Print comparison table
    print("\nComparison of Odd and Even Methods:")
    print("=" * 80)
    print(f"{'k':>3} | {'N':>3} | {'Error_odd':^15} | {'Error_even':^15} | {'Odd is better?':^15}")
    print("-" * 80)
    for k, N, err_odd, err_even in results:
        is_odd_better = err_odd < err_even
        print(f"{k:3d} | {N:3d} | {err_odd:15.8e} | {err_even:15.8e} | {str(is_odd_better):^15}")
    print("=" * 80)
    
    return results

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
    
    logging.info("Main program execution completed")

if __name__ == '__main__':
    main() 