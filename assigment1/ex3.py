import numpy as np
import matplotlib.pyplot as plt

def fourier_differentiation_matrix(N):
    """Compute the Fourier differentiation matrix."""
    D = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                D[i,j] = (-1)**(i+j) / (2 * np.sin((j-i)*np.pi/(N+1)))
    return D

def test_fourier_differentiation():
    """Test the accuracy of Fourier differentiation matrix."""
    k_values = [2, 4, 6, 8, 10, 12]
    tol = 1e-5
    min_N_values_relative = []
    min_N_values_absolute = []
    
    for k in k_values:
        # Test with relative error
        N = 2
        max_error = float('inf')
        while max_error > tol:
            N += 2
            D = fourier_differentiation_matrix(N)
            x = 2 * np.pi * np.arange(N+1) / (N+1)
            u = np.exp(k * np.sin(x))
            du_numeric = u @ D
            du_exact = k * np.cos(x) * np.exp(k * np.sin(x))
            relative_error = np.abs(du_numeric - du_exact) / np.abs(du_exact)
            max_error = np.max(relative_error)
        min_N_values_relative.append(N)
        
        # Test with absolute error
        N = 2
        max_error = float('inf')
        while max_error > tol:
            N += 2
            D = fourier_differentiation_matrix(N)
            x = 2 * np.pi * np.arange(N+1) / (N+1)
            u = np.exp(k * np.sin(x))
            du_numeric = u @ D
            du_exact = k * np.cos(x) * np.exp(k * np.sin(x))
            max_error = np.max(np.abs(du_numeric - du_exact))
        min_N_values_absolute.append(N)
    
    print("\nResults with Relative Error:")
    for k, N in zip(k_values, min_N_values_relative):
        print(f"k={k}, minimum N={N}")
    
    print("\nResults with Absolute Error:")
    for k, N in zip(k_values, min_N_values_absolute):
        print(f"k={k}, minimum N={N}")

if __name__ == '__main__':
    test_fourier_differentiation() 