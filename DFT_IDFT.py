import numpy as np 
import pandas as pd
import sys

def QFT_matrix(n_qubits):    #Computational Basis to Fourier Basis

    N = 2 ** n_qubits
    omega = np.exp(2j * np.pi / N)
    # Fill matrix: M[x, k] = omega^(x*k) / sqrt(N)
    M_QFT = np.zeros((N, N), dtype=complex)
    for x in range(N):
        for k in range(N):
            M_QFT[x, k] = omega ** (x * k) / np.sqrt(N)
    #print("Full Matrix=\n",M)
    return M_QFT
    pass

def IQFT_matrix(n_qubits):  # Bring Back Fourier Basis to Compuational Basis
    N = 2**n_qubits
    omega = np.exp(-2j * np.pi / N)
    M_IQFT= np.zeros((N, N), dtype=complex)
    
    for x in range(N):
        for k in range(N):
            M_IQFT[x, k] = omega ** (x * k) / np.sqrt(N)
    return M_IQFT
    pass


def Show_Matrix(n_qubits,matrix,temp):
    M = matrix
    print(M)
    np.set_printoptions(precision=3, suppress=True)
    print("\n "+temp +f" for {n_qubits} qubits (size {2**n_qubits}×{2**n_qubits}):\n")
    print(M)
    pass

def Test_Unitary(Matrix_QFT,Matrix_IQFT,n):  # QFT*IQFT=I   where IQFT is hermition conjugate of QFT 
    print("\n")
    print("Type Matrix_QFT =", type(Matrix_QFT))
    print("Type Matrix_IQFT =", type(Matrix_IQFT))

    print("Shape Q =", Matrix_QFT.shape)
    print("Shape IQ =", Matrix_IQFT.shape)
    product = Matrix_QFT@ Matrix_IQFT

    # Pretty print settings
    np.set_printoptions(precision=3, suppress=True)

    print(f"\nQFT({n}) @ IQFT({n}):\n")
    print(product,"\n")

    # Check closeness to identity
    I = np.eye(2**n)
    print(I,"\n\nIdentity:", np.allclose(product, I))
    pass

def main():
    n_qubits=int(input("Enter the no of bits for Matrix Geneartion="))
    Matrix_QFT=QFT_matrix(n_qubits)
    Matrix_IQFT=IQFT_matrix(n_qubits)
    Show_Matrix(n_qubits,Matrix_QFT,temp="Matrix_QFT")
    Show_Matrix(n_qubits,Matrix_IQFT,temp="Matrix_IQFT")
    Test_Unitary(Matrix_QFT,Matrix_IQFT,n_qubits)
    pass

main()