import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_state_city
# ------------------------------matplotlib inline
# Correct QFT and IQFT
# ------------------------------
def QFT_circuit(n):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j+1, n):
            qc.cp(np.pi/2**(k-j), j, k)  # control=j, target=k
    # Swap qubits at the end
    for i in range(n//2):
        qc.swap(i, n-1-i)
    return qc

def IQFT_circuit(n):
    qc = QuantumCircuit(n)
    for i in range(n//2):  # integer Division ignores remainder
        qc.swap(i, n-1-i)  # swap first
    for j in reversed(range(n)):
        for k in reversed(range(j+1, n)):
            qc.cp(-np.pi/2**(k-j), j, k)  # inverse of QFT
        qc.h(j)
    return qc



def main():
    n_qubits=int(input("Enter the no of bits for Matrix Geneartion="))
    qc_init = QuantumCircuit(n_qubits)
  
    qc_init = QuantumCircuit(n_qubits)
    # State_Vector_At_very_begining
    sv_init = Statevector.from_instruction(qc_init)
    print("Bydefault statevector:\n", sv_init)
    plot_bloch_multivector(sv_init)
    plt.show()
    plot_state_city(sv_init)

    # Flip 1st and 3rd qubit
    for i in range(0,n_qubits):
        qc_init.x(0)
        qc_init.x(2)

    # Initial statevector
    sv_init = Statevector.from_instruction(qc_init)
    print("Initial statevector:\n", sv_init)
    plot_bloch_multivector(sv_init)
    plt.show()
    plot_state_city(sv_init)
    plt.show()

    # Apply QFT
    # Apply QFT
    sv_qft = sv_init.evolve(QFT_circuit(n_qubits))
    print("\nAfter QFT:\n", sv_qft)
    #  Visualize (Bloch Sphere)
    plot_bloch_multivector(sv_qft)
    plt.show()
    plot_state_city(sv_qft)
    plt.show()

    
    # Apply IQFT
    sv_final = sv_qft.evolve(IQFT_circuit(n_qubits))
    print("\nAfter IQFT:\n", sv_final)
    #  Visualize (Bloch Sphere)
    plot_bloch_multivector(sv_final)
    plt.show()
    plot_state_city(sv_final)
    plt.show()

    # Check equality
    if np.allclose(sv_init.data, sv_final.data):
        print("\n✅ Circuit proof: QFT * IQFT = Identity")
    else:
        print("\n❌ Circuit proof: QFT * IQFT != Identity")

    pass


main()