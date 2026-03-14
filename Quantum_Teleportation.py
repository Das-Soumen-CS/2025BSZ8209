from qiskit import QuantumCircuit, transpile,QuantumRegister
from qiskit.visualization import plot_bloch_multivector, plot_state_city
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import numpy as np
from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_state_city
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram


def show_statevector(sv, title=""):
    print("\n" + "="*70)
    print(title)
    print("="*70)
    
    # Print non-zero amplitudes
    for i, amp in enumerate(sv.data):
        if abs(amp) > 1e-6:
            print(f"|{i:010b}>  {amp}")
            
    # Plot Bloch vectors
    fig = plot_bloch_multivector(sv)
    fig.suptitle(title)
    plt.show()


# STEP 2 :Prepare_Alice_Unknown_state that to be teleported between Alice Lab to Bob's Lab
def Prepare_Alice_Unknown_state(qc):
    qc.h(0)
    qc.barrier(label="Alice Superposition")
    # Simulate
    state_vector= Statevector.from_instruction(qc)
    show_statevector(state_vector, "After Superposition:")
    plt.show()
    print(qc.draw('mpl'))
    plt.show()
    return qc ,state_vector
    pass

# STEP 3 ,Prepare EPR(Bell pair) among Alice and Bob using qubit 1(kept by Alice) and qubit 2(kept by Bob)
def create_EPR_pair_Alice_Bob(qc): 
    qc.h(1) 
    qc.cx(1,2)
    qc.barrier(label="EPR pair")
    # Simulate initial state
    state_vector= Statevector.from_instruction(qc)
    show_statevector(state_vector, "After Entanglement:")
    plt.show()
    print(qc.draw('mpl'))
    plt.show()
    return qc ,state_vector
    pass

# STEP 4 ,Alices Bell State Measurement(BSM)
def Alice_BSM(qc):
    qc.cx(0,1)
    qc.h(0)
     # Simulate initial state
    state_vector= Statevector.from_instruction(qc)
    show_statevector(state_vector, "After BSM:")
    plt.show()
    print(qc.draw('mpl'))
    plt.show()
    #Alice measure
    qc.barrier(label="Alice Measure")
    qc.measure(0,0)
    qc.measure(1,1)
    qc.draw('mpl')
    #print(qc.draw('mpl'))
    plt.show()
    return qc ,state_vector
    pass

# STEP 5 ,Bob Unitary pauli correction on his qubit 1  depending on Alice’s results comming form qubit 0 and qubit 1 (00 =(Idenity)/01=>Bit flip (pauli X)/10=>phase flip (pauli Z)/11 =>Bit flip (pauli X)+phase flip (pauli Z))
def Bob_Unitary_Pauli_Correction(qc):
    qc.barrier(label=" Bob Pauli-correction")
    qc.cx(1,2)  # Apply CNOT (Bit flip) on qubit_1 and qubit_2
    qc.cz(0,2)  # Apply pauli Z (phase flip) on qubit_0 and qubit_2
    qc.draw('mpl')
    #print(qc.draw('mpl'))
    plt.show()     
    return qc 
    pass

def simualte(qc):
    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    job = simulator.run(compiled, shots=1024)
    result = job.result()
    counts = result.get_counts()
    print(counts)
    plot_histogram(counts)
    plt.show()
    pass


def main():
    #Create Initial Circuit (STEP 1)
    qc = QuantumCircuit(3, 2)   # 3 qubits and 2 classical bits  ,#q0 → state to teleport , #q1 → Alice's entangled qubit ,#q2 → Bob's qubit
    qc.barrier(label="initial")
    # Simulate initial state
    state_vector= Statevector.from_instruction(qc)
    show_statevector(state_vector, "Initial:")
    plt.show()
    print(qc.draw('mpl'))
    plt.show()
    qc ,state_vector=Prepare_Alice_Unknown_state(qc)
    qc ,state_vector=create_EPR_pair_Alice_Bob(qc)
    qc ,state_vector=Alice_BSM(qc)
    qc=Bob_Unitary_Pauli_Correction(qc)
    simualte(qc)
    pass

main()