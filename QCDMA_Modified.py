from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.circuit.library import QFT
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit.visualization import plot_bloch_multivector, plot_state_city
import numpy as np

# To show State vector and corresponding qubit position in Bloch Sphere
def show_statevec(sv, title=""):
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

def Create_circuit():
    qr = QuantumRegister(10, "q")
    cr = ClassicalRegister(4, "c")   # For M1..M4
    qc = QuantumCircuit(qr, cr)

    # Simulate initial state
    state_vector= Statevector.from_instruction(qc)
    '''plot_bloch_multivector(state_vector)
    plt.show()'''
    show_statevec(state_vector, "Initial: |0000000000>")
    print(qc.draw('mpl'))
    plt.show()
    return qc

def Prepare_Initial_State(qc):
    # H on D1
    qc.h(0)

    # C1y = 1, C2y = 1
    qc.x(3)
    qc.x(5)

    state_vector_original= Statevector.from_instruction(qc)
    '''plot_bloch_multivector(state_vector)
    plt.show()'''
    print(qc.draw('mpl'))
    plt.show()
    show_statevec(state_vector_original, "After Step 2: Prepare Input (H on D1, Puali X on C1y & C2y)")
    return qc,state_vector_original
    pass

def CDMA_Encoding(qc):
    qc.cx(0, 2)
    qc.cx(0, 4)
    qc.cx(1, 3)
    qc.cx(1, 5)
    sv_before_QFT_CDMA_Encoding = Statevector.from_instruction(qc)
    show_statevec(sv_before_QFT_CDMA_Encoding, "After Step 3: CDMA Encoding")
    print(qc.draw('mpl'))
    plt.show()
    return qc,sv_before_QFT_CDMA_Encoding
    pass

def QFT_Block(qc):
    #QFT operation
    qft4 = QFT(4, do_swaps=False)
    qc.append(qft4, [2,3,4,5])
    sv_after_QFT= Statevector.from_instruction(qc)
    show_statevec(sv_after_QFT, "After Step 4: QFT applied to 4 code qubits (C1x,C1y,C2x,C2y)")
    print(qc.draw('mpl'))
    plt.show()
    return qc,sv_after_QFT,qft4

def IQFT_Block(qc,qft4):
    #QFT operation
    qc.append(qft4.inverse(), [2,3,4,5])
    sv_after_IQFT= Statevector.from_instruction(qc)
    show_statevec(sv_after_IQFT, "After Step 5: Inverse QFT")
    print(qc.draw('mpl'))
    plt.show()
    return qc,sv_after_IQFT

def CDMA_Decoding(qc):
    # CDMA Decode Apply CNOT again as like Encode because: CNOT(CNOT)=original
    qc.cx(0, 2)
    qc.cx(0, 4)
    qc.cx(1, 3)
    qc.cx(1, 5)

    sv_CDMA_Decoded = Statevector.from_instruction(qc)
    show_statevec(sv_CDMA_Decoded, "After Step 7: CDMA Decoding")
    print(qc.draw('mpl'))
    plt.show()
    return qc,sv_CDMA_Decoded

def Measure_Last_4_qubits(qc):
    #qc.measure([2,3,4,5], [0,1,2,3])
    qc.measure(2,0)
    qc.measure(3,1)
    qc.measure(4,2)
    qc.measure(5,3)
    print(qc.draw('mpl'))
    plt.show()
    # To Simulate and Run
    sim = AerSimulator()
    job = sim.run(transpile(qc, sim), shots=1024)
    counts = job.result().get_counts()

    print("\nMeasurement results of (C1x,C1y,C2x,C2y):")
    print(counts)
    return qc

def main():
    qc_init= Create_circuit()
    qc_prepare,state_vector_original=Prepare_Initial_State(qc_init)
    qc_cdma_encoded,sv_before_QFT_CDMA_Encoding=CDMA_Encoding(qc_prepare)
    qc_qft,sv_after_QFT,qft4=QFT_Block(qc_cdma_encoded)
    qc_iqft,sv_after_IQFT=IQFT_Block(qc_qft,qft4)
    qc_cdma_decoded,sv_CDMA_Decoded=CDMA_Decoding(qc_iqft)
    qc_final=Measure_Last_4_qubits(qc_cdma_decoded)


    # Check equality QFT * IQFT = Identity"
    if np.allclose(sv_before_QFT_CDMA_Encoding.data, sv_after_IQFT.data):
        print("\n✅ Circuit proof: QFT * IQFT = Identity")
    else:
        print("\n❌ Circuit proof: QFT * IQFT != Identity")
        
    # Check equality CDMA_Encoding * CDMA_Decoding  = Identity"
    if np.allclose(state_vector_original.data, sv_CDMA_Decoded.data):
        print("\n✅ Circuit proof: CDMA_Encoding * CDMA_Decoding  = Identity")
    else:
        print("\n❌ Circuit proof:CDMA_Encoding * CDMA_Decoding  != Identity")
    pass

main()