# Cell 2: imports and helper utilities
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_bloch_multivector, plot_state_city
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram

def Derive_State_Vector(sv, title=""):
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
    pass


def N_users():
    # Existing list
    bits = []
    # Ask user for input for one user
    N=int(input("Enter the No of user=: "))
    for i in range(0,N):
        a = int(input("Enter first bit (a): "))
        b = int(input("Enter second bit (b): "))
        # Create tuple and append
        bits.append((a, b))
    print("User Input for", N, "users=",bits)
    return N,bits
    pass

def Create_circuit(N):
    #  STEP 1: CREATE REGISTERS
    # --------------------------
    q = QuantumRegister(2*N, 'q')  # 2 qubits per user: sender + receiver
    c = ClassicalRegister(2*N, 'c')
    qc = QuantumCircuit(q, c)
    state_vector_Initial= Statevector.from_instruction(qc)
    Derive_State_Vector(state_vector_Initial, title=f"Inital State Vector for {N} users")
    # STEP 2: Create Bell pairs
    for i in range(N):
        qc.h(q[2*i])           # Hadamard on sender qubit
        qc.cx(q[2*i], q[2*i+1])  # CNOT to receiver qubit
    #Draw Circuit
    print("...After Bell Pair Creation...")
    print(qc.draw('mpl'))
    plt.show()
    #Sate vector after creating Bell pairs
    state_vector_after_bell_pair= Statevector.from_instruction(qc)
    Derive_State_Vector(state_vector_after_bell_pair, title="state_vector_after_creating_Bell_pair")
    return q,c,qc,state_vector_after_bell_pair
    pass

def Super_Dense_Coding(q,qc,bits):
   # STEP 3: Superdense encoding
    for i, (a,b) in enumerate(bits):
        sender = q[2*i]
        if a == 1:
            qc.z(sender)
        if b == 1:
            qc.x(sender)
    #Draw Circuit
    print("...After Super_Dense_Coding...")
    print(qc.draw('mpl'))
    plt.show()
     #State vector after Super Dense Coding
    State_vector_after_Super_Dense_Coding= Statevector.from_instruction(qc)
    Derive_State_Vector(State_vector_after_Super_Dense_Coding, title="State vector after Super Dense Coding")
    return qc, q,State_vector_after_Super_Dense_Coding
    pass

def QFT_IQFT_Block(q,qc,N):
    statevector_before_qft= Statevector.from_instruction(qc)
    # STEP 4: Apply QFT for multiplexing
    qft = QFT(num_qubits=N, do_swaps=False).to_gate()
    qft.name = "QFT"
    qc.append(qft, [q[2*i] for i in range(N)])  # QFT on sender qubits
    statevector_after_qft= Statevector.from_instruction(qc)
    print("...Circuit upto After QFT...")
    Derive_State_Vector(statevector_after_qft, title="State vector after QFT")
    #Draw Circuit
    print("...After QFT...")
    print(qc.draw('mpl'))
    plt.show()
    # STEP 5: Apply Inverse QFT (IQFT)
    iqft = QFT(num_qubits=N, do_swaps=False).inverse().to_gate()
    iqft.name = "IQFT"
    qc.append(iqft, [q[2*i] for i in range(N)])  # IQFT on sender qubits
    statevector_after_iqft = Statevector.from_instruction(qc)
    print("...Circuit After IQFT...")
    Derive_State_Vector(statevector_after_iqft, title="State vector after IQFT")
    #Draw Circuit
    print("...After IQFT...")
    print(qc.draw('mpl'))
    plt.show()
    # Check equality
    if np.allclose(statevector_before_qft.data, statevector_after_iqft.data):
        print("\n✅ Circuit proof: QFT * IQFT = Identity")
    else:
        print("\n❌ Circuit proof: QFT * IQFT != Identity")
    return q,qc 

def Super_Dense_Decoding(qc,q,N):
    for i in range(N):
        qc.cx(q[2*i], q[2*i+1])
        qc.h(q[2*i])
    statevector_after_super_dense_decode = Statevector.from_instruction(qc)
    print("...Circuit After Super Dense Decode...")
    Derive_State_Vector(statevector_after_super_dense_decode , title="statevector_after_super_dense_decode ")
    return q,qc,statevector_after_super_dense_decode
    pass
def Measure(q,c,qc,N):
    for i in range(2*N):
        qc.measure(q[i], c[i])
    #Draw Circuit
    print("...After Measure...")
    print(qc.draw('mpl'))
    plt.show()
    # Run simulation
    # --------------------------
    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=1024).result()
    counts = result.get_counts()

    print("Measurement results (bitstrings):")
    print(counts)
    plot_histogram(counts)
    plt.show()

    # --------------------------
    # Recover classical messages per user
    # --------------------------
    # Measurement order: [top0,bottom0, top1,bottom1,...]
    recovered = []
    for i in range(N):
        bit_sender = None
        bit_receiver = None
        # Find most probable bit (majority vote from counts)
        # Here we just take first key in counts for simplicity
        key = list(counts.keys())[0]
        bit_sender = int(key[2*i])      # top qubit
        bit_receiver = int(key[2*i+1])  # bottom qubit
        recovered.append((bit_receiver, bit_sender))  # reversed per user

    # Optional: reverse entire user order
    recovered = recovered[::-1]

    print("Recovered classical messages (reversed user order):")
    print(recovered)
    pass


def main():
    N,bits=N_users()
    q,c,qc,state_vector_after_bell_pair=Create_circuit(N)
    qc,q,State_vector_after_Super_Dense_Coding=Super_Dense_Coding(q,qc,bits)
    q,qc=QFT_IQFT_Block(q,qc,N)
    q,qc,statevector_after_super_dense_decode=Super_Dense_Decoding(qc,q,N)
    Measure(q,c,qc,N)
    pass


main()