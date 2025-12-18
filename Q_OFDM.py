from qiskit import QuantumCircuit, transpile,QuantumRegister
from qiskit.visualization import plot_bloch_multivector, plot_state_city
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_state_city
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT , Diagonal
from qiskit.quantum_info import Kraus
from qiskit.quantum_info import DensityMatrix

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
#Step 1
def Encoding(qc):
    # Encode |00> + |10>
    qc.h(0)
    qc.x(1)
    state_vector_after_encoding= Statevector.from_instruction(qc)
    show_statevec(state_vector_after_encoding, "State vector: After Encoding")
    print(qc.draw('mpl'))
    plt.show()
    return qc,state_vector_after_encoding
    pass
# Step 2
def QFT_After_Encoding(qc):
    qft = QFT(2, do_swaps=True)
    qc.append(qft, [0, 1])
    state_vector_after_QFT= Statevector.from_instruction(qc)
    show_statevec(state_vector_after_QFT, "State vector: After QFT")
    print(qc.draw('mpl'))
    plt.show()
    return qc,state_vector_after_QFT
    pass

#Step 3: Non-ideal quantum channel (α, θ)  Phase distortion (θₖ) — exact match to equations Uθ​=diag(eiθ0​,eiθ1​,eiθ2​,eiθ3​)
#Implements ∣k⟩→eiθk​∣k⟩
def Noisy_Quantum_Channel(qc):
    #theta = [0.0, 0.3, 0.0, -0.3]  # θ0, θ1, θ2, θ3
    theta=float(input("Enter the theta="))
    theta = [0.0, theta, 0.0, -theta]
    phase_unitary = Diagonal([
    np.exp(1j * theta[0]), # |00>
    np.exp(1j * theta[1]), # |01>
    np.exp(1j * theta[2]), # |10>
    np.exp(1j * theta[3]), # |11>
])
    qc.append(phase_unitary, [0, 1])
    state_vector_after_Noisy_Quantum_Channel= Statevector.from_instruction(qc)
    show_statevec(state_vector_after_Noisy_Quantum_Channel, "State vector: Information passes through after Noisy Quantum Channel")
    print(qc.draw('mpl'))
    plt.show()
    return qc,state_vector_after_Noisy_Quantum_Channel
    pass

def Amplitude_Loss_Non_Unitary_Kraus_operator(qc):
    #gamma = 0.2  # loss strength (controls α)  ,
    gamma=float(input("Enter the gamma="))
    #Amplitude loss is non-unitary, so we use Kraus operators
    if gamma>0:
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])

        amp_damp = Kraus([K0, K1])  #Use amplitude damping ,where α ≈ squareroot(1−γ)
        #Apply independently to both qubits
        qc.append(amp_damp, [0])
        qc.append(amp_damp, [1])

    #state_vector_after_Amplitude_Loss= Statevector.from_instruction(qc)
    #show_statevec(state_vector_after_Amplitude_Loss, "State vector: After_Amplitude_Loss")   # cant use it because it is density matrix mixed state where as sate Vector is a pure statewhich ins unitary
    print(qc.draw('mpl'))
    plt.show()
    rho_after_loss = DensityMatrix.from_instruction(qc)
    print(rho_after_loss)
    #state_vector_after_Amplitude_Loss = Statevector.from_instruction(qc)
    return qc 
    pass

def IQFT_After_Noisy_Quantum_Channel(qc):
    iqft = QFT(2, inverse=True, do_swaps=True)
    qc.append(iqft, [0, 1])
    #state_vector_after_IQFT= Statevector.from_instruction(qc)
    #show_statevec(state_vector_after_IQFT, "State vector: After_IQFT")
    print(qc.draw('mpl'))
    plt.show()
    return qc
    pass

def Measurement(qc):
    #qc.measure_all()
    qc.measure(qc.qubits, qc.clbits)
    print(qc.draw('mpl'))
    plt.show()
    
    sim = AerSimulator()
    job = sim.run(transpile(qc, sim), shots=1024)
    counts = job.result().get_counts()
    print(counts)
    #counts = result.get_counts(qc)  # qc is the circuit object
    print("Measurement results:")
    for outcome, count in counts.items():
        print(outcome, count)
    plot_histogram(counts)
    plt.show()
    return counts
    pass

def QBER_Estimation(counts,shots):
    error_counts = sum(v for k, v in counts.items() if k not in ['00', '10'])
    qber = error_counts / shots
    print("QBER=",qber)
    # Parameter ranges
    theta = np.linspace(0, np.pi, 120)
    gamma = np.linspace(0, 1, 120)

    Theta, Gamma = np.meshgrid(theta, gamma)

    # QBER formula
    QBER = (1 - Gamma) * np.sin(Theta / 2)**2 + Gamma / 2

    # Security threshold
    QBER_th = 0.11
    QBER_plane = QBER_th * np.ones_like(QBER)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # QBER surface
    surf = ax.plot_surface(
        Theta, Gamma, QBER,
        cmap='viridis',
        alpha=0.85,
        edgecolor='none'
    )

    # Threshold plane
    ax.plot_surface(
        Theta, Gamma, QBER_plane,
        color='red',
        alpha=0.35
    )

    ax.set_xlabel("Phase noise θ (radians)")
    ax.set_ylabel("Amplitude damping γ")
    ax.set_zlabel("QBER")

    ax.set_title("QBER Surface with QKD Security Threshold (QBER = 11%)")

    fig.colorbar(surf, shrink=0.5, aspect=10, label="QBER")

    plt.show()
    return qber
    pass


def main():
    QBER=[]
    #for i in range(0,1):
    qc = QuantumCircuit(2, 2)
    state_vector= Statevector.from_instruction(qc)
    show_statevec(state_vector, "Initial State Vector")
    print(qc.draw('mpl'))
    plt.show()
    qc,state_vector_after_encoding=Encoding(qc)
    qc,state_vector_after_QFT=QFT_After_Encoding(qc)
    qc,state_vector_after_Noisy_Quantum_Channel=Noisy_Quantum_Channel(qc)
    qc=Amplitude_Loss_Non_Unitary_Kraus_operator(qc)
    qc=IQFT_After_Noisy_Quantum_Channel(qc)
    counts=Measurement(qc)
    shots=4096
    qber=QBER_Estimation(counts,shots)
    QBER.append(qber)
    print(QBER)
 
    pass

main()