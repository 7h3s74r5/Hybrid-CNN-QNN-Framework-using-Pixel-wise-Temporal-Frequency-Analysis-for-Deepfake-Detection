import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# Configuration
# ==========================================================

N_QUBITS = 4
N_Q_LAYERS = 3

# ==========================================================
# Quantum Device
# ==========================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)

# ==========================================================
# Quantum Circuit
# ==========================================================

@qml.qnode(
    dev,
    interface="torch",
    diff_method="backprop"
)
def quantum_circuit(inputs, weights):

    # ------------------------------------
    # L2 Normalization
    # ------------------------------------

    inputs = F.normalize(
        inputs,
        p=2,
        dim=-1
    )

    # ------------------------------------
    # Amplitude Embedding
    # ------------------------------------

    qml.AmplitudeEmbedding(
        features=inputs,
        wires=range(N_QUBITS),
        normalize=True,
        pad_with=0.0
    )

    # ------------------------------------
    # Variational Layers
    # ------------------------------------

    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    # ------------------------------------
    # Measurements
    # ------------------------------------

    return [
        qml.expval(
            qml.PauliZ(i)
        )
        for i in range(N_QUBITS)
    ]

# ==========================================================
# Weight Shapes
# ==========================================================

weight_shapes = {

    "weights": (
        N_Q_LAYERS,
        N_QUBITS,
        3
    )

}

# ==========================================================
# TorchLayer
# ==========================================================

QuantumLayer = qml.qnn.TorchLayer(
    quantum_circuit,
    weight_shapes
)

# ==========================================================
# Feature Compressor
# ==========================================================

class QuantumFeatureExtractor(nn.Module):

    """
    Input :
        (Batch,64)

    Output :
        Classical Features : (Batch,16)
        Quantum Features   : (Batch,4)
    """

    def __init__(self):

        super().__init__()

        self.feature_compressor = nn.Sequential(

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                32,
                16
            )

        )

        self.quantum = QuantumLayer

    def forward(self, x):

        # ------------------------
        # Compress Features
        # ------------------------
        classical = self.feature_compressor(x)

        # ------------------------
        # Quantum Layer
        # ------------------------

        quantum = self.quantum(classical)

        return classical, quantum
