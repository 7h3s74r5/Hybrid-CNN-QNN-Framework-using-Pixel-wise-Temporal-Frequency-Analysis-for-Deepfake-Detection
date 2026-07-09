import torch
import torch.nn as nn

from models.cnn_encoder import CNNEncoder
from models.quantum_layer import QuantumFeatureExtractor


class HybridModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ---------------------------------------------------
        # CNN Encoder
        # ---------------------------------------------------

        self.encoder = CNNEncoder()

        # ---------------------------------------------------
        # Quantum Feature Extractor
        # ---------------------------------------------------

        self.quantum = QuantumFeatureExtractor()

        # ---------------------------------------------------
        # Feature Fusion
        # ---------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(20, 32),

            nn.ReLU(inplace=True),

            nn.Dropout(0.30),

            nn.Linear(32, 16),

            nn.ReLU(inplace=True),

            nn.Dropout(0.20),

            nn.Linear(16, 1)

        )

    def forward(self, x):

        # -----------------------------------------
        # CNN Encoder
        # -----------------------------------------

        x = self.pwtf(x)
        features = self.encoder(x)
        # Ensure features is on the correct device before passing to quantum module
        features = features.to(next(self.parameters()).device)

        # Shape
        # (Batch,64)

        # -----------------------------------------
        # Quantum Module
        # -----------------------------------------

        classical, quantum = self.quantum(features)

        # Shapes
        # classical -> (Batch,16)
        # quantum   -> (Batch,4)

        # -----------------------------------------
        # Concatenate
        # -----------------------------------------

        fused = torch.cat(

            [classical, quantum],

            dim=1

        )

        # Shape
        # (Batch,20)

        # -----------------------------------------
        # Classification
        # -----------------------------------------

        output = self.classifier(fused)

        return output
