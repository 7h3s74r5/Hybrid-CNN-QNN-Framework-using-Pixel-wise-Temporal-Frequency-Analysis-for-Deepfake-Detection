"""
===========================================================
Training Setup
===========================================================
"""

import os
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.cuda.amp import GradScaler

from models.hybrid_model import HybridModel


# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device :", device)


# =====================================================
# Hyperparameters
# =====================================================

EPOCHS = 30

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 8

MODEL_SAVE_PATH = "best_model.pth"


# =====================================================
# Build Model
# =====================================================

model = HybridModel().to(device)


# =====================================================
# Class Weights
# =====================================================

# Replace these numbers if your dataset changes

num_real = 3224

num_fake = 846

pos_weight = torch.tensor(

    [num_real / num_fake],

    dtype=torch.float32

).to(device)

print("Positive Weight :", pos_weight.item())


# =====================================================
# Loss Function
# =====================================================

criterion = nn.BCEWithLogitsLoss(

    pos_weight=pos_weight

)


# =====================================================
# Optimizer
# =====================================================

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# =====================================================
# Scheduler
# =====================================================

scheduler = CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS,

    eta_min=1e-6

)


# =====================================================
# Mixed Precision
# =====================================================

scaler = GradScaler()


# =====================================================
# Early Stopping Variables
# =====================================================

best_val_loss = float("inf")

early_stop_counter = 0


# =====================================================
# Display Configuration
# =====================================================

print("=" * 60)

print("Model           :", model.__class__.__name__)

print("Epochs          :", EPOCHS)

print("Batch Size      :", BATCH_SIZE)

print("Learning Rate   :", LEARNING_RATE)

print("Weight Decay    :", WEIGHT_DECAY)

print("Scheduler       : CosineAnnealingLR")

print("Loss Function   : BCEWithLogitsLoss")

print("Mixed Precision : Enabled")

print("Device          :", device)

print("=" * 60)
