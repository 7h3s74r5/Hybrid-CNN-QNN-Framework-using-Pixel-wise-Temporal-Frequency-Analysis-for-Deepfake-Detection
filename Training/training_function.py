from tqdm import tqdm
import torch
from torch.amp import autocast

# ==========================================================
# Training Function
# ==========================================================

def train_one_epoch(

    model,
    train_loader,
    optimizer,
    criterion,
    scaler,
    device

):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    predictions = []

    targets = []

    progress = tqdm(

        train_loader,

        desc="Training",

        leave=False

    )

    for features, labels in progress:

        features = features.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        # -----------------------------------------
        # Mixed Precision Forward Pass
        # -----------------------------------------

        with autocast(device_type="cuda"):

            outputs = model(features).squeeze(1)

            loss = criterion(

                outputs,

                labels

            )

        # -----------------------------------------
        # Backpropagation
        # -----------------------------------------

        scaler.scale(loss).backward()

        # -----------------------------------------
        # Gradient Clipping
        # -----------------------------------------

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0

        )

        scaler.step(optimizer)

        scaler.update()

        # -----------------------------------------
        # Statistics
        # -----------------------------------------

        running_loss += loss.item()

        probs = torch.sigmoid(outputs)

        preds = (probs >= 0.5).float()

        correct += (

            preds == labels

        ).sum().item()

        total += labels.size(0)

        predictions.extend(

            preds.detach().cpu().numpy()

        )

        targets.extend(

            labels.detach().cpu().numpy()

        )

        progress.set_postfix(

            loss=f"{loss.item():.4f}",

            acc=f"{100*correct/total:.2f}%"

        )

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = correct / total

    return (

        epoch_loss,

        epoch_acc,

        predictions,

        targets

    )
