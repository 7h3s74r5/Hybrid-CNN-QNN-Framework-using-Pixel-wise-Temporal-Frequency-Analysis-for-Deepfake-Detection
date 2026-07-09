import copy

best_model_weights = copy.deepcopy(model.state_dict())

best_val_loss = float("inf")

early_stop_counter = 0

print("="*80)
print("Starting Training...")
print("="*80)

for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    # ------------------------------------
    # Train
    # ------------------------------------

    train_loss, train_acc, _, _ = train_one_epoch(

        model,

        train_loader,

        optimizer,

        criterion,

        scaler,

        device

    )

    # ------------------------------------
    # Validation
    # ------------------------------------

    metrics = validate_one_epoch(

        model,

        val_loader,

        criterion,

        device

    )

    val_loss = metrics["loss"]

    val_acc = metrics["accuracy"]

    # ------------------------------------
    # Scheduler
    # ------------------------------------

    scheduler.step()

    # ------------------------------------
    # Save History
    # ------------------------------------

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    history["precision"].append(metrics["precision"])
    history["recall"].append(metrics["recall"])
    history["f1"].append(metrics["f1"])

    history["roc_auc"].append(metrics["roc_auc"])
    history["pr_auc"].append(metrics["pr_auc"])

    history["mcc"].append(metrics["mcc"])

    # ------------------------------------
    # Print Results
    # ------------------------------------

    print(f"Train Loss : {train_loss:.4f}")

    print(f"Val Loss   : {val_loss:.4f}")

    print(f"Train Acc  : {train_acc:.4f}")

    print(f"Val Acc    : {val_acc:.4f}")

    print(f"Precision  : {metrics['precision']:.4f}")

    print(f"Recall     : {metrics['recall']:.4f}")

    print(f"F1 Score   : {metrics['f1']:.4f}")

    print(f"ROC AUC    : {metrics['roc_auc']:.4f}")

    print(f"PR AUC     : {metrics['pr_auc']:.4f}")

    print(f"MCC        : {metrics['mcc']:.4f}")

    # ------------------------------------
    # Save Best Model
    # ------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_weights = copy.deepcopy(

            model.state_dict()

        )

        torch.save(

            model.state_dict(),

            MODEL_SAVE_PATH

        )

        print("Best Model Saved")

        early_stop_counter = 0

    else:

        early_stop_counter += 1

        print(

            f"EarlyStopping Counter: {early_stop_counter}/{PATIENCE}"

        )

    # ------------------------------------
    # Save Latest Checkpoint
    # ------------------------------------

    checkpoint = {

        "epoch": epoch + 1,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "best_val_loss": best_val_loss

    }

    torch.save(

        checkpoint,

        "last_checkpoint.pth"

    )

    # ------------------------------------
    # Early Stop
    # ------------------------------------

    if early_stop_counter >= PATIENCE:

        print()

        print("="*50)

        print("Early Stopping Activated")

        print("="*50)

        break


# Re-instantiate the model to ensure it's defined in the current session.
# This assumes HybridModel and device are already defined in previous executed cells.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridModel().to(device)

model.load_state_dict(
    best_model_weights
)

print("Best model restored.")

#save best model

model_save_path = "/content/drive/MyDrive/PwTF_QDeepFake/Hybrid_QNN_Final.pth"
model_save_dir = os.path.dirname(model_save_path)
os.makedirs(model_save_dir, exist_ok=True)

torch.save(
    model.state_dict(),
    model_save_path
)
print("Final model saved.")
