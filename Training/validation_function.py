def validate_one_epoch(

    model,

    val_loader,

    criterion,

    device

):

    model.eval()

    running_loss = 0.0

    all_labels = []

    all_predictions = []

    all_probabilities = []

    with torch.no_grad():

        progress = tqdm(

            val_loader,

            desc="Validation",

            leave=False

        )

        for features, labels in progress:

            features = features.to(device)

            labels = labels.to(device)

            outputs = model(features).squeeze(1)

            loss = criterion(

                outputs,

                labels

            )

            running_loss += loss.item()

            probabilities = torch.sigmoid(outputs)

            predictions = (

                probabilities >= 0.5

            ).float()

            all_labels.extend(

                labels.cpu().numpy()

            )

            all_predictions.extend(

                predictions.cpu().numpy()

            )

            all_probabilities.extend(

                probabilities.cpu().numpy()

            )

    val_loss = running_loss / len(val_loader)

    accuracy = accuracy_score(

        all_labels,

        all_predictions

    )

    precision = precision_score(

        all_labels,

        all_predictions,

        zero_division=0

    )

    recall = recall_score(

        all_labels,

        all_predictions,

        zero_division=0

    )

    f1 = f1_score(

        all_labels,

        all_predictions,

        zero_division=0

    )

    roc_auc = roc_auc_score(

        all_labels,

        all_probabilities

    )

    pr_auc = average_precision_score(

        all_labels,

        all_probabilities

    )

    mcc = matthews_corrcoef(

        all_labels,

        all_predictions

    )

    cm = confusion_matrix(

        all_labels,

        all_predictions

    )

    report = classification_report(

        all_labels,

        all_predictions,

        digits=4

    )

    return {

        "loss": val_loss,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc,

        "pr_auc": pr_auc,

        "mcc": mcc,

        "confusion_matrix": cm,

        "report": report,

        "labels": np.array(all_labels),

        "predictions": np.array(all_predictions),

        "probabilities": np.array(all_probabilities)

    }
