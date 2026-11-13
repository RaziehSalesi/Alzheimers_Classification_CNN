
# ============================================================
# main.py
# Alzheimer Classification using CNN - ADNI2
# ============================================================

import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset import (
    load_image_paths,
    split_dataset,
    Data_Augmentation,
    Prepare_Images,
    AdniDataset
)

from models import (
    CNN_Mode_Proposed,
    CNN_Model_Test2,
    CNN_Model_Test3
)


# ============================================================
# 1. CONFIGURATION

# chnage the path to your data directory
# It should contain two folders: AD and MCI.

DATA_DIR = Path("Alzheimer_s_Disease_Neuroimaging_ADNI_Dataset")

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.0001

RANDOM_SEED = 42

OUTPUT_DIR = Path("results")
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY AND DEVICE

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. ACCURACY FUNCTION

def accuracy_fn(pred, true):

    pred = torch.argmax(pred, dim=1)

    acc = (true == pred).float().mean().item()

    return acc


# ============================================================
# 4. LOAD AND SPLIT DATASET

print("\nLoading ADNI2 dataset...")

image_paths, labels = load_image_paths(DATA_DIR)

print("Total original images:", len(image_paths))

print("\nSplitting original images...")

(
    X_train_paths,
    X_val_paths,
    X_test_paths,
    y_train,
    y_val,
    y_test
) = split_dataset(image_paths, labels)


print("Original training images:", len(X_train_paths))
print("Original validation images:", len(X_val_paths))
print("Original test images:", len(X_test_paths))


# ============================================================
# 5. OFFLINE DATA AUGMENTATION

print("\nApplying offline augmentation to training images...")

X_train, y_train = Data_Augmentation(
    X_train_paths,
    y_train
)

# Validation and test images are NOT augmented.

X_val = Prepare_Images(X_val_paths)
X_test = Prepare_Images(X_test_paths)


print("\nDataset sizes after augmentation:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# ============================================================
# 6. IMAGE NORMALIZATION

# Convert pixel values from [0, 255] to [0, 1].
# This preserves the normalization used in your notebook.

X_train = X_train.astype(np.float32) / 255.0
X_val = X_val.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0


# ============================================================
# 7. CREATE DATASETS AND DATALOADERS

train_dataset = AdniDataset(X_train, y_train)
val_dataset = AdniDataset(X_val, y_val)
test_dataset = AdniDataset(X_test, y_test)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Check image dimensions

sample_images, sample_labels = next(iter(train_loader))

print("\nInput shape:", sample_images.shape)
print("Label shape:", sample_labels.shape)

assert sample_images.ndim == 4
assert sample_images.shape[1:] == (1, 64, 64)


# ============================================================
# 8. TRAINING FUNCTION

def train_model(model, model_name):

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = {
        "train_accuracy": [],
        "val_accuracy": [],
        "train_sample_accuracy": [],
        "val_sample_accuracy": [],
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(1, EPOCHS + 1):

        sum_train_acc_epoch = 0
        sum_val_acc_epoch = 0

        sum_train_loss_epoch = 0
        sum_val_loss_epoch = 0

        train_correct = 0
        train_total = 0

        val_correct = 0
        val_total = 0

        # ---------------- TRAINING ----------------

        model.train()

        for batch_idx, (images, labels) in enumerate(
            train_loader, 1
        ):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            labels_pred = model(images.float())

            loss = loss_fn(labels_pred, labels)

            loss.backward()
            optimizer.step()

            # Original batch-average accuracy
            acc1 = accuracy_fn(labels_pred, labels)

            sum_train_acc_epoch += acc1
            sum_train_loss_epoch += loss.detach().item()

            # Additional sample-based accuracy
            predicted = torch.argmax(labels_pred, dim=1)

            train_correct += (
                predicted == labels
            ).sum().item()

            train_total += labels.size(0)

            print_interval = max(
                1, int(round(len(train_loader) / 6))
            )

            if batch_idx % print_interval == 0:

                print(
                    f"Epoch {epoch} | Train batch {batch_idx} | "
                    f"Batch-average accuracy: "
                    f"{(sum_train_acc_epoch / batch_idx) * 100:.2f}%"
                )

                print(
                    f"Epoch {epoch} | Train batch {batch_idx} | "
                    f"Sample-based accuracy: "
                    f"{(train_correct / train_total) * 100:.2f}%"
                )

        # Store training results

        history["train_accuracy"].append(
            sum_train_acc_epoch / batch_idx
        )

        history["train_sample_accuracy"].append(
            train_correct / train_total
        )

        history["train_loss"].append(
            sum_train_loss_epoch / batch_idx
        )

        # ---------------- VALIDATION ----------------

        model.eval()

        with torch.no_grad():

            for batch_idx, (images, labels) in enumerate(
                val_loader, 1
            ):

                images = images.to(device)
                labels = labels.to(device)

                labels_pred = model(images.float())

                loss = loss_fn(labels_pred, labels)

                # Original batch-average accuracy
                acc1 = accuracy_fn(labels_pred, labels)

                sum_val_acc_epoch += acc1
                sum_val_loss_epoch += loss.detach().item()

                # Additional sample-based accuracy
                predicted = torch.argmax(labels_pred, dim=1)

                val_correct += (
                    predicted == labels
                ).sum().item()

                val_total += labels.size(0)

                print_interval = max(
                    1, int(round(len(val_loader) / 6))
                )

                if batch_idx % print_interval == 0:

                    print(
                        f"Epoch {epoch} | Validation batch {batch_idx} | "
                        f"Batch-average accuracy: "
                        f"{(sum_val_acc_epoch / batch_idx) * 100:.2f}%"
                    )

                    print(
                        f"Epoch {epoch} | Validation batch {batch_idx} | "
                        f"Sample-based accuracy: "
                        f"{(val_correct / val_total) * 100:.2f}%"
                    )

        # Store validation results

        history["val_accuracy"].append(
            sum_val_acc_epoch / batch_idx
        )

        history["val_sample_accuracy"].append(
            val_correct / val_total
        )

        history["val_loss"].append(
            sum_val_loss_epoch / batch_idx
        )

        print(f"\nEpoch {epoch}/{EPOCHS} completed")
        print(
            f"Training sample-based accuracy: "
            f"{history['train_sample_accuracy'][-1] * 100:.2f}%"
        )
        print(
            f"Validation sample-based accuracy: "
            f"{history['val_sample_accuracy'][-1] * 100:.2f}%"
        )
        print("-" * 50)

    # Save model weights

    model_path = MODEL_DIR / f"{model_name}.pth"

    torch.save(model.state_dict(), model_path)

    print(f"Saved model weights: {model_path}")

    return model, history


# ============================================================
# 9. PLOT TRAINING AND VALIDATION CURVES

def plot_training_curves(history, model_name):

    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss

    axes[0].plot(
        epochs_range,
        history["train_loss"],
        label="Training Loss"
    )

    axes[0].plot(
        epochs_range,
        history["val_loss"],
        label="Validation Loss"
    )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid()

    # Accuracy

    axes[1].plot(
        epochs_range,
        history["train_accuracy"],
        label="Training Accuracy (Batch Average)"
    )

    axes[1].plot(
        epochs_range,
        history["val_accuracy"],
        label="Validation Accuracy (Batch Average)"
    )

    axes[1].plot(
        epochs_range,
        history["train_sample_accuracy"],
        linestyle="--",
        label="Training Accuracy (Sample Based)"
    )

    axes[1].plot(
        epochs_range,
        history["val_sample_accuracy"],
        linestyle="--",
        label="Validation Accuracy (Sample Based)"
    )

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid()

    plt.suptitle(model_name)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{model_name}_training_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# 10. TEST EVALUATION

def eval_test(model, model_name):

    model.eval()

    correct = 0
    total_samples = 0
    total_loss = 0

    scores = []
    labels_list = []

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images.float())

            loss = loss_fn(outputs, labels)

            predicted = torch.argmax(outputs, dim=1)

            correct += (
                predicted == labels
            ).sum().item()

            total_samples += labels.size(0)

            total_loss += loss.item() * labels.size(0)

            probabilities = torch.softmax(outputs, dim=1)

            scores.extend(
                probabilities[:, 1].cpu().numpy()
            )

            labels_list.extend(
                labels.cpu().numpy()
            )

    test_accuracy = correct / total_samples
    test_loss = total_loss / total_samples

    print("\nTest results:", model_name)
    print("Number of test samples:", total_samples)
    print("Test accuracy:", test_accuracy)
    print("Test loss:", test_loss)

    return test_accuracy, test_loss, scores, labels_list


# ============================================================
# 11. CALCULATE METRICS AND PLOT RESULTS

def calculate_metrics_and_plots(
    labels,
    scores,
    model_name,
    threshold=0.5
):

    predicted_labels = np.where(
        np.array(scores) >= threshold, 1, 0
    )

    cm = confusion_matrix(
        labels,
        predicted_labels,
        labels=[0, 1]
    )

    accuracy = accuracy_score(labels, predicted_labels)

    precision = precision_score(
        labels, predicted_labels, zero_division=0
    )

    recall = recall_score(
        labels, predicted_labels, zero_division=0
    )

    f1 = f1_score(
        labels, predicted_labels, zero_division=0
    )

    roc_auc_value = roc_auc_score(labels, scores)

    print("\nClassification Report:", model_name)

    print(
        classification_report(
            labels,
            predicted_labels,
            labels=[0, 1],
            target_names=["AD", "MCI"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(cm)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("AUC:", roc_auc_value)

    # Confusion matrix plot

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["AD", "MCI"],
        yticklabels=["AD", "MCI"],
        cbar=False
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plt.savefig(
        PLOT_DIR / f"{model_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # ROC curve

    fpr, tpr, _ = roc_curve(labels, scores)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC curve (AUC = {roc_auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid()

    plt.savefig(
        PLOT_DIR / f"{model_name}_roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(roc_auc_value),
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# 12. RUN ALL THREE MODELS

models_to_train = [
    ("Proposed_Model", CNN_Mode_Proposed),
    ("Testing_Model_2", CNN_Model_Test2),
    ("Testing_Model_3", CNN_Model_Test3)
]

all_results = {}

for model_name, model_class in models_to_train:

    # Create a fresh model
    model = model_class()

    # Train
    trained_model, history = train_model(
        model,
        model_name
    )

    # Plot training curves
    plot_training_curves(
        history,
        model_name
    )

    # Evaluate on test set
    test_acc, test_loss, scores, labels_list = eval_test(
        trained_model,
        model_name
    )

    # Calculate final metrics and plots
    metrics = calculate_metrics_and_plots(
        labels_list,
        scores,
        model_name
    )

    # Store results
    all_results[model_name] = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        **metrics
    }


# ============================================================
# 13. SAVE FINAL RESULTS

results_path = OUTPUT_DIR / "model_comparison.json"

with open(results_path, "w") as f:

    json.dump(
        all_results,
        f,
        indent=4
    )

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

for model_name, results in all_results.items():

    print("\n", model_name)

    for metric, value in results.items():

        print(f"{metric}: {value}")

print("\nAll experiments completed.")
print("Results saved in:", OUTPUT_DIR)
