
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    roc_auc_score,
    auc,
    classification_report
)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss function
loss_fn = nn.CrossEntropyLoss()


# 1. Evaluate model on test dataset
def eval_test(test_loader, model):

    model.to(device)
    model.eval()

    correct = 0
    total_samples = 0
    total_loss = 0

    scores = []
    labels_list = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images.float())

            loss = loss_fn(outputs, labels)

            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            total_loss += loss.item() * labels.size(0)

            probabilities = torch.softmax(outputs, dim=1)

            scores.extend(probabilities[:, 1].cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = correct / total_samples
    average_loss = total_loss / total_samples

    return accuracy, average_loss, scores, labels_list


# 2. Plot ROC curve
def plot_roc_curve(labels, scores):

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label='ROC curve (area = %0.2f)' % roc_auc
    )
    plt.plot(
        [0, 1], [0, 1],
        color='navy',
        lw=2,
        linestyle='--'
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# 3. Calculate metrics and plot confusion matrix
def calculate_metrics_and_confusion_matrix(labels, scores, threshold=0.5):

    scores = np.array(scores)

    # Convert scores to predicted labels based on threshold
    predicted_labels = np.where(scores >= threshold, 1, 0)

    # Calculate evaluation metrics
    tn, fp, fn, tp = confusion_matrix(
        labels, predicted_labels
    ).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    dcs = 2 * (precision * recall) / (precision + recall)

    roc_auc_value = roc_auc_score(labels, scores)

    # Generate confusion matrix
    cm = confusion_matrix(labels, predicted_labels)

    # Plot confusion matrix
    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(labels, predicted_labels))

    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("DCS:", dcs)
    print("AUC:", roc_auc_value)


# =====================================================
# Test Proposed Model

test_acc, test_loss, scores, labels_list = eval_test(
    test_loader, model
)

print(f"Test accuracy for proposed model: {test_acc * 100}%")
print(f"Test loss for proposed model: {test_loss}")
print("Number of test samples:", len(labels_list))

plot_roc_curve(labels_list, scores)

calculate_metrics_and_confusion_matrix(
    labels_list, scores, threshold=0.5
)


# =====================================================
# Test Model 2

test_acc_model_test2, test_loss_model_test2, scores_model_test2, labels_list_model_test2 = eval_test(
    test_loader, model_test2
)

print(f"Test accuracy for model 2: {test_acc_model_test2 * 100}%")
print(f"Test loss for model 2: {test_loss_model_test2}")
print("Number of test samples:", len(labels_list_model_test2))

plot_roc_curve(labels_list_model_test2, scores_model_test2)

calculate_metrics_and_confusion_matrix(
    labels_list_model_test2, scores_model_test2, threshold=0.5
)


# =====================================================
# Test Model 3

test_acc_model_test3, test_loss_model_test3, scores_model_test3, labels_list_model_test3 = eval_test(
    test_loader, model_test3
)

print(f"Test accuracy for model 3: {test_acc_model_test3 * 100}%")
print(f"Test loss for model 3: {test_loss_model_test3}")
print("Number of test samples:", len(labels_list_model_test3))

plot_roc_curve(labels_list_model_test3, scores_model_test3)

calculate_metrics_and_confusion_matrix(
    labels_list_model_test3, scores_model_test3, threshold=0.5
)
