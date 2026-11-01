"""Evaluate the trained model on the held-out test set.

Prints accuracy, per-class precision/recall/F1 and the confusion matrix,
and saves the confusion-matrix figure to results/.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from . import config as C
from .data import get_loaders
from .model import build_model


@torch.no_grad()
def main():
    *_, test_loader = get_loaders()
    model = build_model()
    model.load_state_dict(torch.load(C.CHECKPOINT_PATH, map_location=C.DEVICE))
    model.eval()

    y_true, y_pred = [], []
    for images, labels in test_loader:
        preds = model(images.to(C.DEVICE)).argmax(1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

    names = [C.CLASS_NAMES[i] for i in range(C.NUM_CLASSES)]
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Test accuracy: {100 * acc:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("Saved results/confusion_matrix.png")


if __name__ == "__main__":
    main()
