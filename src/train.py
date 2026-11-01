import os
import torch
import torch.nn as nn
import torch.optim as optim

from . import config as C
from .data import get_loaders, compute_class_weights
from .model import build_model


def main():
    train_ds, _, _, train_loader, _, _ = get_loaders()
    print(f"Train images: {len(train_ds)} | classes: {train_ds.class_to_idx}")

    class_weights = compute_class_weights(train_ds.targets).to(C.DEVICE)
    print(f"Class weights -> NORMAL: {class_weights[0]:.2f}, PNEUMONIA: {class_weights[1]:.2f}")

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.fc.parameters(), lr=C.LR)  # only the new head is trained

    for epoch in range(C.EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(C.DEVICE), labels.to(C.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        print(f"Epoch [{epoch + 1}/{C.EPOCHS}] - "
              f"Loss: {running_loss / len(train_loader.dataset):.4f} - "
              f"Accuracy: {100.0 * correct / total:.2f}%")

    os.makedirs(os.path.dirname(C.CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), C.CHECKPOINT_PATH)
    print(f"Saved checkpoint -> {C.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
