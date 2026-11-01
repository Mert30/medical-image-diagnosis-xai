"""Explainability: gradient saliency (as in the notebook) and Grad-CAM.

Usage:
    python -m src.explain        # first test image, saliency + Grad-CAM
"""
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from . import config as C
from .data import get_loaders
from .model import build_model


def to_rgb(tensor):
    img = tensor.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
    return (img - img.min()) / (img.max() - img.min())


def overlay(rgb_img, mask, alpha=0.3):
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(mask, (C.IMG_SIZE, C.IMG_SIZE))), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(alpha * heat + (1 - alpha) * rgb_img, 0, 1)


def gradient_saliency(model, input_tensor):
    """Vanilla gradient saliency: |d score / d input|, max over channels."""
    x = input_tensor.clone().requires_grad_(True)
    out = model(x)
    pred = out.argmax(1).item()
    model.zero_grad()
    out[0, pred].backward()
    sal = x.grad.abs().max(dim=1)[0].squeeze().cpu().numpy()
    sal = (sal - sal.min()) / (sal.max() + 1e-7)
    return sal, pred


def grad_cam(model, input_tensor, target_class):
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    return cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0]


def main(index=0, out_path="results/explanations.png"):
    *_, test_loader = get_loaders()
    model = build_model()
    model.load_state_dict(torch.load(C.CHECKPOINT_PATH, map_location=C.DEVICE))
    model.eval()
    for p in model.parameters():  # Grad-CAM needs gradients through the backbone
        p.requires_grad = True

    images, labels = next(iter(test_loader))
    x = images[index:index + 1].to(C.DEVICE)
    true_label = labels[index].item()

    sal, pred = gradient_saliency(model, x)
    cam = grad_cam(model, x, pred)
    rgb = to_rgb(x)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    ax[0].imshow(rgb)
    ax[0].set_title(f"True: {C.CLASS_NAMES[true_label]} | Pred: {C.CLASS_NAMES[pred]}")
    ax[1].imshow(overlay(rgb, sal))
    ax[1].set_title("Gradient saliency")
    ax[2].imshow(overlay(rgb, cam, alpha=0.4))
    ax[2].set_title("Grad-CAM (layer4)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
