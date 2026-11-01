import torch.nn as nn
from torchvision import models

from . import config as C


def build_model():
    """ResNet50 (ImageNet weights), frozen backbone, new 2-class head."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, C.HIDDEN_UNITS),
        nn.ReLU(),
        nn.Dropout(C.DROPOUT),
        nn.Linear(C.HIDDEN_UNITS, C.NUM_CLASSES),
    )
    return model.to(C.DEVICE)
