import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from . import config as C


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])
    # No augmentation for validation / test
    eval_tf = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_loaders(data_dir=C.DATA_DIR):
    train_tf, eval_tf = get_transforms()
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=eval_tf)
    test_ds = datasets.ImageFolder(f"{data_dir}/test", transform=eval_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def compute_class_weights(targets):
    """weight_c = N / (num_classes * N_c) -- counters the NORMAL/PNEUMONIA imbalance."""
    counts = np.bincount(targets)
    weights = len(targets) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)
