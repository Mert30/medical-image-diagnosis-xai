import os
import torch

# Dataset root: folder that contains train/ val/ test/ (ImageFolder layout).
# Override with:  export DATA_DIR=/path/to/chest_xray
DATA_DIR = os.environ.get("DATA_DIR", "data/chest_xray")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
DROPOUT = 0.4
HIDDEN_UNITS = 256
NUM_CLASSES = 2
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "results/resnet50_pneumonia.pth"
