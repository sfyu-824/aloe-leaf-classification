"""基线 1：ResNet-50。"""
import torch.nn as nn
from torchvision.models import resnet50


def get_resnet50(num_classes=4):
    model = resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
