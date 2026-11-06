"""基线 2：EfficientNet-B0。"""
import torch.nn as nn
from torchvision.models import efficientnet_b0


def get_efficientnet_b0(num_classes=4):
    model = efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model
