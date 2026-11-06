"""基线 3：原生 MobileNetV3-Small。"""
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def get_mobilenetv3_base(num_classes=4):
    model = mobilenet_v3_small(weights="IMAGENET1K_V1")
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
