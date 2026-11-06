"""数据处理：官方预划分数据集加载、基础增强、CutMix。"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, CONFIG,
)

# 数据集官方子目录名（集合在上、类别在下）
TRAIN_DIR = "Training Set"
VAL_DIR = "Validation Set"
TEST_DIR = "Test Set"

# 训练集基础增强
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE), pad_if_needed=True),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
])

# 验证/测试：仅尺寸调整与归一化
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
])


def build_dataloaders(data_dir, batch_size=None):
    """读取官方预划分的 Training/Validation/Test Set，返回三个 DataLoader 及类别名。

    三个集合内部均为 4 个类别文件夹，ImageFolder 按字母序得到类别：
    ['Disease', 'Dried', 'Mature Healthy', 'Young Healthy']，
    与 config.CLASS_NAMES 的中文类别顺序一一对应。
    """
    batch_size = batch_size or CONFIG["batch_size"]

    train_set = datasets.ImageFolder(
        os.path.join(data_dir, TRAIN_DIR), transform=train_transform
    )
    val_set = datasets.ImageFolder(
        os.path.join(data_dir, VAL_DIR), transform=eval_transform
    )
    test_set = datasets.ImageFolder(
        os.path.join(data_dir, TEST_DIR), transform=eval_transform
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2),
        train_set.classes,
    )


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_batch(images, labels, alpha=None):
    """返回混合图像、两组标签与混合比例 lambda。"""
    alpha = alpha or CONFIG["cutmix_alpha"]
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0), device=images.device)
    x1, y1, x2, y2 = rand_bbox(images.size(), lam)
    images[:, :, x1:x2, y1:y2] = images[perm, :, x1:x2, y1:y2]
    return images, labels, labels[perm], lam


def cutmix_criterion(base_criterion, outputs, y_a, y_b, lam):
    return lam * base_criterion(outputs, y_a) \
        + (1 - lam) * base_criterion(outputs, y_b)
