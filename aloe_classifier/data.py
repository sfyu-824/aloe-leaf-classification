"""数据加载、基础数据增强与 CutMix。"""
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from configs.settings import (
    IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, CONFIG,
)

# 训练集：基础增强（翻转/旋转/裁剪/颜色抖动）
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
    """按 7:2:1 划分并返回训练/验证/测试 DataLoader 及类别名。"""
    batch_size = batch_size or CONFIG["batch_size"]
    full_set = datasets.ImageFolder(data_dir, transform=train_transform)
    total = len(full_set)
    r_train, r_val, _ = CONFIG["split_ratios"]
    n_train = int(total * r_train)
    n_val = int(total * r_val)
    n_test = total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_set, [n_train, n_val, n_test]
    )
    val_set.dataset.transform = eval_transform
    test_set.dataset.transform = eval_transform

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2),
        full_set.classes,
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
