"""数据处理：汇总官方三个集合后按 7:2:1 分层划分、基础增强、CutMix。"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, CONFIG,
)

# 数据集官方目录结构（集合在上、类别在下）
SET_DIRS = ["Training Set", "Validation Set", "Test Set"]
# 类别按字母序，与 config.CLASS_NAMES 中文顺序一一对应
CLASS_DIRS = ["Disease", "Dried", "Mature Healthy", "Young Healthy"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_SEED = 42  # 固定随机种子，保证每次划分结果一致、可复现

# 训练集基础增强
# 说明：RandomResizedCrop 同时完成随机缩放与随机位置裁剪，
# 直接在 224 尺寸上再用同尺寸 RandomCrop 不会产生位置变化（等于空操作），
# 故此处改用 RandomResizedCrop，使“随机裁剪”增强真正生效。
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        (IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.0), ratio=(0.75, 1.333)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
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


class PathDataset(Dataset):
    """按图片路径与标签加载的数据集。"""

    def __init__(self, samples, transform):
        self.samples = samples          # [(path, label), ...]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def collect_by_class(data_dir):
    """汇总三个 Set 目录下的全部图片，返回 {类别名: [图片路径, ...]}。"""
    grouped = {c: [] for c in CLASS_DIRS}
    for set_name in SET_DIRS:
        for c in CLASS_DIRS:
            folder = os.path.join(data_dir, set_name, c)
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                if fn.lower().endswith(IMAGE_EXTS):
                    grouped[c].append(os.path.join(folder, fn))
    return grouped


def build_dataloaders(data_dir, batch_size=None):
    """汇总全部图片，在每个类别内按 7:2:1 分层随机划分（固定种子）。

    返回训练/验证/测试 DataLoader 及类别名。函数签名与 main.py 保持兼容。
    """
    batch_size = batch_size or CONFIG["batch_size"]
    r_train, r_val, _ = CONFIG["split_ratios"]
    rng = random.Random(SPLIT_SEED)

    train_samples, val_samples, test_samples = [], [], []
    grouped = collect_by_class(data_dir)
    for label, c in enumerate(CLASS_DIRS):
        paths = grouped[c]
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * r_train)
        n_val = int(n * r_val)

        train_samples += [(p, label) for p in paths[:n_train]]
        val_samples += [(p, label) for p in paths[n_train:n_train + n_val]]
        test_samples += [(p, label) for p in paths[n_train + n_val:]]

    rng.shuffle(train_samples)
    train_set = PathDataset(train_samples, train_transform)
    val_set = PathDataset(val_samples, eval_transform)
    test_set = PathDataset(test_samples, eval_transform)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2),
        CLASS_DIRS,
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
