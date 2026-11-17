"""数据处理：按 7:2:1 分层划分、基础增强、CutMix 与类别不均衡采样。"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import (
    IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, CONFIG, CLASS_DIRS,
    DATASET_SUBDIR,
)

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
    """扫描 data_dir 下各类别子文件夹，返回 {类别名: [图片路径, ...]}。

    类别文件夹默认位于 DATA_DIR/DATASET_SUBDIR 内（Mendeley 解压后为
    final/ 与 final_Augmented/）；DATASET_SUBDIR 为空时直接扫描 DATA_DIR。
    """
    data_root = os.path.join(data_dir, DATASET_SUBDIR) if DATASET_SUBDIR else data_dir
    grouped = {c: [] for c in CLASS_DIRS}
    for c in CLASS_DIRS:
        folder = os.path.join(data_root, c)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith(IMAGE_EXTS):
                grouped[c].append(os.path.join(folder, fn))
    return grouped


def build_dataloaders(data_dir, batch_size=None):
    """汇总全部图片，在每个类别内按 7:2:1 分层随机划分（固定种子）。

    返回 (train_loader, val_loader, test_loader, class_names, class_counts)。
    class_counts 为各类别训练样本数，供加权损失使用。
    """
    batch_size = batch_size or CONFIG["batch_size"]
    r_train, r_val, _ = CONFIG["split_ratios"]
    rng = random.Random(SPLIT_SEED)

    train_samples, val_samples, test_samples = [], [], []
    grouped = collect_by_class(data_dir)
    class_counts = []
    for label, c in enumerate(CLASS_DIRS):
        paths = grouped[c]
        if not paths:
            print(f"[警告] 类别文件夹为空或未找到：{c}，请检查 DATA_DIR、DATASET_SUBDIR 与 CLASS_DIRS")
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * r_train)
        n_val = int(n * r_val)

        train_samples += [(p, label) for p in paths[:n_train]]
        val_samples += [(p, label) for p in paths[n_train:n_train + n_val]]
        test_samples += [(p, label) for p in paths[n_train + n_val:]]
        class_counts.append(n_train)

    rng.shuffle(train_samples)
    train_set = PathDataset(train_samples, train_transform)
    val_set = PathDataset(val_samples, eval_transform)
    test_set = PathDataset(test_samples, eval_transform)

    train_loader = _build_train_loader(train_set, train_samples, batch_size)

    return (
        train_loader,
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2),
        CLASS_DIRS,
        class_counts,
    )


def _build_train_loader(train_set, train_samples, batch_size):
    """训练集加载器：开启类别过采样时使用 WeightedRandomSampler。"""
    if not CONFIG["use_oversampling"]:
        return DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )

    labels = [label for _, label in train_samples]
    class_counts = [labels.count(i) for i in range(len(CLASS_DIRS))]
    # 每样本权重 = 1 / 所属类别样本数，使小类被更高频采样
    weights = [1.0 / max(class_counts[label], 1) for label in labels]
    sampler = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )
    return DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=2
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
