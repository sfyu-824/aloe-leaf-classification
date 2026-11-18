"""全局配置：自动识别 Kaggle / 本地环境并切换数据与输出路径。"""
import os

import torch


def is_kaggle() -> bool:
    """Kaggle 运行时注入 KAGGLE_KERNEL_RUN_TYPE，且存在 /kaggle 目录。"""
    return (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or os.path.isdir("/kaggle")
    )


IS_KAGGLE = is_kaggle()

# ---------- 路径配置 ----------
if IS_KAGGLE:
    OUTPUT_DIR = "/kaggle/working/output"
    if os.path.isdir("./dataset"):
        DATA_DIR = "./dataset"
    else:
        DATA_DIR = "/kaggle/input/aloe-vera-leaf-disease-detection-dataset"
else:
    DATA_DIR = "./dataset"
    OUTPUT_DIR = "./output"

WEIGHT_DIR = os.path.join(OUTPUT_DIR, "weights")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
for _d in (WEIGHT_DIR, FIGURE_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------- 数据与类别 ----------
IMAGE_SIZE = 224

CLASS_DIRS = ["Aloe Rust", "Anthracnose", "Leaf Spot", "Sunburn", "Healthy"]
CLASS_NAMES = ["锈病", "炭疽病", "叶斑病", "晒伤", "健康叶片"]
NUM_CLASSES = len(CLASS_NAMES)

DATASET_SUBDIR = "final"

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ---------- 训练超参 ----------
CONFIG = {
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 5e-5,
    "weight_decay": 3e-4,
    "label_smoothing": 0.15,
    "early_stop_patience": 8,
    "warmup_epochs": 5,
    "use_cutmix": True,
    "cutmix_alpha": 0.7,
    "use_weighted_loss": True,
    "use_oversampling": True,
    "split_ratios": (0.7, 0.2, 0.1),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
