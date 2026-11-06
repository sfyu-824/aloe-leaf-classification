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
    # 优先使用项目内软链接 ./dataset（可指向任意实际数据集路径，
    # 规避数据集目录名带空格或层级较深的问题）；不存在时回退默认挂载路径
    if os.path.isdir("./dataset"):
        DATA_DIR = "./dataset"
    else:
        DATA_DIR = "/kaggle/input/alvleaf"
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
NUM_CLASSES = 4
CLASS_NAMES = ["病害叶片", "干枯叶片", "成熟健康叶片", "幼年健康叶片"]

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ---------- 训练超参 ----------
CONFIG = {
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "early_stop_patience": 8,
    "use_cutmix": True,              # CutMix 作为常规增强启用，与基础增强叠加
    "cutmix_alpha": 1.0,
    "split_ratios": (0.7, 0.2, 0.1),  # 训练/验证/测试
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
