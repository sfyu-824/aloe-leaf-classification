"""训练主入口：3 条基线 + 1 条改进模型，并在测试集评估。"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    IS_KAGGLE, DEVICE, NUM_CLASSES, CLASS_NAMES, CONFIG,
    DATA_DIR, WEIGHT_DIR, FIGURE_DIR,
)
from utils.preprocess import build_dataloaders, cutmix_batch, cutmix_criterion
from utils.helper import EarlyStopping
from utils.visualize import compute_metrics, plot_confusion_matrix
from models.resnet50 import get_resnet50
from models.efficientnet_b0 import get_efficientnet_b0
from models.mobilenetv3_base import get_mobilenetv3_base
from models.mobilenetv3_se import get_mobilenetv3_se

# 模型清单：名称 -> 构造函数
MODEL_LIST = [
    ("resnet50", get_resnet50),
    ("efficientnet_b0", get_efficientnet_b0),
    ("mobilenetv3_base", get_mobilenetv3_base),
    ("mobilenetv3_se", get_mobilenetv3_se),
]


def run_training(model, model_name, train_loader, val_loader):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=CONFIG["label_smoothing"]
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    early_stop = EarlyStopping(CONFIG["early_stop_patience"])

    weight_path = f"{WEIGHT_DIR}/{model_name}.pth"
    best_val = 0.0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if CONFIG["use_cutmix"]:
                images, ya, yb, lam = cutmix_batch(images, labels)
                outputs = model(images)
                loss = cutmix_criterion(criterion, outputs, ya, yb, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        val_acc = evaluate_accuracy(model, val_loader)
        scheduler.step()
        print(f"[{model_name}] Epoch {epoch + 1:02d} "
              f"Loss {loss_sum / len(train_loader):.4f} ValAcc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), weight_path)

        if early_stop.step(val_acc):
            print(f"[{model_name}] 触发早停")
            break

    return weight_path, best_val


@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        preds = model(images).argmax(dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
    return y_true, y_pred


def main():
    train_loader, val_loader, test_loader, folder_classes = \
        build_dataloaders(DATA_DIR, CONFIG["batch_size"])
    print(f"数据目录类别：{folder_classes}")
    print(f"设备：{DEVICE} | Kaggle：{IS_KAGGLE}")

    summary = {}
    for model_name, builder in MODEL_LIST:
        model = builder(NUM_CLASSES)
        weight_path, best_val = run_training(
            model, model_name, train_loader, val_loader
        )
        model.load_state_dict(
            torch.load(weight_path, map_location=DEVICE)
        )
        y_true, y_pred = collect_predictions(model, test_loader)
        metric = compute_metrics(y_true, y_pred)
        plot_confusion_matrix(
            y_true, y_pred, CLASS_NAMES,
            f"{FIGURE_DIR}/{model_name}_confusion_matrix.png",
        )
        summary[model_name] = (best_val, metric)
        print(f"{model_name} 测试集指标：{metric}\n")

    print("================ 结果汇总 ================")
    for name, (best_val, metric) in summary.items():
        print(f"{name:18s} 验证Acc {best_val:.4f} | "
              f"测试Acc {metric['accuracy']:.4f} | "
              f"Macro-F1 {metric['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
