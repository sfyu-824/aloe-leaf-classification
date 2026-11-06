"""训练与验证引擎：单轮训练、验证、早停与最优权重保存。"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs import settings
from configs.settings import CONFIG
from aloe_classifier.data import cutmix_batch, cutmix_criterion


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best = 0.0
        self.counter = 0

    def step(self, metric):
        if metric > self.best:
            self.best, self.counter = metric, 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def run_training(model, model_name, train_loader, val_loader):
    """完整训练流程，返回最优权重路径与最佳验证精度。"""
    device = settings.DEVICE
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=CONFIG["label_smoothing"]
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    early_stop = EarlyStopping(CONFIG["early_stop_patience"])

    weight_path = f"{settings.WEIGHT_DIR}/{model_name}.pth"
    best_val = 0.0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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

        val_acc = evaluate_accuracy(model, val_loader, device)
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
def evaluate_accuracy(model, loader, device=None):
    device = device or settings.DEVICE
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def collect_predictions(model, loader):
    """收集测试集预测与真实标签，供指标与混淆矩阵使用。"""
    device = settings.DEVICE
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
    return y_true, y_pred
