"""训练总入口：依次训练 3 条基线与本文改进模型，并在测试集评估。"""
import torch

from configs import settings
from configs.settings import CONFIG, CLASS_NAMES
from aloe_classifier.data import build_dataloaders
from aloe_classifier.engine import run_training, collect_predictions
from aloe_classifier.metrics import compute_metrics, plot_confusion_matrix
from aloe_classifier.models import MODEL_REGISTRY


def main():
    train_loader, val_loader, test_loader, folder_classes = \
        build_dataloaders(settings.DATA_DIR, CONFIG["batch_size"])
    print(f"数据目录类别（ImageFolder 排序）：{folder_classes}")
    print(f"运行设备：{settings.DEVICE} | Kaggle 环境：{settings.IS_KAGGLE}")

    summary = {}
    for model_name, builder in MODEL_REGISTRY.items():
        model = builder(settings.NUM_CLASSES)
        weight_path, best_val = run_training(
            model, model_name, train_loader, val_loader
        )

        # 加载最优权重，在测试集评估
        model.load_state_dict(
            torch.load(weight_path, map_location=settings.DEVICE)
        )
        y_true, y_pred = collect_predictions(model, test_loader)
        metric = compute_metrics(y_true, y_pred)
        plot_confusion_matrix(
            y_true, y_pred, CLASS_NAMES,
            f"{settings.FIGURE_DIR}/{model_name}_confusion_matrix.png",
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
