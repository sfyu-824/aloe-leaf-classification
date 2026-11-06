"""评价指标：准确率、Macro-F1、混淆矩阵及可视化。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred):
    """返回整体准确率、Macro-F1 及各类精确率/召回率/F1。"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_precision": precision,
        "macro_recall": recall,
    }


def per_class_report(y_true, y_pred, class_names):
    """返回各类别精确率/召回率/F1 字典，便于论文表格呈现。"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))),
        zero_division=0,
    )
    return {
        name: {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
        }
        for i, name in enumerate(class_names)
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制并保存混淆矩阵热力图。"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
