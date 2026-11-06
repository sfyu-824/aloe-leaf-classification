# 基于改进 MobileNetV3-Small 的 ALV-Leaf 芦荟叶片四分类

针对 ALV-Leaf 数据集，完成芦荟叶片 **病害 / 干枯 / 成熟健康 / 幼年健康** 四分类。
以 ResNet-50、EfficientNet-B0、原生 MobileNetV3-Small 为 3 条基线，
提出在中层逆残差块后嵌入 **SE 通道注意力** 的改进模型，并提供 Flask 演示原型。

## 目录结构

```
aloe-leaf-classification/
├── configs/
│   ├── __init__.py
│   └── settings.py        # 环境自动识别（Kaggle/本地）与路径、超参配置
├── aloe_classifier/
│   ├── __init__.py
│   ├── data.py            # 数据增强、CutMix、DataLoader（7:2:1）
│   ├── models.py          # SE 注意力、3 基线、改进模型及注册表
│   ├── engine.py          # 训练/验证循环、早停、最优权重保存
│   ├── metrics.py         # 准确率、Macro-F1、混淆矩阵
│   └── explain.py         # Grad-CAM 可解释性
├── train.py               # 训练总入口
├── infer.py               # 单图推理
├── web_app.py             # Flask Web 演示原型
├── requirements.txt
└── .gitignore
```

## 本地运行

```bash
pip install -r requirements.txt
# 数据集放入 ./dataset，每个类别一个子文件夹
python train.py
python infer.py --image ./test.jpg
python web_app.py
```

## Kaggle 运行

```bash
# 1. 克隆仓库
!git clone https://github.com/<用户名>/aloe-leaf-classification.git
%cd aloe-leaf-classification

# 2. 安装依赖
!pip install -r requirements.txt

# 3. 挂载数据集（右侧 Data 添加 ALV-Leaf）
!ln -s /kaggle/input/alvleaf ./dataset

# 4. 启动训练
!python train.py
```

`configs/settings.py` 通过检测 `KAGGLE_KERNEL_RUN_TYPE` 与 `/kaggle` 目录自动切换
数据路径（`/kaggle/input/...`）与可写输出路径（`/kaggle/working/...`）。

## 实验设置

| 类型 | 模型 |
| --- | --- |
| 基线 1 | ResNet-50 |
| 基线 2 | EfficientNet-B0 |
| 基线 3 | 原生 MobileNetV3-Small |
| 本文改进 | MobileNetV3-Small + SE 通道注意力 |

CutMix 在 `settings.CONFIG["use_cutmix"]` 中开关，作为对照实验。
