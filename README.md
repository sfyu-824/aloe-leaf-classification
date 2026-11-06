# 基于改进 MobileNetV3-Small 的 ALV-Leaf 芦荟叶片四分类

针对 ALV-Leaf 数据集，完成芦荟叶片 **病害 / 干枯 / 成熟健康 / 幼年健康** 四分类。
以 ResNet-50、EfficientNet-B0、原生 MobileNetV3-Small 为 3 条基线，
提出在中层逆残差块后嵌入 **SE 通道注意力** 的改进模型，并提供 Flask 演示原型。

## 目录结构

```
aloe-leaf-classification/
├── models/
│   ├── __init__.py
│   ├── resnet50.py          # 基线 1
│   ├── efficientnet_b0.py   # 基线 2
│   ├── mobilenetv3_base.py  # 基线 3：原生 MobileNetV3-Small
│   └── mobilenetv3_se.py    # 本文改进：MobileNetV3-Small + SE
├── utils/
│   ├── __init__.py
│   ├── helper.py            # 早停
│   ├── preprocess.py        # 数据增强、CutMix、DataLoader（7:2:1）
│   └── visualize.py         # 指标、混淆矩阵、Grad-CAM
├── config.py                # 环境自动识别与路径、超参配置
├── main.py                  # 训练主入口
├── inference.py             # 单张图片推理
├── app.py                   # Flask Web 演示原型
├── requirements.txt
└── .gitignore
```

## 本地运行

```bash
pip install -r requirements.txt
# 数据集放入 ./dataset，每个类别一个子文件夹
python main.py
python inference.py --image ./test.jpg
python app.py
```

## Kaggle 运行

```bash
!git clone https://github.com/<用户名>/aloe-leaf-classification.git
%cd aloe-leaf-classification
!pip install -r requirements.txt
!ln -s /kaggle/input/alvleaf ./dataset
!python main.py
```

`config.py` 通过检测 `KAGGLE_KERNEL_RUN_TYPE` 与 `/kaggle` 目录自动切换
数据路径（`/kaggle/input/...`）与可写输出路径（`/kaggle/working/...`）。

## 实验设置

| 类型 | 模型 |
| --- | --- |
| 基线 1 | ResNet-50 |
| 基线 2 | EfficientNet-B0 |
| 基线 3 | 原生 MobileNetV3-Small |
| 本文改进 | MobileNetV3-Small + SE 通道注意力 |

CutMix 在 `config.CONFIG["use_cutmix"]` 中开关，作为对照实验。
