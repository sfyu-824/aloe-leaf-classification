# 基于改进 MobileNetV3-Small 的芦荟叶片病害五分类

针对芦荟叶病检测数据集（Aloe Vera Leaf Disease Detection Dataset，Mendeley Data，
DOI:10.17632/7w6t4zx33n.1），完成芦荟叶片 **锈病 / 炭疽病 / 叶斑病 / 晒伤 / 健康** 五分类。
以 ResNet-50、EfficientNet-B0、原生 MobileNetV3-Small 为 3 条基线，
提出在中层逆残差块后嵌入 **CBAM 混合注意力（通道 + 空间）** 的改进模型，
并针对类别不均衡问题引入 **加权交叉熵 + 小类过采样**，同时保留 CutMix 增强，
最终提供 Flask 演示原型。

## 数据集说明

数据集为单个 `.rar` 压缩包（约 1.61 GB），解压后目录结构如下：

```
Aloe vera leaf disease detection dataset/
├── final/                 # 原始 2500 张（论文实验基准，推荐）
│   ├── Aloe Rust/         # 锈病 210 张
│   ├── Anthracnose/       # 炭疽病 500 张
│   ├── Leaf Spot/         # 叶斑病 500 张
│   ├── Sunburn/           # 晒伤 510 张
│   └── Healthy/           # 健康 780 张
└── final_Augmented/       # 已做数据增强的版本
    └── ...（同上 5 个类别文件夹）
```

- 把解压后的根目录（`Aloe vera leaf disease detection dataset`）作为数据目录，
  代码通过 `config.py` 的 `DATASET_SUBDIR`（默认 `"final"`）定位到原始 2500 张。
- 想用增强版时，把 `DATASET_SUBDIR` 改为 `"final_Augmented"`。
- 若类别文件夹直接放在数据目录下，把 `DATASET_SUBDIR` 设为 `""`。
- 若实际文件夹名不一致，只需改 `config.py` 的 `CLASS_DIRS` 与 `CLASS_NAMES`（顺序一一对应）。
- 数据集没有现成的训练/验证/测试划分，代码会按 **7:2:1** 在每个类别内分层随机划分（固定随机种子，可复现）。

## 目录结构

```
aloe-leaf-classification/
├── models/
│   ├── __init__.py
│   ├── resnet50.py            # 基线 1
│   ├── efficientnet_b0.py     # 基线 2
│   ├── mobilenetv3_base.py    # 基线 3：原生 MobileNetV3-Small
│   ├── attention.py           # CBAM 注意力模块（含通道/空间两个分支）
│   └── mobilenetv3_cbam.py    # 本文改进：MobileNetV3-Small + CBAM
├── utils/
│   ├── __init__.py
│   ├── helper.py              # 早停
│   ├── preprocess.py          # 划分、增强、CutMix、类别不均衡采样
│   └── visualize.py           # 指标、混淆矩阵、Grad-CAM
├── config.py                  # 环境自动识别、路径与超参配置
├── main.py                    # 训练主入口
├── inference.py               # 单张图片推理
├── app.py                     # Flask Web 演示原型
├── requirements.txt
└── .gitignore
```

## 本地运行

```bash
pip install -r requirements.txt
# 把解压后的数据集根目录放入 ./dataset（即 ./dataset/final/...）
python main.py
python inference.py --image ./test.jpg
python app.py
```

## Kaggle 运行

```bash
!git clone https://github.com/sfyu-824/aloe-leaf-classification.git
%cd aloe-leaf-classification
!pip install -r requirements.txt
!ln -s /kaggle/input/<你的数据集路径> ./dataset
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
| 本文改进 | MobileNetV3-Small + CBAM 混合注意力 |

**类别不均衡处理**（`config.CONFIG["use_weighted_loss"]` /
`use_oversampling` 控制，默认开启）：加权交叉熵 + WeightedRandomSampler 过采样。

CutMix 通过 `config.CONFIG["use_cutmix"]` 开关，作为对照实验。
