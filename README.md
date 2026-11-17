# 基于改进 MobileNetV3-Small 的芦荟叶片病害五分类

针对芦荟叶病检测数据集（Aloe Vera Leaf Disease Detection Dataset，Mendeley Data，
DOI:10.17632/7w6t4zx33n.1），完成芦荟叶片 **锈病 / 炭疽病 / 叶斑病 / 晒伤 / 健康** 五分类。
以 ResNet-50、EfficientNet-B0、原生 MobileNetV3-Small 为 3 条基线，
提出在中层逆残差块后嵌入 **CBAM 混合注意力（通道 + 空间）** 的改进模型，
并针对类别不均衡问题引入 **加权交叉熵 + 小类过采样**，同时保留 CutMix 增强，
最终提供 Flask 演示原型。

## 数据集说明

数据集为单个 `.rar` 压缩包（约 1.61 GB），解压后应包含 5 个类别子文件夹：

| 类别 | 文件夹名 | 样本数 |
| --- | --- | --- |
| 锈病 | `Aloe Rust` | 210 |
| 炭疽病 | `Anthracnose` | 500 |
| 叶斑病 | `Leaf Spot` | 500 |
| 晒伤 | `SunBurn` | 510 |
| 健康 | `Healthy` | 780 |

- 请把解压后的 **5 个类别文件夹所在的目录** 作为数据根目录。
- 若实际解压出的文件夹名与上表不一致，只需修改 `config.py` 中
  `CLASS_DIRS` 与 `CLASS_NAMES`（两者顺序一一对应）即可。
- 数据集没有现成的训练/验证/测试划分，代码会按 **7:2:1** 在每个类别内
  分层随机划分（固定随机种子，可复现）。

## 目录结构

```
aloe-leaf-classification/
├── models/
│   ├── __init__.py
│   ├── resnet50.py            # 基线 1
│   ├── efficientnet_b0.py     # 基线 2
│   ├── mobilenetv3_base.py    # 基线 3：原生 MobileNetV3-Small
│   ├── attention.py           # SE / SAM / CBAM 注意力模块
│   └── mobilenetv3_cbam.py    # 本文改进：MobileNetV3-Small + CBAM（含消融变体）
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
# 数据集放入 ./dataset，其下为 5 个类别子文件夹
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

**消融实验**（`config.CONFIG["run_ablation"]` 控制，默认开启）：
- `mobilenetv3_se`：仅 SE 通道注意力
- `mobilenetv3_sam`：仅 SAM 空间注意力
- `mobilenetv3_cbam`：完整 CBAM（通道 + 空间）

**类别不均衡处理**（`config.CONFIG["use_weighted_loss"]` /
`use_oversampling` 控制，默认开启）：加权交叉熵 + WeightedRandomSampler 过采样。

CutMix 通过 `config.CONFIG["use_cutmix"]` 开关，作为对照实验。
