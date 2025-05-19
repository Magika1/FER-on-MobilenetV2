# 面部情绪识别 (FER) 模块文档

## 概述

`fer` 目录包含了用于面部情绪识别（Facial Emotion Recognition, FER）的核心代码。这包括了深度学习模型的定义、数据集的处理、模型的训练策略以及相关的辅助工具。

## 1. 模型与分类器设计 (`fer/model.py`)

面部情绪识别模型基于轻量级的 MobileNetV2 架构，旨在实现高效的实时情绪分类。

- **基础模型**: 使用预训练的 `MobileNetV2` 模型作为特征提取器。预训练权重来自于 `torchvision.models.MobileNet_V2_Weights.DEFAULT`。
- **分类器**:
  - 原始 `MobileNetV2` 的分类头被替换为一个新的序列模块。
  - 新的分类器结构通常包含：
    - 一个 Dropout 层，以减少过拟合 (例如 `p=0.2`)。
    - 一个全连接层，将 `MobileNetV2` 输出的特征映射到中间维度 (例如 256)。
    - 一个 ReLU 激活函数。
    - 另一个 Dropout 层。
    - 最后一个全连接层，将中间特征映射到最终的情绪类别数量 (默认为 7 类)。
- **输出类别**: 模型设计用于识别 7 种基本情绪：生气 (angry), 厌恶 (disgust), 恐惧 (fear), 开心 (happy), 悲伤 (sad), 惊讶 (surprise), 和平静 (neutral)。
- **模型创建**: `create_model(device)` 函数负责实例化 `EmotionMobileNetV2` 模型并将其移动到指定的计算设备（CPU 或 CUDA）。

## 2. 数据集处理 (`fer/dataset.py`)

数据集处理模块负责加载面部情绪图像数据，并进行必要的预处理和数据增强，为模型训练提供合适的输入格式。

- **数据来源**: 通常使用 FER2013 数据集，该数据集以 CSV 文件格式提供，其中包含像素值、情绪标签和用途（训练、验证、测试）。
- **数据加载**:
  - `FERDataset` 类继承自 `torch.utils.data.Dataset`，用于从 CSV 文件中读取图像数据和标签。
  - 图像数据（像素字符串）被转换回图像格式（例如，48x48 灰度图或 RGB 图像）。
- **数据预处理与增强**:
  - **转换 (Transforms)**: 应用一系列图像变换操作，通常包括：
    - `Resize`: 将图像调整到特定尺寸 (例如 256x256)。
    - `CenterCrop` 或 `RandomResizedCrop`: 中心裁剪或随机裁剪到模型输入尺寸 (例如 224x224)。
    - `RandomHorizontalFlip`: 随机水平翻转，增加数据多样性。
    - `RandomRotation`: 随机旋转。
    - `ColorJitter`: 随机改变亮度、对比度、饱和度。
    - `ToTensor`: 将 PIL 图像或 NumPy 数组转换为 PyTorch 张量，并将像素值缩放到 [0, 1] 区间。
    - `Normalize`: 使用预定义的均值和标准差对图像张量进行归一化 (通常是 ImageNet 的均值和标准差)。
  - 训练集通常应用更复杂的数据增强，而验证集和测试集则应用较少的或仅有必要的尺寸调整和归一化。
- **数据加载器 (`DataLoaders`)**:
  - `create_datasets` 函数用于创建训练、验证和测试数据集实例。
  - `create_dataloaders` 函数使用 `torch.utils.data.DataLoader` 为这些数据集创建数据加载器，以便进行批量加载、打乱数据（训练集）和并行加载。

## 3. 模型训练策略 (`fer/train.py` & `fer/config.py`)

模型训练采用分阶段的微调策略，并结合了多种训练技巧以提升模型性能和稳定性。

- **训练配置文件 (`fer/config.py`)**:

  - `TrainingConfig` 类集中管理所有训练相关的超参数和配置。
  - **数据集配置**: CSV 路径, `batch_size`, `num_workers`。
  - **模型配置**: `num_classes`。
  - **两阶段训练配置**:
    - **Phase 1**: `phase1_epochs`, `phase1_lr` (学习率)。此阶段通常只训练新添加的分类器层，冻结 MobileNetV2 的特征提取层。
    - **Phase 2**: `phase2_epochs`, `phase2_feature_lr` (特征提取器学习率), `phase2_classifier_lr` (分类器学习率), `phase2_layers_unfreeze` (从 MobileNetV2 末尾解冻的层数)。此阶段对部分特征提取层和分类器进行联合微调。
  - **检查点与日志**: `checkpoint_dir`, `log_dir`, `save_freq` (保存检查点的频率)。
  - **早停参数**: `patience`, `min_delta`，用于在验证集性能不再提升时提前停止训练。
  - **实验配置**: 允许定义和选择不同的实验设置。

- **训练流程 (`fer/train.py`)**:
  - **设备选择**: 自动选择 CUDA (如果可用) 或 CPU。
  - **日志设置**: 使用 `fer/utils/logger.py` 设置日志记录。
  - **数据加载**: 调用 `create_datasets` 和 `create_dataloaders`。
  - **模型创建**: 调用 `create_model`。
  - **两阶段训练 (`train_phase1`, `train_phase2`)**:
    - **Phase 1**:
      - 冻结 `model.model.features` 的参数。
      - 优化器 (例如 `Adam`) 只优化 `model.model.classifier` 的参数。
      - 使用损失函数 (例如 `nn.CrossEntropyLoss`)。
      - 使用学习率调度器 (例如 `CosineAnnealingLR`)。
    - **Phase 2**:
      - 解冻 `model.model.features` 的最后 `phase2_layers_unfreeze` 层参数。
      - 优化器为特征提取器和分类器设置不同的学习率和权重衰减 (L2 正则化)。
      - 继续使用损失函数和学习率调度器。
  - **核心训练循环 (`fer/utils/training.py` 中的 `train_phase` 函数)**:
    - 迭代 epoch，每个 epoch 中包含训练步骤和验证步骤。
    - 计算损失、准确率、精确率、召回率、F1 分数等指标。
    - 记录训练和验证过程中的指标。
    - 根据验证集准确率保存最佳模型检查点。
    - 实现早停逻辑。
  - **检查点管理 (`fer/utils/checkpoint.py`)**:
    - `load_checkpoint`: 加载之前保存的模型状态、优化器状态、调度器状态、当前 epoch 和最佳验证准确率，以支持从中断处继续训练。
    - 训练过程中会定期保存检查点。
  - **结果保存**: 训练完成后，将两个阶段的训练历史（损失、各项指标）和最终评估结果保存到 JSON 文件中。

## 4. 辅助工具 (`fer/utils/`)

`fer/utils/` 目录包含一系列辅助脚本，支持模型训练和评估。

- `__init__.py`: 使 `utils` 成为一个 Python 包。
- `checkpoint.py`: 管理模型的保存和加载检查点。
- `logger.py`: 配置和管理日志记录。
- `training.py`: 包含核心的训练和评估循环逻辑 (`train_epoch`, `validate_epoch`, `train_phase`)。
- `transforms.py`: 定义或组合用于数据预处理和增强的图像变换。
- `visualize.py`: 提供可视化功能，例如绘制训练过程中的损失和准确率曲线、绘制混淆矩阵等。

这些工具共同构成了 `fer` 模块的完整功能，支持从数据准备到模型训练、评估和结果可视化的整个流程。
