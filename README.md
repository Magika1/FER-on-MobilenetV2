# 基于计算机视觉的会议情绪监测系统

## 目录

- [基于计算机视觉的会议情绪监测系统](#基于计算机视觉的会议情绪监测系统)
  - [目录](#目录)
  - [项目简介](#项目简介)
  - [主要功能](#主要功能)
  - [技术栈](#技术栈)
  - [系统架构](#系统架构)
  - [核心模块](#核心模块)
  - [情绪模型详情](#情绪模型详情)
  - [安装与配置](#安装与配置)
  - [如何运行](#如何运行)

## 项目简介

本项目是一个基于计算机视觉的实时会议情绪监测系统。它通过摄像头捕捉会议参与者的面部表情，利用深度学习模型进行情绪识别，并将结果实时显示在用户界面上。系统旨在帮助分析会议氛围、参与者投入程度等，为会议效果评估提供数据支持。

## 主要功能

- **实时人脸检测**: 从摄像头视频流中准确检测人脸。
- **情绪分类**: 将检测到的人脸表情实时分类为七种基本情绪：生气 (angry), 厌恶 (disgust), 恐惧 (fear), 开心 (happy), 悲伤 (sad), 惊讶 (surprise), 平静 (neutral)。
- **图形用户界面 (GUI)**:
  - 显示实时摄像头画面，并在检测到的人脸周围绘制边界框及对应的情绪标签。
  - 使用 Emoji 直观展示当前主要情绪。
  - 显示系统性能指标，如平均推理时间 (ms) 和帧率 (FPS)。
- **情绪平滑处理**: 对连续帧的情绪识别结果进行平滑处理，使情绪显示更稳定。
- **数据记录**: 支持将情绪变化历史（时间戳、情绪类型、置信度）保存为 CSV 文件，方便后续分析。
- **多摄像头支持**: 用户可选择不同的可用摄像头设备。
- **界面主题切换**: 支持多种界面风格（例如：暗色主题、亮色主题）。
- **错误处理机制**: 包含模型加载、摄像头访问等环节的错误提示与处理。

## 技术栈

- **编程语言**: Python
- **深度学习框架**:
  - PyTorch
  - DeepFace (用于情绪分析)
- **计算机视觉库**: OpenCV (`cv2`)
- **图形用户界面 (GUI)**: PyQt6
- **图像处理**: Pillow (PIL)
- **数据处理**: NumPy (通常作为底层依赖)

## 系统架构

系统的工作流程如下：

1.  **摄像头输入**: `CameraManager` 模块负责从选定的摄像头捕获视频帧。
2.  **人脸检测**: `FaceDetector` 模块处理每一帧图像，使用预训练模型（如 OpenCV 内置的 Haar 特征级联分类器或 DNN 模型）定位图像中的人脸。
3.  **情绪分类**:
    - 检测到的人脸区域 (ROI) 被传递给 `EmotionClassifier` 模块。
    - 当前系统主要通过 `ui/modules/emotion_classifier_new.py` 中的实现，利用 `DeepFace.analyze` 接口进行情绪分析，该接口支持多种后端模型。
    - 项目中也包含一个基于 MobileNetV2 的自定义情绪识别模型实现 (`ui/modules/emotion_classifier.py` 和 `fer/model.py`)。
4.  **情绪平滑**: 为了减少情绪标签的频繁跳动，系统采用时间窗口对连续帧的情绪预测结果进行加权平均平滑处理 (`MainWindow._get_smoothed_emotion` 方法)。
5.  **用户界面展示**: `MainWindow` 及其包含的 `ContentArea` 负责：
    - 显示处理后的视频帧，包括人脸框和情绪标签。
    - 以 Emoji 形式展示最显著的情绪。
    - 实时更新性能统计数据（FPS、推理时间）。
6.  **数据记录与导出**: 系统会记录情绪变化的时间序列数据，用户可以通过界面操作将这些数据导出为 CSV 文件。

## 核心模块

- `main.py`: 应用程序的启动入口。
- `fer/`: 包含面部情绪识别的核心深度学习模型和相关脚本。
  - `model.py`: 定义了 `EmotionMobileNetV2` 模型结构。
  - `train.py`: (推测) 用于训练情绪识别模型的脚本。
- `ui/`: 负责图形用户界面的所有组件和逻辑。
  - `widgets/main_window.py`: 主应用窗口，协调各个 UI 组件和核心处理逻辑。
  - `widgets/components/`: 可复用的 UI 组件，如 `ContentArea` (内容显示区), `MenuBar` (菜单栏), `CameraManager` (摄像头管理器)。
  - `modules/`: 核心处理模块。
    - `face_detector.py`: 实现人脸检测功能。
    - `emotion_classifier_new.py`: 使用 DeepFace 实现情绪分类。
    - `emotion_classifier.py`: 使用自定义的 PyTorch MobileNetV2 模型实现情绪分类。
  - `assets/`: 包含静态资源，如 `constants.py` (情绪映射、UI 常量) 和 `styles/` (界面样式表)。
  - `utils/`: UI 相关的辅助工具，如 `style_loader.py` (样式加载器)。
- `utils/`: 通用辅助模块。
  - `error_handler.py`: 统一的错误处理机制。
- `requirements.txt`: 项目依赖的 Python 包列表。
- `final.pth`: (推测) 自定义 MobileNetV2 情绪分类模型的预训练权重文件。

## 情绪模型详情

- 系统当前主要依赖 `DeepFace` 库进行情绪分类。`DeepFace` 是一个功能强大的面部分析库，支持多种先进的深度学习模型作为后端，能够提供鲁棒的情绪识别性能。
- 项目中也提供了一个基于 `MobileNetV2` 的自定义情绪识别模型（定义于 `fer/model.py`）。该模型针对 7 种情绪类别进行训练：生气 (angry), 厌恶 (disgust), 恐惧 (fear), 开心 (happy), 悲伤 (sad), 惊讶 (surprise), 平静 (neutral)。
- 对于自定义模型，输入图像会经过预处理，包括缩放至 256x256，中心裁剪至 224x224，转换为 PyTorch 张量，并进行归一化。

## 安装与配置

1.  **克隆仓库**:
    ```bash
    git clone <your_repository_url>
    cd FER-on-MobilenetV2
    ```
2.  **创建并激活虚拟环境** (推荐):
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    # source venv/bin/activate
    ```
3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **模型文件**:
    - 如果希望使用自定义的 `MobileNetV2` 模型 (通过 `ui/modules/emotion_classifier.py`)，请确保预训练的权重文件 (例如 `final.pth`) 放置在项目根目录，或代码中指定的路径。
    - `DeepFace` 会在首次使用时自动下载其所需的模型文件。

## 如何运行

执行以下命令启动应用程序：

```bash
python main.py
```
