import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from fer.config import TrainingConfig
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from fer.dataset import FERDataset, create_dataloaders
from torch.utils.data import DataLoader

def plot_training_history(history_path):
    """绘制训练历史"""
    with open(history_path, 'r') as f:
        results = json.load(f)
    
    history = results['training_history']
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建保存图表的目录
    plot_dir = os.path.join(TrainingConfig.checkpoint_dir, 'plots')
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    for phase in ['phase1', 'phase2']:
        plt.plot(history[phase]['train_loss'], label=f'{phase}训练损失')
        plt.plot(history[phase]['val_loss'], label=f'{phase}验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'loss_curves.png'))
    plt.close()
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    for phase in ['phase1', 'phase2']:
        train_acc = [m['accuracy'] for m in history[phase]['train_metrics']]
        val_acc = [m['accuracy'] for m in history[phase]['val_metrics']]
        plt.plot(train_acc, label=f'{phase}训练准确率')
        plt.plot(val_acc, label=f'{phase}验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'accuracy_curves.png'))
    plt.close()
    
    # 绘制其他指标
    metrics = ['precision', 'recall', 'f1']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for phase in ['phase1', 'phase2']:
            val_metric = [m[metric] for m in history[phase]['val_metrics']]
            plt.plot(val_metric, label=f'{phase}验证{metric}')
        plt.title(f'验证集{metric}指标')
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric} (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'{metric}_curves.png'))
        plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, phase):
    """绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        phase: 阶段名称(phase1/phase2)
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建保存图表的目录
    plot_dir = os.path.join(TrainingConfig.checkpoint_dir, 'plots')
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{phase}混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(os.path.join(plot_dir, f'{phase}_confusion_matrix.png'))
    plt.close()


def plot_test_confusion_matrix(model, class_names, device='cuda'):
    """使用测试集绘制混淆矩阵
    
    参数:
        model: 训练好的模型
        csv_file: 数据集csv文件路径
        class_names: 类别名称列表
        device: 使用的设备(cuda/cpu)
    """
    # 创建测试集
    _, _, test_loader = create_dataloaders(
        TrainingConfig.csv_path,
        TrainingConfig.batch_size,
        TrainingConfig.num_workers
    )
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 调用之前定义的混淆矩阵绘制函数
    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        phase='test'
    )

if __name__ == '__main__':
    import torch
    from fer.model import create_model
    from fer.config import TrainingConfig
    
    # 模型和数据集配置
    csv_file = 'D:\\program\\FER-on-MobilenetV2\\fer2013.csv'
    class_names = ['愤怒', '厌恶', '恐惧', '高兴', '中性', '悲伤', '惊讶']
    
    # 创建模型并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)
    checkpoint = torch.load('D:/program/FER-on-MobilenetV2/final.pth', map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    
    # 绘制测试集混淆矩阵
    plot_test_confusion_matrix(
        model=model,
        class_names=class_names,
        device=device
    )
    