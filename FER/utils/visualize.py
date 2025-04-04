import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from fer.config import TrainingConfig

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

