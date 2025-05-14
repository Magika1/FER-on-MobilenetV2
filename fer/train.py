import torch
import torch.nn as nn
import torch.optim as optim
from fer.dataset import create_datasets, create_dataloaders
from fer.model import create_model
from fer.config import TrainingConfig
import json
from pathlib import Path
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
import logging
from fer.utils.logger import setup_logger
from fer.utils.checkpoint import load_checkpoint
from fer.utils.training import train_phase


def train_phase1(model, train_loader, val_loader, device):
    """第一阶段训练：仅训练分类器部分"""
    logging.info("开始第一阶段训练...")
    
    # 冻结特征提取器
    for param in model.model.features.parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.model.classifier.parameters(), 
        lr=TrainingConfig.phase1_lr,
        weight_decay=1e-4  # 添加L2正则化
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=TrainingConfig.phase1_epochs,
        eta_min=TrainingConfig.phase1_lr * 0.01
    )
    
    checkpoint = load_checkpoint(phase=1)
    start_epoch = 0
    best_val_acc = 0
    history = None
    
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        history = checkpoint['history']
    
    best_val_acc, history = train_phase(
        1, model, train_loader, val_loader, criterion, optimizer, scheduler,
        TrainingConfig.phase1_epochs, device, start_epoch, best_val_acc, history
    )
    
    return model, best_val_acc, history

def train_phase2(model, train_loader, val_loader, device):
    """第二阶段训练：解冻部分特征提取器并微调"""
    logging.info("\n开始第二阶段训练...")
    
    # 解冻指定层数
    for param in model.model.features[-TrainingConfig.phase2_layers_unfreeze:].parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {
            'params': model.model.features[-TrainingConfig.phase2_layers_unfreeze:].parameters(), 
            'lr': TrainingConfig.phase2_feature_lr,
            'weight_decay': 1e-4  # 为特征提取器添加L2正则化
        },
        {
            'params': model.model.classifier.parameters(), 
            'lr': TrainingConfig.phase2_classifier_lr,
            'weight_decay': 1e-4  # 为分类器添加L2正则化
        }
    ])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=TrainingConfig.phase2_epochs,
        eta_min=TrainingConfig.phase2_feature_lr * 0.01
    )
    
    checkpoint = load_checkpoint(phase=2)
    start_epoch = 0
    best_val_acc = 0
    history = None
    
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        history = checkpoint['history']
    
    best_val_acc, history = train_phase(
        2, model, train_loader, val_loader, criterion, optimizer, scheduler,
        TrainingConfig.phase2_epochs, device, start_epoch, best_val_acc, history
    )
    
    return model, best_val_acc, history

def train_model():
    """主训练函数，协调整个训练流程"""
    setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'使用设备: {device}')
    
    # 添加 CUDA 性能优化
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    train_dataset, val_dataset, _ = create_datasets(TrainingConfig.csv_path)
    
    # 创建数据加载器
    train_loader, val_loader, _ = create_dataloaders(
        TrainingConfig.csv_path,
        TrainingConfig.batch_size,
        TrainingConfig.num_workers
    )
    
    # 创建模型
    model = create_model(device)
    # 执行第一阶段训练
    model, _, history_phase1 = train_phase1(model, train_loader, val_loader, device)
    
    # 执行第二阶段训练
    model, _, history_phase2 = train_phase2(model, train_loader, val_loader, device)
    
    # 保存完整训练历史
    # 保存完整训练历史和最终评估结果
    final_results = {
        'training_history': {
            'phase1': history_phase1,
            'phase2': history_phase2
        },
        'final_metrics': {
            'phase1': history_phase1['val_metrics'][-1],
            'phase2': history_phase2['val_metrics'][-1]
        }
    }
    
    
    Path(TrainingConfig.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存训练历史
    with open(os.path.join(TrainingConfig.checkpoint_dir, 'training_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
    
    # 打印最终结果
    logging.info("\n最终评估结果:")
    logging.info("第一阶段:")
    for metric, value in final_results['final_metrics']['phase1'].items():
        logging.info(f"{metric}: {value:.2f}%")
    
    logging.info("\n第二阶段:")
    for metric, value in final_results['final_metrics']['phase2'].items():
        logging.info(f"{metric}: {value:.2f}%")
    
    logging.info("训练完成！")

if __name__ == '__main__':
    train_model()