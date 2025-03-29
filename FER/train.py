import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from fer.dataset import create_datasets
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
    optimizer = optim.Adam(model.model.classifier.parameters(), lr=TrainingConfig.phase1_lr)
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
    
    # # 打印解冻前的参数状态
    # logging.info(f"第二阶段将解冻最后 {TrainingConfig.phase2_layers_unfreeze} 层特征提取器")
    # unfrozen_count = 0
    # total_layers = len(list(model.model.features))
    # logging.info(f"特征提取器总层数: {total_layers}")
    
    # 解冻指定层数
    for param in model.model.features[-TrainingConfig.phase2_layers_unfreeze:].parameters():
        param.requires_grad = True
        # unfrozen_count += 1
    
    # # 验证解冻状态
    # logging.info(f"已解冻参数数量: {unfrozen_count}")
    # logging.info("解冻的层:")
    # for i, layer in enumerate(model.model.features):
    #     if i >= total_layers - TrainingConfig.phase2_layers_unfreeze:
    #         params_count = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    #         logging.info(f"  - 层 {i}: {layer.__class__.__name__}, 可训练参数: {params_count}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.model.features[-TrainingConfig.phase2_layers_unfreeze:].parameters(), 'lr': TrainingConfig.phase2_feature_lr},
        {'params': model.model.classifier.parameters(), 'lr': TrainingConfig.phase2_classifier_lr}
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
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TrainingConfig.batch_size, 
        shuffle=True, 
        num_workers=TrainingConfig.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TrainingConfig.batch_size, 
        shuffle=False, 
        num_workers=TrainingConfig.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # 创建模型
    model = create_model(device)
    # 执行第一阶段训练
    model, _, history_phase1 = train_phase1(model, train_loader, val_loader, device)
    
    # 执行第二阶段训练
    model, _, history_phase2 = train_phase2(model, train_loader, val_loader, device)
    
    # 保存完整训练历史
    history = {
        'phase1': history_phase1,
        'phase2': history_phase2
    }
    Path(TrainingConfig.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(TrainingConfig.checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    logging.info("训练完成！")

if __name__ == '__main__':
    train_model()