import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import create_datasets
from model import create_model
from config import TrainingConfig
import json
from pathlib import Path
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
import logging
from fer.utils import setup_logger, load_checkpoint, train_phase

def train_model():
    setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'使用设备: {device}')
    
    # 添加 CUDA 性能优化
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    train_dataset, val_dataset, _ = create_datasets(TrainingConfig.csv_path)
    
    # 修改 DataLoader 参数
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
    
    model = create_model(device)
    criterion = nn.CrossEntropyLoss()
    
    # 第一阶段训练
    logging.info("开始第一阶段训练...")
    for param in model.model.features.parameters():
        param.requires_grad = False
    
    optimizer_phase1 = optim.Adam(model.model.classifier.parameters(), lr=TrainingConfig.phase1_lr)
    scheduler_phase1 = CosineAnnealingLR(
        optimizer_phase1,
        T_max=TrainingConfig.phase1_epochs,
        eta_min=TrainingConfig.phase1_lr * 0.01
    )
    
    checkpoint = load_checkpoint(phase=1)
    start_epoch = 0
    best_val_acc = 0
    history_phase1 = None
    
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_phase1.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_phase1.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        history_phase1 = checkpoint['history']
    
    best_val_acc, history_phase1 = train_phase(
        1, model, train_loader, val_loader, criterion, optimizer_phase1, scheduler_phase1,
        TrainingConfig.phase1_epochs, device, start_epoch, best_val_acc, history_phase1
    )
    
    # 第二阶段训练
    logging.info("\n开始第二阶段训练...")
    for param in model.model.features[-TrainingConfig.phase2_layers_unfreeze:].parameters():
        param.requires_grad = True
    
    optimizer_phase2 = optim.Adam([
        {'params': model.model.features[-TrainingConfig.phase2_layers_unfreeze:].parameters(), 'lr': TrainingConfig.phase2_feature_lr},
        {'params': model.model.classifier.parameters(), 'lr': TrainingConfig.phase2_classifier_lr}
    ])
    scheduler_phase2 = CosineAnnealingLR(
        optimizer_phase2,
        T_max=TrainingConfig.phase2_epochs,
        eta_min=TrainingConfig.phase2_feature_lr * 0.01
    )
    
    checkpoint = load_checkpoint(phase=2)
    start_epoch = 0
    best_val_acc = 0
    history_phase2 = None
    
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_phase2.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_phase2.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        history_phase2 = checkpoint['history']
    
    best_val_acc, history_phase2 = train_phase(
        2, model, train_loader, val_loader, criterion, optimizer_phase2, scheduler_phase2,
        TrainingConfig.phase2_epochs, device, start_epoch, best_val_acc, history_phase2
    )
    
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