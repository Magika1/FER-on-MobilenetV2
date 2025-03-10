import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import create_datasets
from model import create_model
from config import TrainingConfig
import time
from tqdm import tqdm
import os
import json
from pathlib import Path
import logging
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn

def setup_logger():
    Path(TrainingConfig.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(TrainingConfig.log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'Loss': running_loss/(batch_idx+1),
            'Acc': 100.*correct/total
        })
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss/len(val_loader), 100.*correct/total, all_predictions, all_labels

def save_checkpoint(state, is_best, phase, filename='checkpoint.pth'):
    Path(TrainingConfig.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存最新的检查点
    checkpoint_path = os.path.join(TrainingConfig.checkpoint_dir, f'phase{phase}_latest.pth')
    torch.save(state, checkpoint_path)
    
    # 如果是最佳模型，单独保存一份
    if is_best:
        best_path = os.path.join(TrainingConfig.checkpoint_dir, f'phase{phase}_best.pth')
        torch.save(state, best_path)
        logging.info(f'保存最佳模型，验证准确率: {state["best_val_acc"]:.2f}%')

def load_checkpoint(phase, filename='latest.pth'):
    # 优先尝试加载最佳模型
    best_path = os.path.join(TrainingConfig.checkpoint_dir, f'phase{phase}_best.pth')
    latest_path = os.path.join(TrainingConfig.checkpoint_dir, f'phase{phase}_latest.pth')
    
    if os.path.exists(best_path):
        logging.info(f'加载最佳检查点: {best_path}')
        return torch.load(best_path)
    elif os.path.exists(latest_path):
        logging.info(f'加载最新检查点: {latest_path}')
        return torch.load(latest_path)
    return None

def train_phase(phase, model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, start_epoch=0, best_val_acc=0, history=None):
    if history is None:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    patience_counter = 0  # 早停计数器
    
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, predictions, labels = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'当前学习率: {current_lr:.6f}')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logging.info(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 检查是否有性能提升
        if val_acc - best_val_acc > TrainingConfig.min_delta:
            is_best = True
            best_val_acc = val_acc
            patience_counter = 0  # 重置计数器
        else:
            is_best = False
            patience_counter += 1  # 增加计数器
            
        if (epoch + 1) % TrainingConfig.save_freq == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }, is_best, phase)
        
        # 检查是否需要早停
        if patience_counter >= TrainingConfig.patience:
            logging.info(f'验证集性能已经 {TrainingConfig.patience} 个epoch没有提升，停止训练')
            break
    
    return best_val_acc, history

def train_model():
    setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch_directml.device()
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
        num_workers=4,  # 增加工作进程数
        pin_memory=True,  # 启用内存钉扎
        prefetch_factor=2,  # 预加载因子
        persistent_workers=True  # 保持工作进程存活
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TrainingConfig.batch_size, 
        shuffle=False, 
        num_workers=4,
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
        eta_min=TrainingConfig.phase1_lr * 0.01  # 最小学习率为初始学习率的1%
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
        eta_min=TrainingConfig.phase2_feature_lr * 0.01  # 最小学习率为初始学习率的1%
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