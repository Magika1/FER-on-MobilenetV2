import torch
from tqdm import tqdm
import logging
from .checkpoint import save_checkpoint
from fer.config import TrainingConfig 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def calculate_metrics(y_true, y_pred):
    """计算准确率、精确率、召回率和F1分数"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        'accuracy': accuracy * 100,  # 转换为百分比
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
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
    """验证模型"""
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
    
    metrics = calculate_metrics(all_labels, all_predictions)
    return running_loss/len(val_loader), metrics, all_predictions, all_labels

def train_phase(phase, model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, start_epoch=0, best_val_acc=0, history=None):
    """训练一个阶段"""
    if history is None:
        history = {
            'train_loss': [], 'train_metrics': [],
            'val_loss': [], 'val_metrics': []
        }
    
    patience_counter = 0  # 早停计数器
    
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_metrics, predictions, labels = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'当前学习率: {current_lr:.6f}')
        
        history['train_loss'].append(train_loss)
        history['train_metrics'].append({'accuracy': train_acc})
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)

        # 使用验证集准确率作为主要指标
        val_acc = val_metrics['accuracy']
        
        logging.info(
            f'Epoch {epoch+1}: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
            f'Val Precision: {val_metrics["precision"]:.2f}%, '
            f'Val Recall: {val_metrics["recall"]:.2f}%, '
            f'Val F1: {val_metrics["f1"]:.2f}%'
        )
        
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