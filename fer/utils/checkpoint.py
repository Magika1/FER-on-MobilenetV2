import torch
from pathlib import Path
import os
import logging
from ..config import TrainingConfig

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
    best_path = os.path.join(TrainingConfig.checkpoint_dir, f'phase{phase}_best.pth')
    latest_path = os.path.join(TrainingConfig.checkpoint_dir, f'phase{phase}_latest.pth')
    
    if os.path.exists(best_path):
        logging.info(f'加载最佳检查点: {best_path}')
        return torch.load(best_path)
    elif os.path.exists(latest_path):
        logging.info(f'加载最新检查点: {latest_path}')
        return torch.load(latest_path)
    return None