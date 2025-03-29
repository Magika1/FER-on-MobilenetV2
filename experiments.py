import os
from pathlib import Path
import shutil
from fer.train import train_model
from fer.config import TrainingConfig
import logging
from fer.utils.logger import setup_logger

def run_experiment(exp_config):
    """运行单个实验"""
    # 保存原始配置
    original_checkpoint_dir = TrainingConfig.checkpoint_dir
    original_log_dir = TrainingConfig.log_dir
    original_layers_unfreeze = TrainingConfig.phase2_layers_unfreeze
    original_feature_lr = TrainingConfig.phase2_feature_lr
    original_classifier_lr = TrainingConfig.phase2_classifier_lr
    
    try:
        # 更新实验配置
        exp_name = exp_config["name"]
        TrainingConfig.experiment_name = exp_name
        TrainingConfig.checkpoint_dir = os.path.join(original_checkpoint_dir, exp_name)
        TrainingConfig.log_dir = os.path.join(original_log_dir, exp_name)
        TrainingConfig.phase2_layers_unfreeze = exp_config["layers_unfreeze"]
        TrainingConfig.phase2_feature_lr = exp_config["feature_lr"]
        TrainingConfig.phase2_classifier_lr = exp_config["classifier_lr"]
        
        # 创建实验目录
        Path(TrainingConfig.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(TrainingConfig.log_dir).mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        setup_logger()
        logging.info(f"\n{'='*50}")
        logging.info(f"开始实验: {exp_name}")
        logging.info(f"解冻层数: {exp_config['layers_unfreeze']}")
        logging.info(f"特征提取器学习率: {exp_config['feature_lr']}")
        logging.info(f"分类器学习率: {exp_config['classifier_lr']}")
        logging.info(f"{'='*50}\n")
        
        # 运行训练
        train_model()
        
        logging.info(f"\n{'='*50}")
        logging.info(f"实验 {exp_name} 完成!")
        logging.info(f"{'='*50}\n")
        
    finally:
        # 恢复原始配置
        TrainingConfig.checkpoint_dir = original_checkpoint_dir
        TrainingConfig.log_dir = original_log_dir
        TrainingConfig.phase2_layers_unfreeze = original_layers_unfreeze
        TrainingConfig.phase2_feature_lr = original_feature_lr
        TrainingConfig.phase2_classifier_lr = original_classifier_lr

def run_all_experiments():
    """运行所有实验"""
    for exp_id, exp_config in TrainingConfig.experiments.items():
        run_experiment(exp_config)

if __name__ == "__main__":
    run_all_experiments()