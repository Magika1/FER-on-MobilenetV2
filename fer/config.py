class TrainingConfig:
    # 数据集配置
    csv_path = 'f:/coding/Graduation_project/fer/fer2013.csv'
    batch_size = 64
    num_workers = 4

    # 模型配置
    num_classes = 7

    # 第一阶段训练配置
    phase1_epochs = 100
    phase1_lr = 0.001

    # 第二阶段训练配置
    phase2_epochs = 200
    phase2_feature_lr = 0.0001
    phase2_classifier_lr = 0.0005
    phase2_layers_unfreeze = 6  # 解冻的层数

    # 修改模型保存相关配置
    checkpoint_dir = 'f:/coding/Graduation_project/fer/checkpoints'
    log_dir = 'f:/coding/Graduation_project/fer/logs'
    save_freq = 5  # 每隔多少个epoch保存一次检查点
    
    # 早停参数
    patience = 10
    min_delta = 0.001
    # 实验配置
    experiment_name = "baseline"  # 当前实验名称
    experiments = {
        "exp1": {
            "name": "light_finetune",
            "layers_unfreeze": 6,
            "feature_lr": 1e-4,
            "classifier_lr": 1e-3
        },
        "exp2": {
            "name": "medium_finetune",
            "layers_unfreeze": 10,
            "feature_lr": 5e-5,
            "classifier_lr": 5e-4
        },
        "exp3": {
            "name": "deep_finetune",
            "layers_unfreeze": 15,
            "feature_lr": 1e-5,
            "classifier_lr": 1e-4
        }
    }
