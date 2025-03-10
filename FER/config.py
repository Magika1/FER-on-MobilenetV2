class TrainingConfig:
    # 数据集配置
    csv_path = 'f:/coding/Graduation_project/FER/fer2013.csv'
    batch_size = 32
    num_workers = 2

    # 模型配置
    num_classes = 7

    # 第一阶段训练配置
    phase1_epochs = 10
    phase1_lr = 0.001

    # 第二阶段训练配置
    phase2_epochs = 20
    phase2_feature_lr = 0.0001
    phase2_classifier_lr = 0.0005
    phase2_layers_unfreeze = 10  # 解冻的层数

    # 模型保存路径
    model_phase1_path = 'f:/coding/Graduation_project/FER/best_model_phase1.pth'
    model_phase2_path = 'f:/coding/Graduation_project/FER/best_model_phase2.pth'

    # 新增配置
    checkpoint_dir = 'f:/coding/Graduation_project/FER/checkpoints'
    log_dir = 'f:/coding/Graduation_project/FER/logs'
    save_freq = 5  # 每隔多少个epoch保存一次检查点
