import logging
from pathlib import Path
import os
from datetime import datetime
from ..config import TrainingConfig

def setup_logger():
    """设置日志记录"""
    Path(TrainingConfig.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(TrainingConfig.log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)