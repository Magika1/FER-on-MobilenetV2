import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from fer.utils.transforms import get_data_transforms

class FERDataset(Dataset):
    def __init__(self, csv_file, transform=None, mode='train'):

        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.mode = mode

        # 根据模式选择数据
        usage_map = {
            'train': 'Training',
            'val': 'PublicTest',
            'test': 'PrivateTest'
        }
        self.data_frame = self.data_frame[self.data_frame['Usage'] == usage_map[mode]]

        if self.transform is None:
            _, test_transform = get_data_transforms()
            self.transform = test_transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 从pixels字符串转换为图像
        pixels = self.data_frame.iloc[idx]['pixels'].split()
        pixels = np.array([int(pixel) for pixel in pixels], dtype='uint8')
        pixels = pixels.reshape(48, 48)
        
        # 将灰度图像转换为RGB图像
        image = Image.fromarray(pixels).convert('RGB')
        label = self.data_frame.iloc[idx]['emotion']

        if self.transform:
            image = self.transform(image)

        return image, label

def create_datasets(csv_file):
    """创建训练、验证和测试数据集"""
    train_transform, test_transform = get_data_transforms()

    datasets = {
        'train': FERDataset(csv_file=csv_file, transform=train_transform, mode='train'),
        'val': FERDataset(csv_file=csv_file, transform=test_transform, mode='val'),
        'test': FERDataset(csv_file=csv_file, transform=test_transform, mode='test')
    }

    return datasets['train'], datasets['val'], datasets['test']

def create_dataloaders(csv_file, batch_size, num_workers):
    """创建数据加载器"""
    train_dataset, val_dataset, test_dataset = create_datasets(csv_file)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader
    