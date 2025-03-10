import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

class FERDataset(Dataset):
    def __init__(self, csv_file, transform=None, mode='train'):
        """
        Args:
            csv_file (string): FER2013数据集的CSV文件路径
            transform (callable, optional): 可选的图像转换
            mode (string): 'train', 'val' 或 'test' 模式
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.mode = mode

        # 根据模式选择数据
        if mode == 'train':
            self.data_frame = self.data_frame[self.data_frame['Usage'] == 'Training']
        elif mode == 'val':
            self.data_frame = self.data_frame[self.data_frame['Usage'] == 'PublicTest']
        elif mode == 'test':
            self.data_frame = self.data_frame[self.data_frame['Usage'] == 'PrivateTest']

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),  # 将图像值缩放到 0-1
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

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

def get_data_transforms():
    """
    返回训练和测试数据的转换
    """
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform

def create_datasets(csv_file):
    """
    从FER2013数据集创建训练、验证和测试数据集
    """
    train_transform, test_transform = get_data_transforms()

    train_dataset = FERDataset(
        csv_file=csv_file,
        transform=train_transform,
        mode='train'
    )

    val_dataset = FERDataset(
        csv_file=csv_file,
        transform=test_transform,
        mode='val'
    )

    test_dataset = FERDataset(
        csv_file=csv_file,
        transform=test_transform,
        mode='test'
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    # 测试数据集加载
    csv_path = 'f:/coding/Graduation_project/FER/fer2013.csv'
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_datasets(csv_path)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 测试数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    
    # 获取一个批次的数据并显示其信息
    for images, labels in train_loader:
        print(f"\n批次数据形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签示例: {labels[:5]}")
        
        # 验证数据范围
        print(f"图像数据范围: [{images.min():.3f}, {images.max():.3f}]")
        
        # 只打印第一个批次的信息就退出
        break