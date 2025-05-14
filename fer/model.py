import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class EmotionMobileNetV2(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionMobileNetV2, self).__init__()
        
        # 加载预训练的MobileNetV2模型
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # 获取原始分类器的输入特征数
        in_features = self.model.classifier[-1].in_features
        
        # 替换分类器
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def create_model(device):
    model = EmotionMobileNetV2()
    model = model.to(device)
    return model

if __name__ == '__main__':
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)
    
    # 打印模型结构
    print(model)
    
    # 测试前向传播
    test_input = torch.randn(1, 3, 48, 48).to(device)  # 批次大小为3，单通道，48x48图像
    output = model(test_input)
    print(f"\n输出形状: {output.shape}")  # 应该是 [1, 7]