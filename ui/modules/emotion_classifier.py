import cv2
import torch
import time
import torchvision.transforms as transforms
from PIL import Image
from fer.model import create_model
from utils.error_handler import ErrorHandler, ErrorType
from PyQt6.QtWidgets import QMessageBox, QFileDialog

class EmotionClassifier:
    def __init__(self, parent=None):
        self.parent = parent
        # 加载MobileNetV2模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(self.device)
        try:
            # 加载模型权重
            checkpoint = torch.load('D:/program/FER-on-MobilenetV2/final.pth', 
                                  map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except FileNotFoundError:
            # 先提示未找到模型文件
            reply = QMessageBox.question(
                self.parent,
                "模型文件未找到",
                "未找到默认模型文件，是否要选择其他模型文件？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # 让用户选择模型文件
                model_path, _ = QFileDialog.getOpenFileName(
                    self.parent,
                    "选择模型文件",
                    "",
                    "PyTorch模型文件 (*.pth *.pt);;所有文件 (*.*)"
                )
                if model_path:
                    try:
                        checkpoint = torch.load(model_path, 
                                             map_location=self.device, weights_only=True)
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.model.eval()
                    except Exception as e:
                        ErrorHandler.show_error(ErrorType.MODEL_LOAD, 
                                             f"模型加载失败: {str(e)}", self.parent)
                        raise
                else:
                    ErrorHandler.show_error(ErrorType.MODEL_LOAD, 
                                         "未选择模型文件，程序将退出", self.parent)
                    raise FileNotFoundError("用户未选择模型文件")
            else:
                ErrorHandler.show_error(ErrorType.MODEL_LOAD, 
                                     "用户选择退出程序", self.parent)
                raise FileNotFoundError("用户选择退出程序")
        except Exception as e:
            ErrorHandler.show_error(ErrorType.MODEL_LOAD, f"模型加载失败: {str(e)}")
            raise
            
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # 将图像值缩放到 0-1
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
        
        # 定义情绪标签
        self.labels = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }
        
        # 性能监控参数
        self.inference_times = []
        self.max_times_stored = 30
        
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.inference_times:
            return {
                'avg_inference_time': 0,
                'fps': 0,
                'device': str(self.device)
            }
            
        avg_time = sum(self.inference_times) / len(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time': round(avg_time * 1000, 2),  # 转换为毫秒并保留两位小数
            'fps': round(fps, 1),
            'device': str(self.device)
        }
        
    def classify(self, frame, faces):
        emotions = []
        start_time = time.time()
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
            
            # 转换为PIL图像
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            
            # 应用预处理
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # 进行推理
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # 返回所有情绪的概率
                emotion_probs = {self.labels[i]: prob.item() for i, prob in enumerate(probs)}
                # 获取最高概率的情绪
                pred_idx = torch.argmax(probs).item()
                max_emotion = self.labels[pred_idx]
                max_prob = probs[pred_idx].item()
                
            emotions.append((max_emotion, max_prob, emotion_probs))
        
        # 记录推理时间
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.max_times_stored:
            self.inference_times.pop(0)
            
        return emotions