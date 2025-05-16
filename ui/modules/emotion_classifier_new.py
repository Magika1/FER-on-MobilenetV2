import time
from deepface import DeepFace
import cv2
from PyQt6.QtWidgets import QMessageBox

class EmotionClassifier:
    def __init__(self, parent=None):
        self.parent = parent
        # 性能监控参数
        self.inference_times = []
        self.max_times_stored = 30
        
        # 定义情绪标签映射
        self.labels = {
            'angry': 'angry',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.inference_times:
            return {
                'avg_inference_time': 0,
                'fps': 0,
                'device': 'CPU'  # DeepFace默认使用CPU
            }
            
        avg_time = sum(self.inference_times) / len(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time': round(avg_time * 1000, 2),  # 转换为毫秒并保留两位小数
            'fps': round(fps, 1),
            'device': 'CPU'
        }
        
    def classify(self, frame, faces):
        emotions = []
        start_time = time.time()
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
                
            try:
                # 使用DeepFace进行情绪分析
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                
                if isinstance(result, list):
                    result = result[0]
                    
                # 获取情绪概率字典
                emotion_probs = result['emotion']
                # 获取最高概率的情绪
                max_emotion = result['dominant_emotion']
                max_prob = emotion_probs[max_emotion] / 100.0  # 转换为0-1范围
                
                # 转换概率字典中的值为0-1范围
                emotion_probs = {k: v/100.0 for k, v in emotion_probs.items()}
                
                emotions.append((self.labels[max_emotion], max_prob, emotion_probs))
                
            except Exception as e:
                print(f"情绪分析失败: {str(e)}")
                continue
        
        # 记录推理时间
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.max_times_stored:
            self.inference_times.pop(0)
            
        return emotions