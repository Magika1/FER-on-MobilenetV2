import cv2
from collections import deque
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QMessageBox
from PyQt6.QtCore import QTimer, pyqtSignal
from ui.modules import FaceDetector, EmotionClassifier
from ui.utils import load_stylesheet
from ui.assets import (
    EMOTION_MAPPING, EMOTION_NAMES, EMOJI_MAPPING,
    WINDOW_TITLE, WINDOW_GEOMETRY, WINDOW_MIN_HEIGHT
)
from .components import MenuBar, ContentArea, CameraManager
from utils.error_handler import ErrorHandler, ErrorType

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.error_handler = ErrorHandler()
        # 初始化数据存储
        self.emotion_times = deque(maxlen=30)
        self.emotion_values = deque(maxlen=30)
        self.emotion_labels = deque(maxlen=30)
        
        # 添加情绪平滑相关变量
        self.emotion_window = deque(maxlen=10)  # 存储最近10帧的情绪
        self.current_emotion = None  # 当前显示的情绪
        self.emotion_threshold = 0.6  # 情绪切换阈值
        
        # 修改情绪统计数据结构
        self.emotion_history = []  # 存储时间序列数据，每项格式为 (timestamp, emotion, confidence)
        self.max_history_size = 1000  # 最多保存1000个数据点
        self.stats_interval = 5  # 每5帧记录一次
        self.total_frames = 0  # 添加帧计数器初始化
        
        # 初始化模块
        try:
            self.face_detector = FaceDetector()
        except Exception as e:
            self.error_handler.show_error(
                ErrorType.MODEL_LOAD,
                f"人脸检测模型加载失败: {str(e)}",
                self
            )
            raise

        self.emotion_classifier = EmotionClassifier(self)  # 传递self作为parent

        self.camera_manager = CameraManager()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

        # 初始化UI
        self.setStyleSheet(load_stylesheet('dark', self.error_handler))
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(*WINDOW_GEOMETRY)
        self.setMinimumHeight(WINDOW_MIN_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建UI组件
        self.menu_bar = MenuBar(self)
        self.content_area = ContentArea(self)
        layout.addWidget(self.menu_bar)
        layout.addWidget(self.content_area)

        # 设置信号连接
        self.setup_signals()
        # 启动摄像头
        self.camera_manager.start_camera()
        self.timer.start(30)
        
        # 添加样式相关变量
        self.current_style = 'dark'

    def setup_signals(self):
        # 摄像头相关信号
        self.menu_bar.camera_changed.connect(self.on_camera_changed)
        self.camera_manager.camera_error.connect(self.on_camera_error)
        self.camera_manager.cameras_updated.connect(self.menu_bar.update_camera_list)
        # 样式切换信号
        self.menu_bar.style_changed.connect(self.on_style_changed)
        # 初始扫描摄像头
        self.camera_manager.scan_cameras()

    def on_style_changed(self, style_name):
        """处理样式切换事件"""
        self.current_style = style_name
        self.setStyleSheet(load_stylesheet(style_name, self.error_handler))

    @staticmethod
    def _draw_detection_results(frame, faces, emotions):
        """在图像上绘制检测结果"""
        if not faces or not emotions:
            return

        for bbox, (emotion, score) in zip(faces, emotions):
            x, y, w, h = bbox
            cv2.rectangle(frame, bbox, (0, 255, 0), 1)

            label = f"{emotion}: {score:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def on_camera_changed(self, camera_id):
        """处理摄像头切换事件"""
        self.camera_manager.start_camera(camera_id)
        
    def on_camera_error(self, error_msg):
        """处理摄像头错误"""
        QMessageBox.critical(self, "错误", error_msg)
        
    def closeEvent(self, event):
        """关闭窗口事件"""
        self.timer.stop()
        self.camera_manager.release()
        event.accept()

    def _get_smoothed_emotion(self, current_emotions):
        """使用加权平均获取平滑处理后的情绪预测结果"""
        if not current_emotions:
            return []
            
        # 将当前帧的情绪结果添加到窗口中
        self.emotion_window.append(current_emotions)
        
        # 如果窗口中的样本太少，直接返回当前帧的结果
        if len(self.emotion_window) < 3:
            return [(emotion, prob) for emotion, prob, _ in current_emotions]
            
        smoothed_emotions = []
        # 对每个检测到的人脸进行处理
        for face_idx in range(len(current_emotions)):
            # 收集窗口中该位置的所有情绪概率
            all_probs = {}
            # 使用指数衰减权重，最近的帧权重最大
            total_weight = 0
            
            for frame_idx, frame_emotions in enumerate(self.emotion_window):
                if face_idx >= len(frame_emotions):
                    continue
                    
                weight = 0.8 ** (len(self.emotion_window) - 1 - frame_idx)  # 指数衰减权重
                total_weight += weight
                
                # 累加加权概率
                _, _, emotion_probs = frame_emotions[face_idx]
                for emotion, prob in emotion_probs.items():
                    all_probs[emotion] = all_probs.get(emotion, 0) + prob * weight
            
            # 如果有有效数据
            if total_weight > 0:
                # 归一化概率
                for emotion in all_probs:
                    all_probs[emotion] /= total_weight
                    
                # 获取概率最高的情绪
                max_emotion = max(all_probs.items(), key=lambda x: x[1])
                smoothed_emotions.append((max_emotion[0], max_emotion[1]))
            else:
                # 如果没有有效数据，使用当前帧的结果
                emotion, prob, _ = current_emotions[face_idx]
                smoothed_emotions.append((emotion, prob))
                
        return smoothed_emotions


    def process_frame(self):
        """处理摄像头帧"""
        try:
            frame = self.camera_manager.get_frame()
            if frame is None:
                return
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                faces, face_detected = self.face_detector.detect(frame)
            except Exception as e:
                self.error_handler.show_error(
                    ErrorType.MODEL_INFERENCE,
                    f"人脸检测失败: {str(e)}",
                    self
                )
                return
                
            try:
                emotions = self.emotion_classifier.classify(frame, faces)
                stats = self.emotion_classifier.get_performance_stats()
                smoothed_emotions = self._get_smoothed_emotion(emotions)
                
                # 每隔一定帧数记录情绪数据
                self.total_frames += 1
                if self.total_frames % self.stats_interval == 0 and smoothed_emotions:
                    # 只记录第一个检测到的人脸的情绪
                    emotion, confidence = smoothed_emotions[0]
                    timestamp = self.total_frames * (1.0 / 30)  # 假设30fps，计算时间戳
                    self.emotion_history.append((timestamp, EMOTION_NAMES[EMOTION_MAPPING[emotion]], confidence))
                    # 保持历史记录在限定大小内
                    if len(self.emotion_history) > self.max_history_size:
                        self.emotion_history.pop(0)
                        
            except Exception as e:
                self.error_handler.show_error(
                    ErrorType.MODEL_INFERENCE,
                    f"表情识别失败: {str(e)}",
                    self
                )
                return
            
            # 使用平滑后的情绪结果进行绘制和更新UI
            self._draw_detection_results(frame_rgb, faces, smoothed_emotions)
            # 更新UI时传递情绪历史数据
            self.content_area.update_ui({
                'frame': frame_rgb,
                'faces': faces,
                'face_detected': face_detected,
                'emotions': smoothed_emotions,
                'stats': stats,
                'emotion_history': self.emotion_history
            })
            
        except Exception as e:
            self.error_handler.show_error(
                ErrorType.SYSTEM,
                f"处理视频帧时发生错误: {str(e)}",
                self
            )
            self.timer.stop()