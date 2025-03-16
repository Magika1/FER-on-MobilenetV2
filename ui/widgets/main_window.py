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
        
        # 初始化UI和模块
        self.init_modules()
        self.init_ui()
        self.setup_signals()
        
        # 启动摄像头
        self.camera_manager.start_camera()
        self.timer.start(30)
        
    def init_modules(self):
        """初始化功能模块"""
        try:
            self.face_detector = FaceDetector()
        except Exception as e:
            self.error_handler.show_error(
                ErrorType.MODEL_LOAD,
                f"人脸检测模型加载失败: {str(e)}",
                self
            )
            raise
            
        try:
            self.emotion_classifier = EmotionClassifier()
        except Exception as e:
            self.error_handler.show_error(
                ErrorType.MODEL_LOAD,
                f"表情识别模型加载失败: {str(e)}",
                self
            )
            raise
            
        self.camera_manager = CameraManager()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        
    def init_ui(self):
        """初始化UI"""
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
        
    def setup_signals(self):
        """设置信号连接"""
        # 摄像头相关信号
        self.menu_bar.camera_changed.connect(self.on_camera_changed)
        self.camera_manager.camera_error.connect(self.on_camera_error)
        self.camera_manager.cameras_updated.connect(self.menu_bar.update_camera_list)
        # 初始扫描摄像头
        self.camera_manager.scan_cameras()
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
            except Exception as e:
                self.error_handler.show_error(
                    ErrorType.MODEL_INFERENCE,
                    f"表情识别失败: {str(e)}",
                    self
                )
                return
            
            self._draw_detection_results(frame_rgb, faces, emotions)
            self.content_area.update_ui({
                'frame': frame_rgb,
                'faces': faces,
                'face_detected': face_detected,
                'emotions': emotions,
                'stats': stats
            })
            
        except Exception as e:
            self.error_handler.show_error(
                ErrorType.SYSTEM,
                f"处理视频帧时发生错误: {str(e)}",
                self
            )
            self.timer.stop()
        
    def _draw_detection_results(self, frame, faces, emotions):
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