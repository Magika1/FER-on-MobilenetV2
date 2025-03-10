import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, 
                           QMessageBox, QGridLayout, QComboBox)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from face_detector import FaceDetector
from emotion_classifier import EmotionClassifier


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_camera = 0
        self.cap = None
        self.initUI()
        self.setupCamera()
        self.face_detector = FaceDetector()
        self.emotion_classifier = EmotionClassifier()

    def setupCamera(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.current_camera)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", f"无法打开摄像头 {self.current_camera}！")
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not hasattr(self, 'timer'):
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def initUI(self):
        self.setWindowTitle("实时摄像头 - 人脸检测")
        self.setGeometry(100, 100, 1280, 720)

        # 创建主布局
        main_layout = QGridLayout()
        self.setLayout(main_layout)
        
        # 主显示标签
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 状态标签
        self.status_label = QLabel("未检测到人脸")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: red;
                background-color: rgba(0, 0, 0, 0.5);
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.status_label.setFixedSize(250, 50)
        
        # 性能信息标签
        self.performance_label = QLabel()
        self.performance_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: white;
                background-color: rgba(0, 0, 0, 0.5);
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.performance_label.setFixedSize(270, 80)
        
        # 添加摄像头选择下拉框
        self.camera_combo = QComboBox()
        self.camera_combo.setStyleSheet("""
            QComboBox {
                font-size: 16px;
                color: white;
                background-color: rgba(0, 0, 0, 0.5);
                padding: 5px;
                border-radius: 5px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)
        self.camera_combo.setFixedSize(150, 30)
        
        # 获取可用摄像头
        self.refresh_camera_list()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        
        # 添加到布局
        
        # 使用网格布局
        main_layout.addWidget(self.display_label, 0, 0, 3, 3)  # 占据整个网格
        main_layout.addWidget(self.status_label, 0, 2, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(self.camera_combo, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        main_layout.addWidget(self.performance_label, 2, 2, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        
        # 设置布局间距
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
    
    def refresh_camera_list(self):
        self.camera_combo.clear()
        # 检测系统中的摄像头
        for i in range(5):  # 检查前5个摄像头索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                self.camera_combo.addItem(f"摄像头 {i}", i)

    def on_camera_changed(self, index):
        if index >= 0:
            self.current_camera = self.camera_combo.currentData()
            self.setupCamera()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        event.accept()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 人脸检测
        faces, face_detected = self.face_detector.detect(frame)
        
        # 表情分类
        emotions = self.emotion_classifier.classify(frame, faces)
        
        # 获取性能统计
        stats = self.emotion_classifier.get_performance_stats()
        
        # 更新性能信息标签
        performance_text = (
            f"Inference Time: {stats['avg_inference_time']:.1f}ms\n"
            f"FPS: {stats['fps']:.1f}\n"
            f"Device: {stats['device']}"
        )
        self.performance_label.setText(performance_text)

        # 绘制检测框和表情标签
        for bbox, (emotion, score) in zip(faces, emotions):
            x, y, w, h = bbox
            cv2.rectangle(frame_rgb, bbox, (0, 255, 0), 1)
            
            # 显示表情标签
            label = f"{emotion}: {score:.2f}"
            cv2.putText(frame_rgb, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 更新显示
        height, width, channel = frame_rgb.shape
        bytes_per_line = channel * width
        qt_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.display_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

        # 更新状态显示
        self.status_label.setText("已检测到人脸" if face_detected else "未检测到人脸")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: %s;
                background-color: rgba(0, 0, 0, 0.5);
                padding: 10px;
                border-radius: 5px;
            }
        """ % ("green" if face_detected else "red"))

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
