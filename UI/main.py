import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget,
                            QMessageBox, QGridLayout, QComboBox, QSizePolicy, QMenuBar, QMenu)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from face_detector import FaceDetector
from emotion_classifier import EmotionClassifier
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from collections import deque

# 常量定义
CAMERA_CHECK_RANGE = 5
EMOTION_MAPPING = {
    'angry': 0, 'disgust': 1, 'fear': 2,
    'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6
}
EMOJI_MAPPING = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲',
    'neutral': '😐'
}
EMOTION_NAMES = ['生气', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '平静']
PLOT_BACKGROUND_COLOR = '#2b2b2b'
PLOT_LINE_COLOR = '#00ff00'
PLOT_GRID_COLOR = 'gray'
PLOT_GRID_ALPHA = 0.2
PLOT_GRID_STYLE = '--'


class UIapp(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_matplotlib()
        self.current_camera = 0
        self.cap = None
        self.emotion_times = deque(maxlen=30)  # 保存最近30秒的记录
        self.emotion_values = deque(maxlen=30)
        self.emotion_labels = deque(maxlen=30)
        self.initUI()
        self.setupCamera()
        self.face_detector = FaceDetector()
        self.emotion_classifier = EmotionClassifier()

    def _setup_matplotlib(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
        self.setGeometry(100, 100, 1000, 550)
        self.setMinimumHeight(300)
        
        # 创建主布局
        main_layout = QVBoxLayout()                # 创建垂直布局
        self.setLayout(main_layout)                # 设置为窗口的主布局
        main_layout.setSpacing(0)                  # 设置组件间距为0
        main_layout.setContentsMargins(0, 0, 0, 0) # 设置边距为0
        
        # 创建并添加菜单栏
        self._create_menubar(main_layout)
        
        # 创建并添加内容区域
        self._create_content_area(main_layout)

    def _create_content_area(self, parent_layout):
        content_widget = QWidget()
        content_widget.setMinimumHeight(250)
        content_layout = QGridLayout(content_widget)
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建并添加左右面板
        self._create_left_panel(content_layout)
        self._create_right_panel(content_layout)
        
        content_layout.setColumnStretch(0, 3)
        content_layout.setColumnStretch(3, 1)
        
        parent_layout.addWidget(content_widget)
        return content_widget

    def _create_menubar(self, parent_layout):
        menubar = QMenuBar(self)
        menubar.setFixedHeight(30)  # 固定菜单栏高度
        menubar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)  # 设置大小策略
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #2b2b2b;
                color: white;
                font-size: 14px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background: #404040;
            }
            QMenu {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #404040;
            }
            QMenu::item {
                padding: 8px 25px;
            }
            QMenu::item:selected {
                background-color: #404040;
            }
        """)
        
        camera_menu = QMenu('摄像头', self)
        menubar.addMenu(camera_menu)
        self.refresh_camera_list(camera_menu)
        parent_layout.addWidget(menubar, 0)
        return menubar

    def _create_left_panel(self, parent_layout):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._create_display_label(left_layout)
        parent_layout.addWidget(left_panel, 0, 0, 1, 3)
        return left_panel

    def _create_display_label(self, parent_layout):
        label = QLabel()
        label.setMinimumSize(800, 450)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("background-color: black;")
        parent_layout.addWidget(label)
        return label

    def _create_right_panel(self, parent_layout):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(10, 10, 10, 10)

        self._create_status_label(right_layout)
        self._create_emoji_label(right_layout)  
        self._create_performance_label(right_layout)

        right_layout.addStretch()
        
        parent_layout.addWidget(right_panel, 0, 3, 1, 1)
        return right_panel

    def _create_emoji_label(self, parent_layout):
        label = QLabel()
        label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                background-color: rgba(0, 0, 0, 0.5);
                padding: 10px;
                border-radius: 5px;
            }
        """)
        label.setFixedHeight(60)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        parent_layout.addWidget(label)
        return label

    def _create_status_label(self, parent_layout):
        label = QLabel("未检测到人脸")
        label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: red;
                background-color: rgba(0, 0, 0, 0.5);
                padding: 10px;
                border-radius: 5px;
            }
        """)
        label.setFixedHeight(50)
        parent_layout.addWidget(label)
        return label

    def _create_performance_label(self, parent_layout):
        label = QLabel()
        label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: white;
                background-color: rgba(0, 0, 0, 0.5);
                padding: 10px;
                border-radius: 5px;
            }
        """)
        label.setFixedHeight(80)
        parent_layout.addWidget(label)
        return label

    def refresh_camera_list(self, menu):
        menu.clear()
        for i in range(CAMERA_CHECK_RANGE):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                action = menu.addAction(f"摄像头 {i}")
                action.setData(i)
                action.triggered.connect(lambda checked, x=i: self.on_camera_changed(x))

    def on_camera_changed(self, camera_id):
        self.current_camera = camera_id
        self.setupCamera()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces, face_detected = self.face_detector.detect(frame)
        emotions = self.emotion_classifier.classify(frame, faces)
        stats = self.emotion_classifier.get_performance_stats()

        self._update_performance_label(stats)
        self._draw_faces_and_emotions(frame_rgb, faces, emotions)
        self._update_display(frame_rgb)
        self._update_status_label(face_detected, faces)

    def _update_performance_label(self, stats):
        performance_text = (
            f"Inference Time: {stats['avg_inference_time']:.1f}ms\n"
            f"FPS: {stats['fps']:.1f}\n"
            f"Device: {stats['device']}"
        )
        self.performance_label.setText(performance_text)

    def _draw_faces_and_emotions(self, frame_rgb, faces, emotions):
        if not faces or not emotions:
            self.emoji_label.setText("❓")
            return

        for bbox, (emotion, score) in zip(faces, emotions):
            x, y, w, h = bbox
            cv2.rectangle(frame_rgb, bbox, (0, 255, 0), 1)

            label = f"{emotion}: {score:.2f}"
            cv2.putText(frame_rgb, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 更新 emoji 显示（只显示第一个检测到的人脸的表情）
            emoji = EMOJI_MAPPING.get(emotion, '❓')
            self.emoji_label.setText(f"{EMOTION_NAMES[EMOTION_MAPPING[emotion]]} {emoji}")
            break

    def _update_display(self, frame_rgb):
        height, width, channel = frame_rgb.shape
        bytes_per_line = channel * width
        qt_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def _update_status_label(self, face_detected, faces):
        face_count = len(faces) if faces is not None and face_detected else 0
        self.status_label.setText(f"已检测到 {face_count} 个人脸" if face_detected else "未检测到人脸")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 24px;
                color: {'green' if face_detected else 'red'};
                background-color: rgba(0, 0, 0, 0.5);
                padding: 10px;
                border-radius: 5px;
            }}
        """)

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UIapp()
    window.show()
    sys.exit(app.exec())