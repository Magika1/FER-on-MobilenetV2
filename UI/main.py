import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget,
                             QMessageBox, QGridLayout, QComboBox, QSizePolicy)
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
EMOTION_NAMES = ['生气', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '平静']
PLOT_BACKGROUND_COLOR = '#2b2b2b'
PLOT_LINE_COLOR = '#00ff00'
PLOT_GRID_COLOR = 'gray'
PLOT_GRID_ALPHA = 0.2
PLOT_GRID_STYLE = '--'


class CameraApp(QWidget):
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
        self.setGeometry(100, 100, 1280, 720)

        main_layout = QGridLayout()
        self.setLayout(main_layout)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()

        main_layout.addWidget(left_panel, 0, 0, 1, 3)
        main_layout.addWidget(right_panel, 0, 3, 1, 1)
        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(3, 1)

    def _create_left_panel(self):
        left_panel = QWidget()
        left_panel.setStyleSheet(f"background-color: {PLOT_BACKGROUND_COLOR};")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.display_label = self._create_display_label()
        plot_container = self._create_plot_container()

        left_layout.addWidget(self.display_label, stretch=7)
        left_layout.addWidget(plot_container, stretch=3)

        return left_panel

    def _create_display_label(self):
        label = QLabel()
        label.setMinimumSize(800, 450)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("background-color: black;")
        return label

    def _create_plot_container(self):
        plot_container = QWidget()
        plot_container.setMinimumHeight(250)
        plot_container.setMaximumHeight(250)
        plot_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = plt.figure(figsize=(10, 4))
        self.figure.patch.set_facecolor(PLOT_BACKGROUND_COLOR)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(PLOT_BACKGROUND_COLOR)

        # 设置坐标轴样式
        self.ax.tick_params(colors='white', labelsize=10)
        self.ax.set_xlabel('时间', color='white', fontsize=12)

        # 设置y轴刻度和标签
        self.ax.set_yticks(range(len(EMOTION_NAMES)))
        self.ax.set_yticklabels(EMOTION_NAMES)
        self.ax.set_ylim(-0.5, len(EMOTION_NAMES) - 0.5)

        # 设置网格
        self.ax.grid(True, color=PLOT_GRID_COLOR, alpha=PLOT_GRID_ALPHA, linestyle=PLOT_GRID_STYLE)

        # 设置边框颜色
        for spine in self.ax.spines.values():
            spine.set_color('white')

        # 调整布局
        self.figure.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(f"background-color: {PLOT_BACKGROUND_COLOR};")
        plot_layout.addWidget(self.canvas)

        return plot_container

    def _create_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(10, 10, 10, 10)

        self.camera_combo = self._create_camera_combo()
        self.status_label = self._create_status_label()
        self.performance_label = self._create_performance_label()

        self.refresh_camera_list()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)

        right_layout.addWidget(self.camera_combo)
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.performance_label)
        right_layout.addStretch()

        return right_panel

    def _create_camera_combo(self):
        combo = QComboBox()
        combo.setStyleSheet("""
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
        """)
        combo.setFixedHeight(30)
        return combo

    def _create_status_label(self):
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
        return label

    def _create_performance_label(self):
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
        return label

    def refresh_camera_list(self):
        self.camera_combo.clear()
        for i in range(CAMERA_CHECK_RANGE):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                self.camera_combo.addItem(f"摄像头 {i}", i)

    def on_camera_changed(self, index):
        if index >= 0:
            self.current_camera = self.camera_combo.currentData()
            self.setupCamera()

    def update_emotion_plot(self):
        if not hasattr(self, 'ax') or len(self.emotion_times) == 0:
            return

        self.ax.clear()

        self.ax.set_facecolor(PLOT_BACKGROUND_COLOR)
        self.ax.tick_params(colors='white', labelsize=10)
        self.ax.set_xlabel('时间', color='white', fontsize=12)

        times = list(self.emotion_times)
        formatted_times = [t.strftime('%M:%S') for t in times]
        emotion_values = [EMOTION_MAPPING[label] for label in self.emotion_labels]

        self.ax.set_yticks(range(len(EMOTION_NAMES)))
        self.ax.set_yticklabels(EMOTION_NAMES)
        self.ax.set_ylim(-0.5, len(EMOTION_NAMES) - 0.5)

        self.ax.plot(formatted_times, emotion_values, '-o', color=PLOT_LINE_COLOR, alpha=0.8, linewidth=2)

        for i, (t, v, s) in enumerate(zip(formatted_times, emotion_values, self.emotion_values)):
            if i % 2 == 0:
                self.ax.annotate(f'{s:.2f}', (t, v), color='white',
                                 xytext=(0, 5), textcoords='offset points',
                                 ha='center', fontsize=9)

        self.ax.tick_params(axis='x', rotation=45)
        self.ax.grid(True, color=PLOT_GRID_COLOR, alpha=PLOT_GRID_ALPHA, linestyle=PLOT_GRID_STYLE)
        self.figure.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)

        self.canvas.draw()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces, face_detected = self.face_detector.detect(frame)
        emotions = self.emotion_classifier.classify(frame, faces)
        stats = self.emotion_classifier.get_performance_stats()

        self._update_performance_label(stats)
        self._draw_faces_and_emotions(frame_rgb, faces, emotions)
        self._update_display(frame_rgb)
        self._update_status_label(face_detected)

    def _update_performance_label(self, stats):
        performance_text = (
            f"Inference Time: {stats['avg_inference_time']:.1f}ms\n"
            f"FPS: {stats['fps']:.1f}\n"
            f"Device: {stats['device']}"
        )
        self.performance_label.setText(performance_text)

    def _draw_faces_and_emotions(self, frame_rgb, faces, emotions):
        for bbox, (emotion, score) in zip(faces, emotions):
            x, y, w, h = bbox
            cv2.rectangle(frame_rgb, bbox, (0, 255, 0), 1)

            label = f"{emotion}: {score:.2f}"
            cv2.putText(frame_rgb, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            self._record_emotion(emotion, score)

    def _record_emotion(self, emotion, score):
        current_time = datetime.now()
        if not hasattr(self, 'last_record_time') or \
                (current_time - self.last_record_time).total_seconds() >= 1.0:
            self.last_record_time = current_time
            self.emotion_times.append(current_time)
            self.emotion_values.append(score)
            self.emotion_labels.append(emotion)
            self.update_emotion_plot()

    def _update_display(self, frame_rgb):
        height, width, channel = frame_rgb.shape
        bytes_per_line = channel * width
        qt_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def _update_status_label(self, face_detected):
        self.status_label.setText("已检测到人脸" if face_detected else "未检测到人脸")
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
    window = CameraApp()
    window.show()
    sys.exit(app.exec())