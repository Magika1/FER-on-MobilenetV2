from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, 
    QLabel, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2
from assets import EMOTION_MAPPING, EMOTION_NAMES, EMOJI_MAPPING

class ContentArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        self.setMinimumHeight(250)
        self.layout = QGridLayout(self)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        self._create_left_panel()
        self._create_right_panel()
        
        self.layout.setColumnStretch(0, 3)
        self.layout.setColumnStretch(3, 1)
        
    def _create_left_panel(self):
        left_panel = QWidget()
        self.left_layout = QVBoxLayout(left_panel)
        self.left_layout.setSpacing(10)
        self.left_layout.setContentsMargins(0, 0, 0, 0)

        self.display_label = self._create_display_label()
        self.layout.addWidget(left_panel, 0, 0, 1, 3)
        
    def _create_right_panel(self):
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        self.right_layout.setSpacing(20)
        self.right_layout.setContentsMargins(10, 10, 10, 10)

        self.emoji_label = self._create_emoji_label()
        self.status_label = self._create_status_label()
        self.performance_label = self._create_performance_label()

        self.right_layout.addStretch()
        self.layout.addWidget(right_panel, 0, 3, 1, 1)
        
    def _create_display_label(self):
        label = QLabel()
        label.setObjectName("display_label")
        label.setMinimumSize(800, 450)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_layout.addWidget(label)
        return label
        
    def _create_emoji_label(self):
        label = QLabel()
        label.setObjectName("emoji_label")
        label.setFixedHeight(60)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_layout.addWidget(label)
        return label
        
    def _create_status_label(self):
        label = QLabel("未检测到人脸")
        label.setObjectName("status_label")
        label.setFixedHeight(50)
        self.right_layout.addWidget(label)
        return label
        
    def _create_performance_label(self):
        label = QLabel()
        label.setObjectName("performance_label")
        label.setFixedHeight(80)
        self.right_layout.addWidget(label)
        return label
        
    def update_ui(self, data):
        """更新UI显示"""
        frame = data['frame']
        faces = data['faces']
        face_detected = data['face_detected']
        emotions = data['emotions']
        stats = data['stats']
        
        self._update_display(frame)
        self._update_status(face_detected, faces)
        self._update_emotions(faces, emotions)
        self._update_performance(stats)

    def _update_display(self, frame):
        """更新显示画面"""
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def _update_status(self, face_detected, faces):
        """更新状态显示"""
        face_count = len(faces) if faces is not None and face_detected else 0
        self.status_label.setText(f"已检测到 {face_count} 个人脸" if face_detected else "未检测到人脸")
        self.status_label.setProperty("detected", face_detected)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _update_emotions(self, faces, emotions):
        """更新表情显示"""
        if not faces or not emotions:
            self.emoji_label.setText("❓")
            return

        # 只显示第一个检测到的人脸的表情
        emotion, _ = emotions[0]
        emoji = EMOJI_MAPPING.get(emotion, '❓')
        self.emoji_label.setText(f"{EMOTION_NAMES[EMOTION_MAPPING[emotion]]} {emoji}")

    def _update_performance(self, stats):
        """更新性能信息显示"""
        performance_text = (
            f"Inference Time: {stats['avg_inference_time']:.1f}ms\n"
            f"FPS: {stats['fps']:.1f}\n"
            f"Device: {stats['device']}"
        )
        self.performance_label.setText(performance_text)