from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, 
    QLabel, QSizePolicy, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2
from ui.assets import EMOTION_MAPPING, EMOTION_NAMES, EMOJI_MAPPING
from PyQt6.QtWidgets import QFileDialog

class ContentArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        # 绑定按钮功能
        self.btn_save.clicked.connect(self.save_stats)  # 修改连接到新的保存统计方法
        self.btn_reset.clicked.connect(self.reset_status)
        self.btn_exit.clicked.connect(self.exit_app)
        self._last_frame = None  # 用于保存当前帧
        
    def setup_ui(self):
        """设置UI布局"""
        self.setMinimumHeight(250)
        self.layout = QGridLayout(self)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        self._create_left_panel()
        self._create_right_panel()
        
        self.layout.setColumnStretch(0, 3)
        self.layout.setColumnStretch(3, 1)
        
    def _create_left_panel(self):
        """创建左侧面板"""
        left_panel = QWidget()
        self.left_layout = QVBoxLayout(left_panel)
        self.left_layout.setSpacing(10)
        self.left_layout.setContentsMargins(0, 0, 0, 0)

        self.display_label = self._create_display_label()
        self.layout.addWidget(left_panel, 0, 0, 1, 3)
        
    def _create_right_panel(self):
        """创建右侧面板"""
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        self.right_layout.setSpacing(20)
        self.right_layout.setContentsMargins(10, 10, 10, 10)

        self.emoji_label = self._create_emoji_label()
        self.status_label = self._create_status_label()
        self.performance_label = self._create_performance_label()

        self.right_layout.addStretch()
        # 新增：按钮区域
        self.button_bar = self._create_button_bar()
        self.right_layout.addWidget(self.button_bar)
        self.layout.addWidget(right_panel, 0, 3, 1, 1)
        
    def _create_display_label(self):
        """创建显示标签"""
        label = QLabel()
        label.setObjectName("display_label")
        label.setMinimumSize(800, 450)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_layout.addWidget(label)
        return label
        
    def _create_emoji_label(self):
        """创建表情标签"""
        label = QLabel()
        label.setObjectName("emoji_label")
        label.setFixedHeight(60)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_layout.addWidget(label)
        return label
        
    def _create_status_label(self):
        """创建状态标签"""
        label = QLabel("未检测到人脸")
        label.setObjectName("status_label")
        label.setFixedHeight(50)
        self.right_layout.addWidget(label)
        return label
        
    def _create_performance_label(self):
        """创建性能信息标签"""
        label = QLabel()
        label.setObjectName("performance_label")
        label.setFixedHeight(80)
        self.right_layout.addWidget(label)
        return label
        
    def _create_button_bar(self):
        """创建按钮区域"""
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        # 保存统计结果按钮
        self.btn_save = QPushButton("保存统计结果")
        self.btn_save.setObjectName("btn_save")
        layout.addWidget(self.btn_save)
        # 重置统计按钮
        self.btn_reset = QPushButton("重置统计")
        self.btn_reset.setObjectName("btn_reset")
        layout.addWidget(self.btn_reset)
        # 退出按钮
        self.btn_exit = QPushButton("退出")
        self.btn_exit.setObjectName("btn_exit")
        layout.addWidget(self.btn_exit)
        layout.addStretch()
        return bar
        
    def save_stats(self):
        """保存情绪变化历史数据为CSV文件"""
        if not hasattr(self, '_emotion_history') or not self._emotion_history:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存情绪历史", "emotion_history.csv", "CSV Files (*.csv)"
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("时间(秒),情绪类型,置信度\n")
                for timestamp, emotion, confidence in self._emotion_history:
                    f.write(f"{timestamp:.2f},{emotion},{confidence:.3f}\n")
                    
    def reset_status(self):
        """重置统计数据"""
        if hasattr(self, 'window') and hasattr(self.window(), 'emotion_history'):
            self.window().emotion_history = []
            self.window().total_frames = 0
        self.emoji_label.setText("❓")
        self.status_label.setText("未检测到人脸")
        self.status_label.setProperty("detected", False)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.performance_label.setText("")
        self.display_label.clear()
                    
    def update_ui(self, data):
        """更新UI显示"""
        frame = data['frame']
        faces = data['faces']
        face_detected = data['face_detected']
        emotions = data['emotions']
        stats = data['stats']
        if 'emotion_stats' in data:  # 保存统计数据
            self._last_stats = data['emotion_stats']
        self._emotion_history = data['emotion_history']
        self._last_frame = frame.copy()
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

    def exit_app(self):
        """关闭主窗口"""
        self.window().close()