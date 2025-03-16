import sys
import cv2
from collections import deque
from PyQt6.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget,
                            QMessageBox, QGridLayout, QSizePolicy, QMenuBar, QMenu)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from modules import FaceDetector, EmotionClassifier
from utils.style_loader import load_stylesheet
from assets import (
    EMOTION_MAPPING,
    EMOTION_NAMES,
    EMOJI_MAPPING,
    CAMERA_CHECK_RANGE,
    WINDOW_TITLE,
    WINDOW_GEOMETRY,
    WINDOW_MIN_HEIGHT
)


class UIapp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_camera = 0
        self.cap = None
        self.emotion_times = deque(maxlen=30)  # 保存最近30秒的记录
        self.emotion_values = deque(maxlen=30)
        self.emotion_labels = deque(maxlen=30)

        self.setStyleSheet(load_stylesheet('dark'))

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
        self.setGeometry(100, 100, 1000, 550)
        self.setMinimumHeight(300)
        
        # 创建主布局
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self._create_menubar()
        self._create_content_area()

    def _create_content_area(self):
        content_widget = QWidget()
        content_widget.setMinimumHeight(250)
        self.content_layout = QGridLayout(content_widget)
        self.content_layout.setSpacing(10)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        
        self._create_left_panel()
        self._create_right_panel()
        
        self.content_layout.setColumnStretch(0, 3)
        self.content_layout.setColumnStretch(3, 1)
        
        self.main_layout.addWidget(content_widget)
        return content_widget

    def _create_menubar(self):
        menubar = QMenuBar(self)
        menubar.setFixedHeight(30)
        menubar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
        self.main_layout.addWidget(menubar, 0)
        return menubar

    def _create_left_panel(self):
        left_panel = QWidget()
        self.left_layout = QVBoxLayout(left_panel)
        self.left_layout.setSpacing(10)
        self.left_layout.setContentsMargins(0, 0, 0, 0)

        self.display_label = self._create_display_label()
        self.content_layout.addWidget(left_panel, 0, 0, 1, 3)
        return left_panel

    def _create_display_label(self):
        label = QLabel()
        label.setObjectName("display_label")  # 设置对象名，用于样式表匹配
        label.setMinimumSize(800, 450)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_layout.addWidget(label)
        return label

    def _create_right_panel(self):
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        self.right_layout.setSpacing(20)
        self.right_layout.setContentsMargins(10, 10, 10, 10)

        self.emoji_label = self._create_emoji_label()
        self.status_label = self._create_status_label()
        self.performance_label = self._create_performance_label()

        self.right_layout.addStretch()
        
        self.content_layout.addWidget(right_panel, 0, 3, 1, 1)
        return right_panel

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
        self.status_label.setProperty("detected", face_detected)  # 使用属性来控制颜色
        self.status_label.style().unpolish(self.status_label)  # 刷新样式
        self.status_label.style().polish(self.status_label)

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