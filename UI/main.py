import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from face_detector import FaceDetector
from emotion_classifier import EmotionClassifier


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setupCamera()
        self.face_detector = FaceDetector()
        self.emotion_classifier = EmotionClassifier()  # 添加表情分类器

    def setupCamera(self):
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头！")
            sys.exit()
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def initUI(self):
        self.setWindowTitle("实时摄像头 - 人脸检测")
        self.setGeometry(100, 100, 1280, 720)

        # 主显示标签
        self.display_label = QLabel(self)
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 状态标签
        self.status_label = QLabel(self)
        self.status_label.setText("未检测到人脸")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: red;
                background-color: rgba(0, 0, 0, 0.5);
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.display_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

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
        
        # 显示性能信息，使用英文并优化显示格式
        performance_text = [
            f"Inference Time: {stats['avg_inference_time']:.1f}ms",
            f"FPS: {stats['fps']:.1f}",
            f"Device: {stats['device']}"
        ]
        
        # 在左上角显示性能信息，每行一个参数
        for i, text in enumerate(performance_text):
            y_pos = 30 + (i * 25)  # 每行间隔25像素
            # 添加半透明黑色背景
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(frame_rgb, (10, y_pos - 20), (10 + text_size[0], y_pos + 5), 
                         (0, 0, 0), -1)
            # 显示白色文字
            cv2.putText(frame_rgb, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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
