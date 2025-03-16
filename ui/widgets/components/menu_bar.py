from PyQt6.QtWidgets import QMenuBar, QMenu, QMessageBox
from PyQt6.QtCore import pyqtSignal
import cv2
from assets import CAMERA_CHECK_RANGE

class MenuBar(QMenuBar):
    camera_changed = pyqtSignal(int)  # 摄像头切换信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setup_ui()
        
    def setup_ui(self):
        camera_menu = QMenu('摄像头', self)
        self.addMenu(camera_menu)
        self.refresh_camera_list(camera_menu)
        
    def refresh_camera_list(self, menu):
        menu.clear()
        for i in range(CAMERA_CHECK_RANGE):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                action = menu.addAction(f"摄像头 {i}")
                action.setData(i)
                action.triggered.connect(
                    lambda checked, x=i: self.camera_changed.emit(x)
                )