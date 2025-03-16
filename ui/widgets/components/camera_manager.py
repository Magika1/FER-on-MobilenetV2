import cv2
from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np

class CameraManager(QObject):
    frame_ready = pyqtSignal(np.ndarray)  # 帧准备好信号
    camera_error = pyqtSignal(str)  # 摄像头错误信号

    def __init__(self):
        super().__init__()
        self.cap = None
        self.current_camera = 0
        
    def start_camera(self, camera_id=None):
        if camera_id is not None:
            self.current_camera = camera_id
            
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.current_camera)
        if not self.cap.isOpened():
            self.camera_error.emit(f"无法打开摄像头 {self.current_camera}")
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return True
        
    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        return frame
        
    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None