from PyQt6.QtCore import QObject, pyqtSignal
import cv2
import numpy as np
from utils.error_handler import ErrorHandler, ErrorType
from ui.assets import CAMERA_CHECK_RANGE  # 添加这行导入

class CameraError(Exception):
    """摄像头异常基类"""
    pass

class CameraNotFoundError(CameraError):
    """摄像头未找到异常"""
    pass

class CameraInitError(CameraError):
    """摄像头初始化异常"""
    pass

class CameraManager(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    camera_error = pyqtSignal(str)
    cameras_updated = pyqtSignal(list)  # 添加摄像头列表更新信号

    def __init__(self):
        super().__init__()
        self.cap = None
        self.current_camera = 0
        self.error_handler = ErrorHandler()
        self.available_cameras = []
        
    def scan_cameras(self):
        """扫描可用摄像头"""
        available_cameras = []
        try:
            for i in range(CAMERA_CHECK_RANGE):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cap.release()
                    available_cameras.append(i)
                    
            self.available_cameras = available_cameras
            self.cameras_updated.emit(available_cameras)
            return available_cameras
            
        except Exception as e:
            self.error_handler.show_error(
                ErrorType.CAMERA,
                f"扫描摄像头时发生错误: {str(e)}"
            )
            return []

    def start_camera(self, camera_id=None):
        """启动摄像头"""
        try:
            if camera_id is not None:
                self.current_camera = camera_id
                
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.current_camera)
            if not self.cap.isOpened():
                raise CameraNotFoundError(f"无法打开摄像头 {self.current_camera}")
                
            # 设置摄像头参数
            if not self._setup_camera_params():
                raise CameraInitError("摄像头参数设置失败")
                
            return True
            
        except CameraNotFoundError as e:
            self.error_handler.show_error(ErrorType.CAMERA, str(e))
            self.camera_error.emit(str(e))
            return False
        except CameraInitError as e:
            self.error_handler.show_error(ErrorType.CAMERA, str(e))
            self.camera_error.emit(str(e))
            return False
        except Exception as e:
            self.error_handler.show_error(ErrorType.SYSTEM, f"摄像头发生未知错误: {str(e)}")
            self.camera_error.emit(str(e))
            return False
            
    def _setup_camera_params(self):
        """设置摄像头参数"""
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return True
        except Exception:
            return False
            
    def get_frame(self):
        """获取摄像头帧"""
        try:
            if self.cap is None or not self.cap.isOpened():
                return None
                
            ret, frame = self.cap.read()
            if not ret:
                self.error_handler.show_warning(
                    ErrorType.CAMERA, 
                    "读取摄像头帧失败，请检查摄像头连接"
                )
                return None
                
            return frame
            
        except Exception as e:
            self.error_handler.show_error(
                ErrorType.CAMERA, 
                f"获取摄像头帧时发生错误: {str(e)}"
            )
            return None
        
    def release(self):
        """释放摄像头资源"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None