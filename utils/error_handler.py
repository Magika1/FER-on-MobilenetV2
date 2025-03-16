from PyQt6.QtWidgets import QMessageBox
from enum import Enum, auto

class ErrorType(Enum):
    CAMERA = auto()
    UI = auto()
    MODEL = auto()
    SYSTEM = auto()
    MODEL_LOAD = auto()
    MODEL_INFERENCE = auto()
    FILE_NOT_FOUND = auto()    # 文件未找到
    FILE_PERMISSION = auto()   # 文件权限错误
    FILE_IO = auto()          # 文件读写错误

class ErrorHandler:
    @staticmethod
    def show_error(error_type: ErrorType, message: str, parent=None):
        title_map = {
            ErrorType.CAMERA: "摄像头错误",
            ErrorType.UI: "界面错误",
            ErrorType.MODEL: "模型错误",
            ErrorType.SYSTEM: "系统错误",
            ErrorType.MODEL_LOAD: "模型加载错误",
            ErrorType.MODEL_INFERENCE: "模型推理错误",
            ErrorType.FILE_NOT_FOUND: "文件未找到",
            ErrorType.FILE_PERMISSION: "文件权限错误",
            ErrorType.FILE_IO: "文件读写错误"
        }
        QMessageBox.critical(parent, title_map[error_type], message)

    @staticmethod
    def show_warning(error_type: ErrorType, message: str, parent=None):
        """显示警告对话框"""
        title_map = {
            ErrorType.CAMERA: "摄像头警告",
            ErrorType.UI: "界面警告",
            ErrorType.MODEL: "模型警告",
            ErrorType.SYSTEM: "系统警告"
        }
        QMessageBox.warning(parent, title_map[error_type], message)