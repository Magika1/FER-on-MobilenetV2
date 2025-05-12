from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtCore import pyqtSignal

class MenuBar(QMenuBar):
    camera_changed = pyqtSignal(int)
    style_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        # 摄像头菜单
        self.camera_menu = QMenu('摄像头', self)
        self.addMenu(self.camera_menu)
        
        # 样式菜单
        self.style_menu = QMenu('主题', self)
        dark_action = self.style_menu.addAction("深色主题")
        dark_action.triggered.connect(lambda: self.style_changed.emit('dark'))
        light_action = self.style_menu.addAction("浅色主题")
        light_action.triggered.connect(lambda: self.style_changed.emit('light'))
        self.addMenu(self.style_menu)

    def update_camera_list(self, cameras):
        """更新摄像头菜单列表"""
        self.camera_menu.clear()
        for camera_id in cameras:
            action = self.camera_menu.addAction(f"摄像头 {camera_id}")
            action.setData(camera_id)
            action.triggered.connect(
                lambda checked, x=camera_id: self.camera_changed.emit(x)
            )