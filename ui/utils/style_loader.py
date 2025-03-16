from pathlib import Path
from utils.error_handler import ErrorHandler, ErrorType

def load_stylesheet(name, error_handler=None):
    """加载指定的样式表文件"""
    if error_handler is None:
        error_handler = ErrorHandler()
        
    style_path = Path(__file__).parent.parent / 'assets' / 'styles' / f'{name}.qss'
    
    try:
        if not style_path.exists():
            raise FileNotFoundError(f"样式表文件不存在: {style_path}")
        
        with open(style_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    except FileNotFoundError as e:
        error_handler.show_error(ErrorType.FILE_NOT_FOUND, str(e))
        return ""
    except PermissionError as e:
        error_handler.show_error(ErrorType.FILE_PERMISSION, f"无法访问样式表文件: {str(e)}")
        return ""
    except Exception as e:
        error_handler.show_error(ErrorType.FILE_IO, f"读取样式表文件时发生错误: {str(e)}")
        return ""