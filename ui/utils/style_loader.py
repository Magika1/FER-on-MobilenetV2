from pathlib import Path

def load_stylesheet(name):
    """加载指定的样式表文件"""
    style_path = Path(__file__).parent.parent / 'assets' / 'styles' / f'{name}.qss'
    if not style_path.exists():
        return ""
    
    with open(style_path, 'r', encoding='utf-8') as f:
        return f.read()