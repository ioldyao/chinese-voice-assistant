"""
Pytest 配置文件

自动添加项目根目录到 Python 路径，以便测试可以导入 src 模块。
"""
import sys
from pathlib import Path

# 获取项目根目录（tests 目录的父目录）
project_root = Path(__file__).parent.parent

# 将项目根目录添加到 Python 路径
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
