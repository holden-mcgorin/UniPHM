"""
path_helper.py
----------------
自动将项目根目录添加到 sys.path，确保跨目录导入 uniphm 模块时不会报错。
在脚本中仅需：
    import path_helper
即可自动生效。
"""

import os
import sys


def add_project_root():
    """将项目根目录添加到 sys.path（仅添加一次）"""
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print('注册')


# 执行自动注册
add_project_root()
