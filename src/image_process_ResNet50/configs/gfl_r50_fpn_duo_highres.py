"""旧高分辨率配置文件名的兼容入口。

仅用于保持已有命令可运行，所有实际参数均来自新的高分辨率配置文件。
"""

_base_ = ['./gfl_r50_fpn_underwater_objectnav_highres.py']
