"""旧配置文件名的兼容入口。

历史命令或下游模块仍可能引用本文件，因此暂不删除。它本身不再保存模型
参数，而是完整继承新的两分类 Underwater ObjectNav 配置。新实验应直接
使用 ``gfl_r50_fpn_underwater_objectnav.py``，以免名称继续产生误解。
"""

_base_ = ['./gfl_r50_fpn_underwater_objectnav.py']
