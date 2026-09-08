"""本项目的工具脚本与 MMDetection 扩展包。

保留这个 ``__init__.py`` 后，工程根目录加入 ``sys.path`` 时即可用
``tools.xxx`` 导入自定义指标；配置文件中的 ``custom_imports`` 正是通过
这种方式注册 ``TargetPrecisionMetric``。
"""
