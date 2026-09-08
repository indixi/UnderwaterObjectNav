"""旧数据检查命令的兼容入口。

原工程文档使用 ``python tools/inspect_duo_dataset.py``。为了不让历史命令
立即失效，本文件仅转发到新版 ``inspect_dataset.main``；新代码应直接使用
更符合当前数据集名称的 ``tools/inspect_dataset.py``。
"""

# 脚本运行时 tools 目录位于 sys.path，因此可以直接导入同目录模块。
from inspect_dataset import main


if __name__ == '__main__':
    main()
