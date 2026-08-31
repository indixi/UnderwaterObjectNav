# Windows PowerShell 训练脚本。
#
# 这个脚本是给 Windows 环境准备的；Ubuntu 上请使用 scripts/train.sh。
# 它只是把常用参数整理成一个命令，真正训练逻辑仍然在 tools/train.py。

param(
    # DUO 数据集根目录。需要包含 annotations/、images/train/、images/test/。
    [string]$DataRoot = "..\数据\DUO\DUO",

    # 模型权重、日志、resolved_config.py 会保存到这里。
    [string]$WorkDir = "work_dirs\gfl_r50_fpn_duo"
)

# --seed 42 固定随机种子；--amp 开启混合精度；--resume auto 自动续训。
python tools/train.py `
    --config configs/gfl_r50_fpn_duo_base.py `
    --data-root $DataRoot `
    --work-dir $WorkDir `
    --seed 42 `
    --amp `
    --resume auto
