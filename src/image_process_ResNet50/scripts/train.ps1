param(
    [string]$DataRoot = "data\underwater_objectnav_rgb",
    [string]$WorkDir = "work_dirs\gfl_r50_fpn_underwater_objectnav"
)

python tools/train.py `
    --config configs/gfl_r50_fpn_underwater_objectnav.py `
    --data-root $DataRoot `
    --work-dir $WorkDir `
    --seed 42 `
    --amp `
    --resume auto `
    --min-echinus-precision 0.95
