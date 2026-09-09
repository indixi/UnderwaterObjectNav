# Frozen detector deployment artifacts

Copy these two files from the completed detector experiment into this folder:

- `best_echinus_recall_at_precision_95_epoch_5.pth`
- `echinus_threshold.json`

The inference-only model config is already included here. ObjectNav resolves
all three paths through `objectnav_bc/config/bc.yaml` and does not import the
`image_process_ResNet50` training project.

Class order is part of the checkpoint contract and must remain:

```text
0 = echinus
1 = rock
```

The echinus threshold comes from `recommended_threshold` in the JSON file.
Rock currently uses the `score_threshold: 0.50` fallback in `bc.yaml`.
