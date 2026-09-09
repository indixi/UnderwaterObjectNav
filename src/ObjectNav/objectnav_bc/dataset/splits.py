"""Episode-level train/validation/test splitting without ML dependencies."""

import numpy as np


def split_by_episode(records, seed=42, ratios=(0.70, 0.15, 0.15)):
    episodes = sorted({str(record["episode_id"]) for record in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    count = len(episodes)
    train_count = int(count * ratios[0])
    val_count = int(count * ratios[1])
    groups = (
        set(episodes[:train_count]),
        set(episodes[train_count:train_count + val_count]),
        set(episodes[train_count + val_count:]),
    )
    return [
        [record for record in records if str(record["episode_id"]) in group]
        for group in groups
    ]
