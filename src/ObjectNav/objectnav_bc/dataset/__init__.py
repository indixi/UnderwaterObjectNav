"""Dataset tools, with PyTorch imports deferred until training is used."""

__all__ = ["BehaviorCloningDataset", "split_by_episode"]


def __getattr__(name):
    if name == "split_by_episode":
        from .splits import split_by_episode

        return split_by_episode
    if name == "BehaviorCloningDataset":
        from .bc_dataset import BehaviorCloningDataset

        return BehaviorCloningDataset
    raise AttributeError(name)
