"""Efficiently rasterize camera-to-depth visibility rays in a 2-D map."""

import numpy as np


def explored_ray_points(sensor_xy, endpoints, pixels, resolution_m):
    """Return vectorized points along one farthest valid ray per image column."""
    sensor_xy = np.asarray(sensor_xy, dtype=np.float32).reshape(2)
    endpoints = np.asarray(endpoints, dtype=np.float32).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    if not len(endpoints):
        return np.empty((0, 2), dtype=np.float32)

    # Vertical image samples share almost the same horizontal bearing. Keep
    # one farthest endpoint per sampled column to bound work by image width.
    columns = pixels[:, 0].astype(np.int32)
    delta = endpoints[:, :2] - sensor_xy
    distances = np.linalg.norm(delta, axis=1)
    chosen = []
    for column in np.unique(columns):
        indices = np.flatnonzero(columns == column)
        chosen.append(indices[np.argmax(distances[indices])])
    chosen = np.asarray(chosen, dtype=np.int64)
    rays = endpoints[chosen, :2] - sensor_xy
    ray_lengths = np.linalg.norm(rays, axis=1)
    steps = np.maximum(1, np.ceil(ray_lengths / resolution_m).astype(np.int32))
    sample_index = np.arange(int(steps.max()) + 1, dtype=np.float32)
    mask = sample_index[None, :] <= steps[:, None]
    fractions = sample_index[None, :] / steps[:, None]
    points = sensor_xy[None, None, :] + fractions[:, :, None] * rays[:, None, :]
    return points[mask].astype(np.float32)

