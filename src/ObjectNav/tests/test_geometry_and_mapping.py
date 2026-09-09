import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from objectnav_bc.dataset.episode_io import episode_directories
from objectnav_bc.mapping.global_semantic_map import GridMap, WorldMapSpec
from objectnav_bc.mapping.local_egocentric_map import (
    LocalEvidenceMap,
    LocalMapSpec,
)
from objectnav_bc.perception.depth_projection import (
    CameraIntrinsics,
    pixels_depth_to_world,
    transform_from_translation_rpy,
)
from objectnav_bc.perception.detection_depth_fusion import (
    DetectionDepthFusion,
)
from objectnav_bc.perception.semantic_detector import Detection


class ProjectionTests(unittest.TestCase):
    def test_stonefish_camera_center_projects_along_base_x(self):
        intrinsics = CameraIntrinsics(100.0, 100.0, 1.5, 1.5, 4, 4)
        camera = transform_from_translation_rpy(
            [0.20, 0.0, -0.02], [math.pi / 2, 0.0, math.pi / 2]
        )
        base = np.eye(4, dtype=np.float32)
        base[:3, 3] = [1.0, 2.0, 3.3]
        point = pixels_depth_to_world(
            [[1.5, 1.5]], [2.0], intrinsics, camera, base
        )[0]
        np.testing.assert_allclose(point, [3.2, 2.0, 3.28], atol=1e-5)

    def test_target_is_zero_when_echinus_depth_is_invalid(self):
        intrinsics = CameraIntrinsics(100.0, 100.0, 1.5, 1.5, 4, 4)
        depth_config = {
            "min_depth_m": 0.2,
            "max_depth_m": 10.0,
            "echinus_center_roi_ratio": 0.4,
            "echinus_min_roi_size_px": 2,
            "echinus_min_valid_depth_pixels": 3,
            "rock_center_roi_ratio": 0.4,
            "rock_depth_stride": 1,
            "rock_depth_gate_m": 0.2,
            "rock_min_valid_depth_pixels": 3,
        }
        result = DetectionDepthFusion(intrinsics, depth_config).process(
            np.full((4, 4), np.nan, dtype=np.float32),
            [Detection((0, 0, 4, 4), 0.9, "echinus")],
            np.eye(4, dtype=np.float32),
            np.eye(4, dtype=np.float32),
        )
        self.assertEqual(result.target.visible, 0.0)
        self.assertEqual(result.target.bearing_rad, 0.0)
        self.assertEqual(result.target.distance_m, 0.0)
        self.assertEqual(result.echinus_points.shape, (0, 3))


class MappingTests(unittest.TestCase):
    def test_confidence_uses_episode_max_and_reset_clears_it(self):
        grid = GridMap(
            ("echinus",), WorldMapSpec(-1.0, -1.0, 2.0, 2.0, 0.1)
        )
        grid.mark_points("echinus", [[0.0, 0.0]], [0.8])
        grid.mark_points("echinus", [[0.0, 0.0]], [0.3])
        gx, gy, valid = grid.grid_indices([[0.0, 0.0]])
        self.assertTrue(valid[0])
        self.assertAlmostEqual(grid.map[0, gy[0], gx[0]], 0.8)
        grid.reset()
        self.assertEqual(float(grid.map.max()), 0.0)

    def test_turn_left_moves_fixed_world_point_clockwise_in_local_map(self):
        world = WorldMapSpec(-3.0, -3.0, 6.0, 6.0, 0.05)
        local = LocalMapSpec(3.0, 3.0, 0.05)
        evidence = LocalEvidenceMap(world, 0.05)
        evidence.mark_disks("obstacle", [[1.0, 0.0]], 0.10)

        before = np.argwhere(evidence.render((0.0, 0.0), 0.0, local)[0] > 0)
        after = np.argwhere(
            evidence.render((0.0, 0.0), -math.pi / 2, local)[0] > 0
        )
        center_row = local.height_cells / 2
        center_col = local.width_cells / 2
        self.assertLess(before[:, 0].mean(), center_row)
        self.assertAlmostEqual(before[:, 1].mean(), center_col, delta=2.0)
        self.assertAlmostEqual(after[:, 0].mean(), center_row, delta=2.0)
        self.assertGreater(after[:, 1].mean(), center_col)


class EpisodeSelectionTests(unittest.TestCase):
    def test_failed_and_episode_false_directories_are_always_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "episode_0001": True,
                "episode_0002": False,
                "episode_false_0003": True,
            }
            for name, success in cases.items():
                episode = root / name
                episode.mkdir()
                (episode / "episode.yaml").write_text(
                    f"success: {str(success).lower()}\n", encoding="utf-8"
                )
            selected = episode_directories(root, include_failed=False)
            self.assertEqual([path.name for path in selected], ["episode_0001"])
            selected_with_failed = episode_directories(root, include_failed=True)
            self.assertEqual(
                [path.name for path in selected_with_failed],
                ["episode_0001", "episode_0002"],
            )


if __name__ == "__main__":
    unittest.main()
