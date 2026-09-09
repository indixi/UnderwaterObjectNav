import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PolicyShapeTests(unittest.TestCase):
    def test_default_global_local_policy_forward(self):
        from objectnav_bc.config_loader import load_bc_config
        from objectnav_bc.models import build_policy_components

        runtime = load_bc_config(
            "objectnav_bc/config/bc.yaml", validate_detector=False
        )
        global_encoder, local_encoder, policy = build_policy_components(
            runtime.model, num_goals=1
        )
        global_feature = global_encoder(torch.zeros(2, 6, 80, 140))
        local_feature = local_encoder(torch.zeros(2, 4, 60, 60))
        logits = policy(
            global_feature,
            local_feature,
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, 3),
            torch.zeros(2),
        )
        self.assertEqual(tuple(global_feature.shape), (2, 256))
        self.assertEqual(tuple(local_feature.shape), (2, 128))
        self.assertEqual(tuple(logits.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
