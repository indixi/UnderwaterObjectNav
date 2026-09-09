"""Policy-only adapter retained for later online integration (no ROS code)."""

import numpy as np
import torch

from ..constants import ACTION_NAMES, GOAL_TO_ID
from ..models import build_policy_components


class ClosedLoopPolicy:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.state = torch.load(checkpoint, map_location=self.device)
        self.components = build_policy_components(
            self.state["model_config"], len(self.state["goal_names"]))
        for component, key in zip(
            self.components, ("global_encoder", "local_encoder", "policy")
        ):
            component.load_state_dict(self.state[key])
            component.to(self.device).eval()

    @torch.no_grad()
    def predict(
        self,
        global_map,
        local_map,
        target_visible,
        target_bearing_rad,
        target_distance_m,
        goal_category="echinus",
        yaw=0.0,
    ):
        target_config = self.state["target_config"]
        cue = np.zeros(3, dtype=np.float32)
        if target_visible:
            cue[:] = (
                1.0,
                np.clip(
                    target_bearing_rad / target_config["bearing_scale_rad"],
                    -1.0,
                    1.0,
                ),
                np.clip(
                    target_distance_m / target_config["distance_scale_m"],
                    0.0,
                    1.0,
                ),
            )
        global_tensor = torch.as_tensor(
            global_map, dtype=torch.float32, device=self.device).unsqueeze(0)
        local_tensor = torch.as_tensor(
            local_map, dtype=torch.float32, device=self.device).unsqueeze(0)
        target_tensor = torch.as_tensor(
            cue, dtype=torch.float32, device=self.device).unsqueeze(0)
        goal = torch.tensor(
            [GOAL_TO_ID[goal_category]], device=self.device)
        yaw_tensor = torch.tensor(
            [yaw], dtype=torch.float32, device=self.device)
        global_feature = self.components[0](global_tensor)
        local_feature = self.components[1](local_tensor)
        probabilities = self.components[2](
            global_feature,
            local_feature,
            goal,
            target_tensor,
            yaw_tensor,
        ).softmax(-1)[0]
        action = int(probabilities.argmax())
        return {
            "action_id": action,
            "action": ACTION_NAMES[action],
            "probabilities": probabilities.cpu().tolist(),
        }
