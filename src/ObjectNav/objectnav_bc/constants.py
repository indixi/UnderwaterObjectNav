"""Stable action, goal, and map-channel contracts."""

# This order is the model-logit contract and must match expert labels.
ACTION_NAMES = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
ACTION_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}

# Goal IDs are independent of detector labels and semantic-map channels.
GOAL_NAMES = ("echinus",)
GOAL_TO_ID = {name: index for index, name in enumerate(GOAL_NAMES)}

GLOBAL_MAP_CHANNELS = (
    "obstacle",
    "explored",
    "visited",
    "robot",
    "echinus",
    "rock",
)
LOCAL_MAP_CHANNELS = ("obstacle", "rock", "echinus", "visited")
NUM_GLOBAL_MAP_CHANNELS = len(GLOBAL_MAP_CHANNELS)
NUM_LOCAL_MAP_CHANNELS = len(LOCAL_MAP_CHANNELS)

# Compatibility aliases for older callers that only inspect constants.
SEMANTIC_NAMES = GLOBAL_MAP_CHANNELS
NUM_MAP_CHANNELS = NUM_GLOBAL_MAP_CHANNELS
