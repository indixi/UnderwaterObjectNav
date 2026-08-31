# 动作名称的顺序就是模型输出 logits 的顺序，不能随意调整；训练标签
# 会依据这个顺序编码成 0、1、2、3。
ACTION_NAMES = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
ACTION_TO_ID = {name: i for i, name in enumerate(ACTION_NAMES)}

# The first version of the collector uses one navigation goal. Keeping this
# table explicit makes adding goals later backward compatible.
# 目标类别单独维护，后续增加 algae、reef 等类别时只需要扩展这个表，
# Dataset 和 Embedding 会使用对应的整数 ID。
GOAL_NAMES = ("sea_urchin",)
GOAL_TO_ID = {name: i for i, name in enumerate(GOAL_NAMES)}

# 语义地图通道的索引定义。MapEncoder 的输入通道数必须与这里一致。
SEMANTIC_NAMES = ("obstacle", "explored", "robot", "visited", "sea_urchin", "rock", "sand")
NUM_MAP_CHANNELS = len(SEMANTIC_NAMES)
