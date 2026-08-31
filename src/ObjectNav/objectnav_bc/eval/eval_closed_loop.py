"""Small closed-loop policy adapter.

The ROS node can call ``predict`` after updating its persistent SemanticMapper.
Environment-specific ROS topics and low-level action execution intentionally
remain outside this package.
"""
import torch
from ..models import MapEncoder, BCPolicy
from ..constants import ACTION_NAMES, GOAL_TO_ID


class ClosedLoopPolicy:
    """加载 BC checkpoint，并为 ROS 闭环提供单步动作预测。"""

    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device); state = torch.load(checkpoint, map_location=self.device)    #选择运行的设备cpu还是gpu，并加载checkpoint文件
        self.encoder, self.policy = MapEncoder(), BCPolicy(num_goals=len(state.get("goal_names", ("sea_urchin",)))) #创建编码器和策略网络，num_goals是目标类别的数量，这里是1
        self.encoder.load_state_dict(state["encoder"]); self.policy.load_state_dict(state["policy"])#加载模型参数
        self.encoder.to(self.device).eval(); self.policy.to(self.device).eval() #将模型移动到指定设备，并设置为评估模式，关闭dropout和batchnorm等训练特性

    @torch.no_grad()
    def predict(self, semantic_map, goal_category="sea_urchin", yaw=0.0):
        """输入当前累计地图和状态，返回动作 ID、名称及四类概率。"""
        x = torch.as_tensor(semantic_map, dtype=torch.float32, device=self.device).unsqueeze(0) #将输入的累计语义地图转换为张量，并添加一个批次维度在最前面
        goal = torch.tensor([GOAL_TO_ID.get(goal_category, 0)], device=self.device)
        angle = torch.tensor([yaw], dtype=torch.float32, device=self.device)
        probs = self.policy(self.encoder(x), goal, angle).softmax(-1)[0]    #计算动作概率分布，先通过编码器提取特征，再通过策略网络得到动作 logits，最后对 logits 进行 softmax 得到概率分布，并取出第一个样本的概率，对最后一个维度进行softmax，得到四类动作的概率分布
        action = int(probs.argmax()); return {"action_id": action, "action": ACTION_NAMES[action], "probabilities": probs.cpu().tolist()}
