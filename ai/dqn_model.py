# ============================================================
#  dqn_model.py  —  深度 Q 网络定义
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    4 层全连接 DQN：
    输入层 → 256 → 128 → 64 → 输出层（动作数）
    使用 ReLU 激活 + Dropout 防过拟合
    """

    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x