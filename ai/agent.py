# ============================================================
#  agent.py  —  DQN 智能体（Q-Learning 核心）
# ============================================================
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from config import *
from .dqn_model import DQN


class DQNAgent:
    """
    基于经验回放 + 目标网络的 DQN 智能体。
    同时用于控制大鱼和小鱼（通过不同实例）。
    """

    def __init__(self, state_size: int, action_size: int,
                 model_path: str = None):
        self.state_size  = state_size
        self.action_size = action_size
        self.model_path  = model_path
        self.device      = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 主网络 & 目标网络
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        # 经验回放缓冲区
        self.memory  = deque(maxlen=MEMORY_SIZE)

        # ε-贪心参数
        self.epsilon = EPSILON_START
        self.steps   = 0
        self.episode = 0

        # 如果有已保存模型，加载它
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            print(f"[Agent] 已加载模型: {model_path}")

    # ------------------------------------------------------------------ #
    #  动作选择（ε-贪心策略）
    # ------------------------------------------------------------------ #
    def choose_action(self, state: list, training: bool = True) -> int:
        """
        ε-贪心：
          - 训练模式：以 ε 概率随机探索，否则贪心利用
          - 推理模式：始终贪心
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------ #
    #  存储经验
    # ------------------------------------------------------------------ #
    def remember(self, state, action, reward, next_state, done):
        """将一条经验 (s, a, r, s', done) 存入回放缓冲区"""
        self.memory.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    # ------------------------------------------------------------------ #
    #  训练一步
    # ------------------------------------------------------------------ #
    def train_step(self) -> float:
        """
        从经验池随机采样一个批次，执行 DQN 更新：
          Q_target = r + γ · max_a' Q_target(s', a')  (非终止)
          Q_target = r                                  (终止)
        返回当次 loss 值。
        """
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.BoolTensor(dones).to(self.device)

        # 当前 Q 值
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # 目标 Q 值（使用目标网络）
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q   = rewards_t + GAMMA * max_next_q * (~dones_t)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        return loss.item()

    # ------------------------------------------------------------------ #
    #  更新目标网络
    # ------------------------------------------------------------------ #
    def update_target_net(self):
        """将 policy_net 的权重复制到 target_net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------ #
    #  ε 衰减
    # ------------------------------------------------------------------ #
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.episode += 1

    # ------------------------------------------------------------------ #
    #  模型保存 / 加载
    # ------------------------------------------------------------------ #
    def save(self, path: str = None):
        path = path or self.model_path
        if path:
            torch.save({
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer" : self.optimizer.state_dict(),
                "epsilon"   : self.epsilon,
                "steps"     : self.steps,
                "episode"   : self.episode,
            }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", EPSILON_END)
        self.steps   = ckpt.get("steps",   0)
        self.episode = ckpt.get("episode", 0)