# ============================================================
#  trainer.py  —  批量训练入口（无窗口快速训练）
# ============================================================
import random
import sys
import os
import numpy as np
from config import *
from entities import BigFish, SmallFish, Seaweed
from ai import DQNAgent


class HeadlessTrainer:
    """
    无界面快速训练器，用于在没有 Pygame 窗口的情况下
    大批量训练大鱼和小鱼的 DQN 模型。
    """

    def __init__(self):
        self.big_agent   = DQNAgent(STATE_SIZE, ACTION_SIZE,
                                    BIG_FISH_MODEL_PATH)
        self.small_agent = DQNAgent(STATE_SIZE, ACTION_SIZE,
                                    SMALL_FISH_MODEL_PATH)

    def run(self, episodes: int = TRAIN_EPISODES):
        print(f"开始无界面训练，共 {episodes} 回合...")

        for ep in range(episodes):
            big_fishes   = [BigFish()   for _ in range(BIG_FISH_INIT_COUNT)]
            small_fishes = [SmallFish() for _ in range(SMALL_FISH_INIT_COUNT)]
            seaweeds     = [Seaweed()   for _ in range(SEAWEED_INIT_COUNT)]

            total_reward_big   = 0.0
            total_reward_small = 0.0
            ep_loss_big        = 0.0
            ep_loss_small      = 0.0
            step_count         = 0

            # 每回合最多跑 2000 帧
            for frame in range(1, 2001):
                # 海草刷新
                alive_sw = sum(1 for s in seaweeds if s.alive)
                if frame % SEAWEED_SPAWN_INTERVAL == 0 and alive_sw < SEAWEED_MAX_COUNT:
                    seaweeds.append(Seaweed())
                seaweeds = [s for s in seaweeds if s.alive]

                # 大鱼步
                new_big = []
                for fish in big_fishes:
                    if not fish.alive:
                        continue
                    state  = fish.get_state(small_fishes, big_fishes)
                    action = self.big_agent.choose_action(state, training=True)
                    dx, dy = ACTIONS[action]
                    fish.move(dx, dy)

                    eat_r = fish.try_eat(small_fishes)
                    fish.update(small_fishes, big_fishes)

                    reward = self._big_reward(fish, eat_r, small_fishes)
                    done   = not fish.alive
                    ns     = fish.get_state(small_fishes, big_fishes)

                    self.big_agent.remember(state, action, reward, ns, done)
                    loss = self.big_agent.train_step()
                    ep_loss_big    += loss
                    total_reward_big += reward

                    new_big.extend(fish.try_breed(big_fishes))

                big_fishes.extend(new_big)

                # 小鱼步
                new_small = []
                for fish in small_fishes:
                    if not fish.alive:
                        continue
                    state  = fish.get_state(big_fishes, seaweeds)
                    action = self.small_agent.choose_action(state, training=True)
                    dx, dy = ACTIONS[action]
                    fish.move(dx, dy)

                    eat_r = fish.try_eat(seaweeds)
                    fish.update(big_fishes, seaweeds)

                    reward = self._small_reward(fish, eat_r, big_fishes)
                    done   = not fish.alive
                    ns     = fish.get_state(big_fishes, seaweeds)

                    self.small_agent.remember(state, action, reward, ns, done)
                    loss = self.small_agent.train_step()
                    ep_loss_small    += loss
                    total_reward_small += reward

                    new_small.extend(fish.try_breed(small_fishes))

                small_fishes.extend(new_small)
                step_count += 1

                # 清理死亡实体
                big_fishes   = [f for f in big_fishes   if f.alive]
                small_fishes = [f for f in small_fishes if f.alive]

                # 防止种群灭绝
                if not big_fishes:
                    big_fishes = [BigFish() for _ in range(3)]
                if not small_fishes:
                    small_fishes = [SmallFish() for _ in range(6)]

            # ε 衰减
            self.big_agent.decay_epsilon()
            self.small_agent.decay_epsilon()

            # 目标网络更新
            if ep % TARGET_UPDATE == 0:
                self.big_agent.update_target_net()
                self.small_agent.update_target_net()

            # 打印进度
            if ep % 10 == 0 or ep == episodes - 1:
                avg_r_big   = total_reward_big   / max(step_count, 1)
                avg_r_small = total_reward_small / max(step_count, 1)
                avg_l_big   = ep_loss_big        / max(step_count, 1)
                print(
                    f"[Episode {ep:4d}/{episodes}] "
                    f"大鱼奖励: {avg_r_big:+6.2f} | "
                    f"小鱼奖励: {avg_r_small:+6.2f} | "
                    f"大鱼Loss: {avg_l_big:.4f} | "
                    f"ε: {self.big_agent.epsilon:.3f}"
                )

            # 定期保存
            if ep % 50 == 0 and ep > 0:
                self.big_agent.save()
                self.small_agent.save()
                print(f"  → 模型已保存 (Episode {ep})")

        # 最终保存
        self.big_agent.save()
        self.small_agent.save()
        print("训练完成！模型已保存。")

    # ------------------------------------------------------------------ #
    def _big_reward(self, fish, eat_r, small_fishes):
        reward = eat_r
        if not fish.alive:
            return reward - 50.0
        reward += 0.1
        hunger_ratio = fish.hunger / fish.hunger_max
        if hunger_ratio > 0.5:
            reward -= hunger_ratio * 0.8
        margin = 40
        if (fish.x < margin or fish.x > SCREEN_WIDTH - margin
                or fish.y < margin or fish.y > SCREEN_HEIGHT - margin):
            reward -= 3.0
        return reward

    def _small_reward(self, fish, eat_r, big_fishes):
        reward = eat_r
        if not fish.alive:
            return reward - 30.0
        reward += 0.05
        hunger_ratio = fish.hunger / fish.hunger_max
        if hunger_ratio > 0.5:
            reward -= hunger_ratio * 0.4
        for bf in big_fishes:
            if bf.alive:
                d = fish.dist(bf)
                if d < 60:
                    reward -= (60 - d) * 0.05
                elif d < 120:
                    reward += 1.0
        return reward


# ---- 命令行入口 ----
if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else TRAIN_EPISODES
    HeadlessTrainer().run(episodes)