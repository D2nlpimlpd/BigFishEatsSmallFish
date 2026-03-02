# ============================================================
#  game_core.py  —  游戏核心逻辑
# ============================================================
import random
import math
import pygame
from config import *
from entities import BigFish, SmallFish, Seaweed
from ai import DQNAgent
from .ui import UI


class GameCore:
    """
    游戏核心：
      - 管理所有实体（大鱼、小鱼、海草）
      - 驱动大鱼/小鱼 DQN 智能体
      - 处理繁殖、进食、死亡
      - 海草定时刷新（更快更多）
      - 渲染所有对象及 UI
    """

    def __init__(self, screen: pygame.Surface, training: bool = True):
        self.screen    = screen
        self.training  = training
        self.paused    = False
        self.frame     = 0
        self.speed_mul = 1
        self.clock     = pygame.time.Clock()

        # ---- 实体 ----
        self.big_fishes   : list[BigFish]   = []
        self.small_fishes : list[SmallFish] = []
        self.seaweeds     : list[Seaweed]   = []

        # ---- AI 智能体 ----
        self.big_agent   = DQNAgent(STATE_SIZE, ACTION_SIZE,
                                    BIG_FISH_MODEL_PATH)
        self.small_agent = DQNAgent(STATE_SIZE, ACTION_SIZE,
                                    SMALL_FISH_MODEL_PATH)

        # ---- UI ----
        self.ui = UI(screen)

        # ---- 统计 ----
        self.total_reward_big   = 0.0
        self.total_reward_small = 0.0
        self.reward_count       = 0
        self.avg_reward         = 0.0

        self.reset()

    # ================================================================== #
    #  初始化 / 重置
    # ================================================================== #
    def reset(self):
        self.big_fishes   = [BigFish()   for _ in range(BIG_FISH_INIT_COUNT)]
        self.small_fishes = [SmallFish() for _ in range(SMALL_FISH_INIT_COUNT)]
        self.seaweeds     = [Seaweed()   for _ in range(SEAWEED_INIT_COUNT)]
        self.frame                = 0
        self.total_reward_big     = 0.0
        self.total_reward_small   = 0.0
        self.reward_count         = 0

    # ================================================================== #
    #  主循环
    # ================================================================== #
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                    if not running:
                        break

            if not self.paused:
                for _ in range(self.speed_mul):
                    self._step()

            self._render()
            self.clock.tick(FPS)

        pygame.quit()

    # ================================================================== #
    #  单步更新
    # ================================================================== #
    def _step(self):
        self.frame += 1

        # ---- 海草刷新（更快：每 SEAWEED_SPAWN_INTERVAL 帧刷新 2~3 棵）----
        alive_sw = sum(1 for s in self.seaweeds if s.alive)
        if self.frame % SEAWEED_SPAWN_INTERVAL == 0:
            batch = random.randint(2, 3)          # 每次刷新 2~3 棵
            for _ in range(batch):
                if alive_sw < SEAWEED_MAX_COUNT:
                    self.seaweeds.append(Seaweed())
                    alive_sw += 1

        # 移除死亡海草
        self.seaweeds = [s for s in self.seaweeds if s.alive]

        # 更新海草动画
        for sw in self.seaweeds:
            sw.update()

        # ================================================================ #
        #  大鱼 AI 步
        # ================================================================ #
        new_big = []
        for fish in self.big_fishes:
            if not fish.alive:
                continue

            state  = fish.get_state(self.small_fishes, self.big_fishes)
            action = self.big_agent.choose_action(state, self.training)
            fish.action = action

            dx, dy = ACTIONS[action]
            fish.move(dx, dy)

            eat_reward = fish.try_eat(self.small_fishes)
            fish.update(self.small_fishes, self.big_fishes)

            reward = self._calc_big_reward(fish, eat_reward)
            done   = not fish.alive

            next_state = fish.get_state(self.small_fishes, self.big_fishes)

            if self.training:
                self.big_agent.remember(state, action, reward,
                                        next_state, done)
                self.big_agent.train_step()

            self.total_reward_big += reward
            self.reward_count     += 1

            offspring = fish.try_breed(self.big_fishes)
            new_big.extend(offspring)

        self.big_fishes.extend(new_big)

        # ================================================================ #
        #  小鱼 AI 步
        # ================================================================ #
        new_small = []
        for fish in self.small_fishes:
            if not fish.alive:
                continue

            state  = fish.get_state(self.big_fishes, self.seaweeds)
            action = self.small_agent.choose_action(state, self.training)
            fish.action = action

            dx, dy = ACTIONS[action]
            fish.move(dx, dy)

            eat_reward = fish.try_eat(self.seaweeds)
            fish.update(self.big_fishes, self.seaweeds)

            reward = self._calc_small_reward(fish, eat_reward)
            done   = not fish.alive

            next_state = fish.get_state(self.big_fishes, self.seaweeds)

            if self.training:
                self.small_agent.remember(state, action, reward,
                                          next_state, done)
                self.small_agent.train_step()

            self.total_reward_small += reward
            self.reward_count       += 1

            offspring = fish.try_breed(self.small_fishes)
            new_small.extend(offspring)

        self.small_fishes.extend(new_small)

        # ---- ε 衰减（每 100 帧）----
        if self.training and self.frame % 100 == 0:
            self.big_agent.decay_epsilon()
            self.small_agent.decay_epsilon()

        # ---- 目标网络更新 ----
        if self.training and self.frame % (TARGET_UPDATE * 10) == 0:
            self.big_agent.update_target_net()
            self.small_agent.update_target_net()

        # ---- 清理死亡实体 ----
        self.big_fishes   = [f for f in self.big_fishes   if f.alive]
        self.small_fishes = [f for f in self.small_fishes if f.alive]

        # ---- 平均奖励 ----
        if self.reward_count > 0:
            self.avg_reward = ((self.total_reward_big + self.total_reward_small)
                               / self.reward_count)

        # ---- UI 历史 ----
        if self.frame % 10 == 0:
            self.ui.update_history(
                sum(1 for f in self.big_fishes   if f.alive),
                sum(1 for f in self.small_fishes if f.alive),
                self.avg_reward,
            )

        # ---- 种群灭绝补救 ----
        if not any(f.alive for f in self.big_fishes):
            for _ in range(3):
                self.big_fishes.append(BigFish())

        if not any(f.alive for f in self.small_fishes):
            for _ in range(10):
                self.small_fishes.append(SmallFish())

        # ---- 定期自动保存 ----
        if self.training and self.frame % 3000 == 0:
            self.big_agent.save()
            self.small_agent.save()
            print(f"[Frame {self.frame}] 模型已自动保存")

    # ================================================================== #
    #  奖励函数
    # ================================================================== #
    def _calc_big_reward(self, fish: "BigFish", eat_reward: float) -> float:
        """
        大鱼奖励：
          +20.0  吃到小鱼
          +0.1   每帧存活
          -0.8×r 饥饿惩罚（r > 50%）
          +0.5   视野内有小鱼（引导追逐）
          -3.0   贴近边界
          -50.0  死亡
        """
        reward = eat_reward

        if not fish.alive:
            return reward - 50.0

        reward += 0.1

        hunger_ratio = fish.hunger / fish.hunger_max
        if hunger_ratio > 0.5:
            reward -= hunger_ratio * 0.8

        # 引导大鱼追小鱼：视野内有小鱼时给小奖励
        visible_sf = [sf for sf in self.small_fishes
                      if sf.alive and fish.can_see(sf.x, sf.y,
                                                   fish.vision, fov_deg=120)]
        if visible_sf:
            nearest = min(visible_sf, key=lambda s: fish.dist(s))
            d = fish.dist(nearest)
            # 越近奖励越大
            reward += max(0.0, 0.5 * (1.0 - d / fish.vision))

        # 边界惩罚
        margin = 40
        if (fish.x < margin
                or fish.x > SCREEN_WIDTH - UI.PANEL_WIDTH - margin
                or fish.y < margin
                or fish.y > SCREEN_HEIGHT - 60 - margin):
            reward -= 3.0

        return reward

    def _calc_small_reward(self, fish: "SmallFish", eat_reward: float) -> float:
        """
        小鱼奖励：
          +10.0  吃到海草
          +0.05  每帧存活
          -0.4×r 饥饿惩罚（r > 50%）
          +0.3   视野内有海草（引导觅食）
          -k     视野内有大鱼且距离近（引导逃跑）
          -1.5   贴近边界
          -30.0  死亡
        """
        reward = eat_reward

        if not fish.alive:
            return reward - 30.0

        reward += 0.05

        hunger_ratio = fish.hunger / fish.hunger_max
        if hunger_ratio > 0.5:
            reward -= hunger_ratio * 0.4

        # 引导觅食：视野内有海草时给小奖励
        visible_sw = [sw for sw in self.seaweeds
                      if sw.alive and fish.can_see(sw.x, sw.y,
                                                   fish.vision, fov_deg=160)]
        if visible_sw:
            nearest_sw = min(visible_sw, key=lambda s: fish.dist_xy(s.x, s.y))
            d_sw = fish.dist_xy(nearest_sw.x, nearest_sw.y)
            reward += max(0.0, 0.3 * (1.0 - d_sw / fish.vision))

        # 引导逃跑：视野内大鱼越近惩罚越大；视野外大鱼距离够远给奖励
        for bf in self.big_fishes:
            if not bf.alive:
                continue
            d = fish.dist(bf)
            if fish.can_see(bf.x, bf.y, fish.vision, fov_deg=140):
                # 视野内大鱼：越近惩罚越重
                if d < fish.eat_range * 2:
                    reward -= 5.0
                else:
                    reward -= max(0.0, (fish.vision - d) / fish.vision * 2.0)
            else:
                # 视野外大鱼：离得远给小奖励
                if d > 120:
                    reward += 0.3

        # 边界惩罚
        margin = 40
        if (fish.x < margin
                or fish.x > SCREEN_WIDTH - UI.PANEL_WIDTH - margin
                or fish.y < margin
                or fish.y > SCREEN_HEIGHT - 60 - margin):
            reward -= 1.5

        return reward

    # ================================================================== #
    #  渲染
    # ================================================================== #
    def _render(self):
        self.screen.fill(COLOR_BG)
        self._draw_background()

        for sw in self.seaweeds:
            sw.draw(self.screen)
        for sf in self.small_fishes:
            sf.draw(self.screen)
        for bf in self.big_fishes:
            bf.draw(self.screen)

        self._draw_status_bar()

        self.ui.draw(
            frame           = self.frame,
            big_fishes      = self.big_fishes,
            small_fishes    = self.small_fishes,
            seaweeds        = self.seaweeds,
            big_agent_eps   = self.big_agent.epsilon,
            small_agent_eps = self.small_agent.epsilon,
            avg_reward      = self.avg_reward,
            training        = self.training,
            paused          = self.paused,
        )

        pygame.display.flip()

    def _draw_background(self):
        """深海渐变背景 + 光线 + 气泡"""
        for y in range(0, SCREEN_HEIGHT - 60, 4):
            ratio = y / (SCREEN_HEIGHT - 60)
            r = int(10 * (1 - ratio))
            g = int(40 * (1 - ratio * 0.5))
            b = int(80 + 40 * (1 - ratio))
            pygame.draw.rect(self.screen, (r, g, b),
                             (0, y, SCREEN_WIDTH - UI.PANEL_WIDTH, 4))

        if self.frame % 5 == 0:
            bx = random.randint(0, SCREEN_WIDTH - UI.PANEL_WIDTH)
            by = random.randint(0, SCREEN_HEIGHT - 60)
            pygame.draw.circle(self.screen, (100, 160, 220),
                               (bx, by), random.randint(1, 3))

        for i in range(3):
            angle  = math.radians(-70 + i * 20)
            length = 600
            ex = int(200 + i * 300 + length * math.cos(angle))
            ey = int(length * math.sin(angle))
            ray_surf = pygame.Surface(
                (SCREEN_WIDTH - UI.PANEL_WIDTH, SCREEN_HEIGHT - 60),
                pygame.SRCALPHA)
            pygame.draw.polygon(ray_surf, (255, 255, 200, 8), [
                (200 + i * 300,      0),
                (200 + i * 300 + 30, 0),
                (ex + 30, ey),
                (ex,      ey),
            ])
            self.screen.blit(ray_surf, (0, 0))

    def _draw_status_bar(self):
        bar_rect = pygame.Rect(0, SCREEN_HEIGHT - 60,
                               SCREEN_WIDTH - UI.PANEL_WIDTH, 60)
        pygame.draw.rect(self.screen, COLOR_UI_BG, bar_rect)
        pygame.draw.line(self.screen, (60, 60, 100),
                         (0, SCREEN_HEIGHT - 60),
                         (SCREEN_WIDTH - UI.PANEL_WIDTH, SCREEN_HEIGHT - 60), 1)

        font        = pygame.font.SysFont("microsoftyahei", 14)
        alive_big   = sum(1 for f in self.big_fishes   if f.alive)
        alive_small = sum(1 for f in self.small_fishes if f.alive)
        alive_sw    = sum(1 for s in self.seaweeds      if s.alive)

        texts = [
            (f"帧: {self.frame}",              10),
            (f"大鱼: {alive_big}",            120),
            (f"小鱼: {alive_small}",          230),
            (f"海草: {alive_sw}",             340),
            (f"速度: x{self.speed_mul}",      450),
            (f"平均奖励: {self.avg_reward:+.2f}", 560),
        ]
        for text, x in texts:
            surf = font.render(text, True, COLOR_TEXT)
            self.screen.blit(surf, (x, SCREEN_HEIGHT - 42))

    # ================================================================== #
    #  按键处理
    # ================================================================== #
    def _handle_key(self, key) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        elif key == pygame.K_p:
            self.paused = not self.paused
            print("暂停" if self.paused else "继续")
        elif key == pygame.K_r:
            self.reset()
            print("游戏已重置")
        elif key == pygame.K_s:
            self.big_agent.save()
            self.small_agent.save()
            print("模型已手动保存")
        elif key == pygame.K_t:
            self.training = not self.training
            print(f"切换到{'训练' if self.training else '游戏'}模式")
        elif key == pygame.K_EQUALS or key == pygame.K_PLUS:
            self.speed_mul = min(8, self.speed_mul * 2)
            print(f"速度: x{self.speed_mul}")
        elif key == pygame.K_MINUS:
            self.speed_mul = max(1, self.speed_mul // 2)
            print(f"速度: x{self.speed_mul}")
        return True