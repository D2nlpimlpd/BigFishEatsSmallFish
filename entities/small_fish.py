# ============================================================
#  small_fish.py  —  小鱼类
# ============================================================
import random
import math
import pygame
from config import *
from .fish import Fish


class SmallFish(Fish):
    """
    小鱼：
    - 由 DQN 智能体控制（逃跑 + 觅食海草）
    - 只能吃海草
    - 饥饿超过 SMALL_FISH_HUNGER_MAX 帧后饿死
    - 前方扇形视野内感知大鱼（140°逃跑）和海草（160°觅食）
    """

    def __init__(self, x=None, y=None):
        x = x if x is not None else random.uniform(50, SCREEN_WIDTH  - 50)
        y = y if y is not None else random.uniform(50, SCREEN_HEIGHT - 100)
        super().__init__(
            x=x, y=y,
            size=SMALL_FISH_SIZE,
            speed=SMALL_FISH_SPEED,
            lifespan=SMALL_FISH_LIFESPAN,
            hunger_max=SMALL_FISH_HUNGER_MAX,
            breed_age=SMALL_FISH_BREED_AGE,
            breed_cd=SMALL_FISH_BREED_CD,
            color_male=COLOR_SMALL_MALE,
            color_female=COLOR_SMALL_FEMALE,
        )
        self.vision      = SMALL_FISH_VISION
        self.eat_range   = SMALL_FISH_EAT_RANGE
        self.breed_range = SMALL_FISH_BREED_RANGE
        self.reward      = 0.0
        self.action      = 8

    # ------------------------------------------------------------------ #
    #  每帧更新
    # ------------------------------------------------------------------ #
    def update(self, big_fishes, seaweeds):
        self.update_base()
        if not self.alive:
            return

        # 饥饿超限 → 直接饿死（与大鱼一致）
        if self.hunger >= self.hunger_max:
            self.die("饿死")
            return

        # 饥饿 > 60% 时开始扣 HP，越饿扣得越快
        if self.hunger > self.hunger_max * 0.6:
            hunger_ratio = self.hunger / self.hunger_max
            # 范围：0.08（60%饥饿）→ 0.20（100%饥饿）
            self.hp -= 0.08 + 0.12 * (hunger_ratio - 0.6) / 0.4

        if self.hp <= 0:
            self.die("HP归零")

    # ------------------------------------------------------------------ #
    #  进食海草
    # ------------------------------------------------------------------ #
    def try_eat(self, seaweeds) -> float:
        for sw in seaweeds:
            if sw.alive and self.dist_xy(sw.x, sw.y) < self.eat_range:
                sw.alive = False
                self.eat(nutrition=25)
                return 10.0
        return 0.0

    # ------------------------------------------------------------------ #
    #  繁殖
    # ------------------------------------------------------------------ #
    def try_breed(self, small_fishes) -> list:
        if not self.can_breed():
            return []
        for partner in small_fishes:
            if (partner.alive
                    and partner.gender == "male"
                    and partner.age >= SMALL_FISH_BREED_AGE
                    and self.dist(partner) < self.breed_range):
                self.breed_timer    = self.breed_cd
                partner.breed_timer = self.breed_cd
                offspring = []
                for _ in range(random.randint(2, 5)):
                    nx = self.x + random.uniform(-30, 30)
                    ny = self.y + random.uniform(-30, 30)
                    offspring.append(SmallFish(nx, ny))
                return offspring
        return []

    # ------------------------------------------------------------------ #
    #  状态向量（供 DQN 使用）
    # ------------------------------------------------------------------ #
    def get_state(self, big_fishes, seaweeds) -> list:
        """
        20 维状态向量：
        [自身 x, y, 饥饿率, HP率, 年龄率,           (5)
         视野内最近 3 条大鱼 dx,dy（逃跑）           (6)
         视野内最近 3 棵海草 dx,dy（觅食）           (6)
         朝向 dir_x, dir_y                            (2)
         边界距离 × 4                                 (4)  → 共 23，截到 20]
        实际取前 STATE_SIZE=20 维
        """
        W, H = SCREEN_WIDTH, SCREEN_HEIGHT - 60
        state = [
            self.x / W,
            self.y / H,
            self.hunger / self.hunger_max,
            self.hp / 100.0,
            self.age / self.lifespan,
        ]

        # 视野内大鱼（前方 140°，用于逃跑感知）
        visible_bf = [bf for bf in big_fishes
                      if bf.alive and self.can_see(bf.x, bf.y,
                                                   self.vision, fov_deg=140)]
        visible_bf.sort(key=lambda b: self.dist(b))
        for i in range(3):
            if i < len(visible_bf):
                bf = visible_bf[i]
                state += [(bf.x - self.x) / W,
                          (bf.y - self.y) / H]
            else:
                state += [0.0, 0.0]

        # 视野内海草（前方 160°，用于觅食感知）
        visible_sw = [sw for sw in seaweeds
                      if sw.alive and self.can_see(sw.x, sw.y,
                                                   self.vision, fov_deg=160)]
        visible_sw.sort(key=lambda s: self.dist_xy(s.x, s.y))
        for i in range(3):
            if i < len(visible_sw):
                sw = visible_sw[i]
                state += [(sw.x - self.x) / W,
                          (sw.y - self.y) / H]
            else:
                state += [0.0, 0.0]

        # 朝向
        state += [self.dir_x, self.dir_y]

        # 边界距离
        state += [
            self.x / W,
            (W - self.x) / W,
            self.y / H,
            (H - self.y) / H,
        ]

        while len(state) < STATE_SIZE:
            state.append(0.0)
        return state[:STATE_SIZE]