# ============================================================
#  big_fish.py  —  大鱼类
# ============================================================
import random
import math
import pygame
from config import *
from .fish import Fish


class BigFish(Fish):
    """
    大鱼：
    - 由 DQN 智能体控制
    - 饥饿超过 BIG_FISH_HUNGER_MAX 帧后饿死
    - 吃小鱼恢复饱腹
    - 前方扇形视野（120°）内感知小鱼
    - 到达繁殖年龄、冷却结束后可繁殖
    """

    def __init__(self, x=None, y=None):
        x = x if x is not None else random.uniform(50, SCREEN_WIDTH  - 50)
        y = y if y is not None else random.uniform(50, SCREEN_HEIGHT - 100)
        super().__init__(
            x=x, y=y,
            size=BIG_FISH_SIZE,
            speed=BIG_FISH_SPEED,
            lifespan=BIG_FISH_LIFESPAN,
            hunger_max=BIG_FISH_HUNGER_MAX,
            breed_age=BIG_FISH_BREED_AGE,
            breed_cd=BIG_FISH_BREED_CD,
            color_male=COLOR_BIG_MALE,
            color_female=COLOR_BIG_FEMALE,
        )
        self.vision      = BIG_FISH_VISION
        self.eat_range   = BIG_FISH_EAT_RANGE
        self.breed_range = BIG_FISH_BREED_RANGE
        self.total_eaten = 0
        self.reward      = 0.0
        self.action      = 8   # 默认停止

    # ------------------------------------------------------------------ #
    #  每帧更新
    # ------------------------------------------------------------------ #
    def update(self, small_fishes, big_fishes):
        self.update_base()
        if not self.alive:
            return

        # 饥饿超限 → 直接饿死
        if self.hunger >= self.hunger_max:
            self.die("饿死")
            return

        # 饥饿 > 70% 时开始扣 HP
        if self.hunger > self.hunger_max * 0.7:
            hunger_ratio = self.hunger / self.hunger_max
            self.hp -= 0.05 + 0.10 * (hunger_ratio - 0.7) / 0.3

        if self.hp <= 0:
            self.die("HP归零")

    # ------------------------------------------------------------------ #
    #  进食
    # ------------------------------------------------------------------ #
    def try_eat(self, small_fishes) -> float:
        for sf in small_fishes:
            if sf.alive and self.dist(sf) < self.eat_range:
                sf.die("被大鱼吃掉")
                self.eat(nutrition=40)
                self.total_eaten += 1
                return 20.0
        return 0.0

    # ------------------------------------------------------------------ #
    #  繁殖
    # ------------------------------------------------------------------ #
    def try_breed(self, big_fishes) -> list:
        if not self.can_breed():
            return []
        for partner in big_fishes:
            if (partner.alive
                    and partner.gender == "male"
                    and partner.age >= BIG_FISH_BREED_AGE
                    and self.dist(partner) < self.breed_range):
                self.breed_timer    = self.breed_cd
                partner.breed_timer = self.breed_cd
                offspring = []
                for _ in range(random.randint(1, 3)):
                    nx = self.x + random.uniform(-40, 40)
                    ny = self.y + random.uniform(-40, 40)
                    offspring.append(BigFish(nx, ny))
                return offspring
        return []

    # ------------------------------------------------------------------ #
    #  状态向量（供 DQN 使用）
    # ------------------------------------------------------------------ #
    def get_state(self, small_fishes, big_fishes) -> list:
        """
        20 维状态向量：
        [自身 x, y, 饥饿率, HP率, 年龄率,          (5)
         视野内最近 3 条小鱼 dx,dy                   (6)
         最近繁殖伙伴 dx,dy                           (2)
         朝向 dir_x, dir_y                            (2)
         边界距离 × 4                                 (4)
         补 0 到 20]
        """
        W, H = SCREEN_WIDTH, SCREEN_HEIGHT - 60
        state = [
            self.x / W,
            self.y / H,
            self.hunger / self.hunger_max,
            self.hp / 100.0,
            self.age / self.lifespan,
        ]

        # 视野内的小鱼（前方 120° 扇形）
        visible_sf = [sf for sf in small_fishes
                      if sf.alive and self.can_see(sf.x, sf.y,
                                                   self.vision, fov_deg=120)]
        visible_sf.sort(key=lambda s: self.dist(s))
        for i in range(3):
            if i < len(visible_sf):
                sf = visible_sf[i]
                state += [(sf.x - self.x) / W,
                          (sf.y - self.y) / H]
            else:
                state += [0.0, 0.0]

        # 最近繁殖伙伴（不限视野）
        partners = [b for b in big_fishes
                    if b.alive and b.id != self.id
                    and b.gender == (
                        "male" if self.gender == "female" else "female")]
        if partners:
            partners.sort(key=lambda b: self.dist(b))
            bp = partners[0]
            state += [(bp.x - self.x) / W, (bp.y - self.y) / H]
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

    # ------------------------------------------------------------------ #
    #  渲染（在基类眼睛基础上加鱼鳍）
    # ------------------------------------------------------------------ #
    def draw(self, surface: pygame.Surface):
        super().draw(surface)
        if not self.alive:
            return

        # 鱼鳍（与朝向相关，画在鱼身后上方）
        # 垂直方向
        perp_x = -self.dir_y
        perp_y =  self.dir_x

        # 鳍三个顶点
        tip_x  = self.x - self.dir_x * self.size * 0.3 + perp_x * self.size * 1.1
        tip_y  = self.y - self.dir_y * self.size * 0.3 + perp_y * self.size * 1.1
        base1x = self.x + self.dir_x * self.size * 0.4
        base1y = self.y + self.dir_y * self.size * 0.4
        base2x = self.x - self.dir_x * self.size * 0.8
        base2y = self.y - self.dir_y * self.size * 0.8

        fin_color = tuple(max(0, c - 50) for c in self.color)
        pygame.draw.polygon(surface, fin_color, [
            (int(tip_x),  int(tip_y)),
            (int(base1x), int(base1y)),
            (int(base2x), int(base2y)),
        ])

        # 视野弧线（调试可视化，仅在训练模式下显示淡蓝色扇形）
        # 如不需要可删除此段
        fov_surf = pygame.Surface(
            (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        angle_center = math.degrees(math.atan2(self.dir_y, self.dir_x))
        rect = pygame.Rect(
            int(self.x - self.vision),
            int(self.y - self.vision),
            int(self.vision * 2),
            int(self.vision * 2))
        pygame.draw.arc(fov_surf, (100, 200, 255, 25),
                        rect,
                        math.radians(angle_center - 60),
                        math.radians(angle_center + 60), 2)
        surface.blit(fov_surf, (0, 0))