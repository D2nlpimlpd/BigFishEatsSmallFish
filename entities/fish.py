# ============================================================
#  fish.py  —  鱼类基类
# ============================================================
import random
import math
import pygame
from config import *


class Fish:
    """
    所有鱼的基类，封装通用属性：
    位置、性别、年龄、生命值、饥饿、繁殖冷却、朝向、视野
    """
    _id_counter = 0

    def __init__(self, x: float, y: float,
                 size: int, speed: float,
                 lifespan: int, hunger_max: int,
                 breed_age: int, breed_cd: int,
                 color_male: tuple, color_female: tuple):

        Fish._id_counter += 1
        self.id   = Fish._id_counter

        # 位置与运动
        self.x     = x
        self.y     = y
        self.vx    = random.uniform(-speed, speed)
        self.vy    = random.uniform(-speed, speed)
        self.speed = speed
        self.size  = size

        # 朝向（单位向量，默认朝右）
        self.dir_x = 1.0
        self.dir_y = 0.0

        # 生物属性
        self.gender   = random.choice(["male", "female"])
        self.age      = 0
        self.lifespan = lifespan
        self.hp       = 100
        self.alive    = True

        # 饥饿
        self.hunger     = 0
        self.hunger_max = hunger_max

        # 繁殖
        self.breed_age   = breed_age
        self.breed_cd    = breed_cd
        self.breed_timer = 0

        # 颜色
        self.color = color_male if self.gender == "male" else color_female

        # 尾迹
        self.trail = []

    # ------------------------------------------------------------------ #
    #  移动
    # ------------------------------------------------------------------ #
    def move(self, dx: float, dy: float):
        """按指定方向移动，并限制在屏幕内，同时更新朝向"""
        norm = math.hypot(dx, dy)
        if norm > 0:
            # 更新朝向（归一化方向向量）
            self.dir_x = dx / norm
            self.dir_y = dy / norm
            dx = self.dir_x * self.speed
            dy = self.dir_y * self.speed

        self.x = max(self.size,
                     min(SCREEN_WIDTH  - self.size, self.x + dx))
        self.y = max(self.size,
                     min(SCREEN_HEIGHT - self.size - 60, self.y + dy))

        # 记录尾迹
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 8:
            self.trail.pop(0)

    def wander(self):
        """随机游荡（无 AI 控制时使用）"""
        self.vx += random.uniform(-0.3, 0.3)
        self.vy += random.uniform(-0.3, 0.3)
        spd = math.hypot(self.vx, self.vy)
        if spd > self.speed:
            self.vx = self.vx / spd * self.speed
            self.vy = self.vy / spd * self.speed
        self.move(self.vx, self.vy)

    # ------------------------------------------------------------------ #
    #  视野检测（前方扇形）
    # ------------------------------------------------------------------ #
    def can_see(self, tx: float, ty: float,
                vision_dist: float, fov_deg: float = 120.0) -> bool:
        """
        判断点 (tx, ty) 是否在以 self 为顶点的前方扇形视野内。
        vision_dist : 视野半径（像素）
        fov_deg     : 视野张角（度），左右各 fov/2
        """
        dx = tx - self.x
        dy = ty - self.y
        d  = math.hypot(dx, dy)
        if d < 1e-6 or d > vision_dist:
            return False

        # 朝向单位向量
        fn = math.hypot(self.dir_x, self.dir_y)
        if fn < 1e-6:
            fx, fy = 1.0, 0.0
        else:
            fx, fy = self.dir_x / fn, self.dir_y / fn

        # 目标方向单位向量
        ux, uy = dx / d, dy / d

        # 夹角
        dot = max(-1.0, min(1.0, ux * fx + uy * fy))
        angle = math.degrees(math.acos(dot))
        return angle <= (fov_deg / 2.0)

    # ------------------------------------------------------------------ #
    #  状态更新
    # ------------------------------------------------------------------ #
    def update_base(self):
        """每帧调用，更新年龄、饥饿、繁殖冷却"""
        if not self.alive:
            return

        self.age    += 1
        self.hunger += 1
        if self.breed_timer > 0:
            self.breed_timer -= 1

        if self.age >= self.lifespan:
            self.die("老死")

    def eat(self, nutrition: int = 30):
        """进食，恢复 HP 并清空饥饿"""
        self.hunger = 0
        self.hp = min(100, self.hp + nutrition)

    def die(self, reason: str = "unknown"):
        self.alive = False

    # ------------------------------------------------------------------ #
    #  繁殖
    # ------------------------------------------------------------------ #
    def can_breed(self) -> bool:
        return (self.alive
                and self.age >= self.breed_age
                and self.breed_timer == 0
                and self.gender == "female")

    def breed_with(self, partner):
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  距离工具
    # ------------------------------------------------------------------ #
    def dist(self, other) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def dist_xy(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)

    # ------------------------------------------------------------------ #
    #  渲染
    # ------------------------------------------------------------------ #
    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return

        # ---- 尾迹 ----
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int(120 * i / max(len(self.trail), 1))
            r = max(2, self.size // 3 - i)
            trail_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, (*self.color, alpha), (r, r), r)
            surface.blit(trail_surf, (tx - r, ty - r))

        # ---- 鱼身 ----
        pygame.draw.circle(surface, self.color,
                           (int(self.x), int(self.y)), self.size)

        # ---- 眼睛（跟随朝向）----
        # 眼睛偏移：朝向前方偏上方
        perp_x = -self.dir_y   # 垂直于朝向（用于上下偏移）
        perp_y =  self.dir_x

        eye_offset_fwd  = self.size * 0.45   # 沿朝向向前偏移
        eye_offset_side = self.size * 0.30   # 垂直朝向偏移（偏上）

        ex = self.x + self.dir_x * eye_offset_fwd + perp_x * eye_offset_side
        ey = self.y + self.dir_y * eye_offset_fwd + perp_y * eye_offset_side

        eye_r = max(2, self.size // 4)

        # 眼白
        pygame.draw.circle(surface, (240, 240, 240),
                           (int(ex), int(ey)), eye_r)
        # 瞳孔（稍微朝向前方）
        px = ex + self.dir_x * eye_r * 0.35
        py = ey + self.dir_y * eye_r * 0.35
        pygame.draw.circle(surface, (15, 15, 15),
                           (int(px), int(py)), max(1, eye_r // 2))
        # 高光
        hx = ex - self.dir_x * eye_r * 0.15 + perp_x * eye_r * 0.2
        hy = ey - self.dir_y * eye_r * 0.15 + perp_y * eye_r * 0.2
        pygame.draw.circle(surface, (255, 255, 255),
                           (int(hx), int(hy)), max(1, eye_r // 3))

        # ---- 性别标记 ----
        icon = "♂" if self.gender == "male" else "♀"
        font = pygame.font.SysFont("segoeui", 14, bold=True)
        txt  = font.render(icon, True, (255, 255, 255))
        surface.blit(txt, (int(self.x) - 5, int(self.y) - 7))

        # ---- HP 血条 ----
        bar_w = self.size * 2
        bar_x = int(self.x) - self.size
        bar_y = int(self.y) - self.size - 10
        pygame.draw.rect(surface, COLOR_HP_BAR_BG, (bar_x, bar_y, bar_w, 4))
        hp_ratio = max(0.0, min(1.0, self.hp / 100.0))
        pygame.draw.rect(surface, COLOR_HP_BAR_FG,
                         (bar_x, bar_y, int(bar_w * hp_ratio), 4))

        # ---- 饥饿条 ----
        hunger_ratio = max(0.0, min(1.0, self.hunger / self.hunger_max))
        hunger_color = (
            int(50  + 200 * hunger_ratio),
            int(200 - 180 * hunger_ratio),
            50
        )
        pygame.draw.rect(surface, COLOR_HP_BAR_BG,
                         (bar_x, bar_y + 5, bar_w, 3))
        pygame.draw.rect(surface, hunger_color,
                         (bar_x, bar_y + 5, int(bar_w * hunger_ratio), 3))