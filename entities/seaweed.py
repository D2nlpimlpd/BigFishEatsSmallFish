# ============================================================
#  seaweed.py  —  海草类
# ============================================================
import random
import math
import pygame
from config import *


class Seaweed:
    """
    海草：随机刷新在地图上，小鱼吃后消失，有摇曳动画。
    """

    def __init__(self, x=None, y=None):
        self.x     = x if x is not None else random.uniform(30, SCREEN_WIDTH  - 30)
        self.y     = y if y is not None else random.uniform(30, SCREEN_HEIGHT - 100)
        self.alive = True
        self.size  = SEAWEED_SIZE
        self.sway       = random.uniform(0, math.pi * 2)   # 随机初始相位
        self.sway_speed = random.uniform(0.03, 0.07)       # 稍快摇曳
        self.height     = random.randint(24, 48)
        # 随机色调让海草更自然
        self.color_base = (
            random.randint(15, 30),
            random.randint(120, 180),
            random.randint(15, 40),
        )

    def update(self):
        self.sway += self.sway_speed

    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return

        segments = 5
        x0, y0 = self.x, self.y
        seg_h   = self.height / segments

        for i in range(segments):
            sway_offset = math.sin(self.sway + i * 0.6) * 7
            x1 = x0 + sway_offset
            y1 = y0 - seg_h

            # 越往顶部颜色越亮
            ratio = i / segments
            color = (
                min(255, int(self.color_base[0] + 10  * ratio)),
                min(255, int(self.color_base[1] + 60  * ratio)),
                min(255, int(self.color_base[2] + 10  * ratio)),
            )
            pygame.draw.line(surface, color,
                             (int(x0), int(y0)),
                             (int(x1), int(y1)), 3)

            # 叶片（椭圆，交替左右）
            side = 1 if i % 2 == 0 else -1
            lx = x1 + side * 7
            ly = (y0 + y1) / 2
            pygame.draw.ellipse(surface, (34, 139, 34),
                                (int(lx) - 7, int(ly) - 4, 14, 8))
            x0, y0 = x1, y1

        # 底部根节
        pygame.draw.circle(surface, (20, 90, 20),
                           (int(self.x), int(self.y)), 5)