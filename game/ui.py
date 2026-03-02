# ============================================================
#  ui.py  —  UI 渲染（侧边信息面板 + 统计图）
# ============================================================
import pygame
import math
from config import *


class UI:
    """
    负责渲染游戏右侧信息面板：
      - 种群数量统计
      - 大鱼 / 小鱼存活数折线图
      - 当前帧 ε 值、平均奖励
      - 操作说明
    """

    PANEL_WIDTH = 220
    GRAPH_H     = 120
    MAX_HISTORY = 300

    def __init__(self, screen: pygame.Surface):
        self.screen   = screen
        self.font_lg  = pygame.font.SysFont("microsoftyahei", 18, bold=True)
        self.font_sm  = pygame.font.SysFont("microsoftyahei", 13)
        self.font_xs  = pygame.font.SysFont("microsoftyahei", 11)

        # 历史数据（用于折线图）
        self.big_history   = []
        self.small_history = []
        self.reward_history= []

        # 面板区域
        self.panel_rect = pygame.Rect(
            SCREEN_WIDTH - self.PANEL_WIDTH, 0,
            self.PANEL_WIDTH, SCREEN_HEIGHT
        )

    # ------------------------------------------------------------------ #
    def update_history(self, big_count: int, small_count: int,
                       avg_reward: float):
        self.big_history.append(big_count)
        self.small_history.append(small_count)
        self.reward_history.append(avg_reward)
        if len(self.big_history) > self.MAX_HISTORY:
            self.big_history.pop(0)
            self.small_history.pop(0)
            self.reward_history.pop(0)

    # ------------------------------------------------------------------ #
    def draw(self, frame: int, big_fishes: list, small_fishes: list,
             seaweeds: list, big_agent_eps: float,
             small_agent_eps: float, avg_reward: float,
             training: bool, paused: bool):

        # ---- 面板背景 ----
        panel_surf = pygame.Surface(
            (self.PANEL_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        panel_surf.fill((*COLOR_UI_BG, 220))
        self.screen.blit(panel_surf, (SCREEN_WIDTH - self.PANEL_WIDTH, 0))

        px = SCREEN_WIDTH - self.PANEL_WIDTH + 10
        py = 10

        # ---- 标题 ----
        self._text("🐟 大鱼吃小鱼", px, py, self.font_lg,
                   (255, 215, 0)); py += 28
        mode_txt = "【训练模式】" if training else "【游戏模式】"
        mode_col = (255, 100, 100) if training else (100, 255, 100)
        self._text(mode_txt, px, py, self.font_sm, mode_col); py += 22
        if paused:
            self._text("⏸  已暂停", px, py, self.font_sm, (255, 200, 0))
        py += 22

        # ---- 分割线 ----
        self._divider(py); py += 12

        # ---- 种群信息 ----
        alive_big   = sum(1 for f in big_fishes   if f.alive)
        alive_small = sum(1 for f in small_fishes if f.alive)
        alive_sw    = sum(1 for s in seaweeds      if s.alive)

        self._text(f"⏱  帧数: {frame}", px, py, self.font_sm,
                   COLOR_TEXT); py += 20
        self._text(f"🦈 大鱼: {alive_big}", px, py, self.font_sm,
                   COLOR_BIG_FEMALE); py += 20
        self._text(f"🐠 小鱼: {alive_small}", px, py, self.font_sm,
                   COLOR_SMALL_FEMALE); py += 20
        self._text(f"🌿 海草: {alive_sw}", px, py, self.font_sm,
                   COLOR_SEAWEED); py += 22

        # ---- ε 值 ----
        self._divider(py); py += 12
        self._text("Q-Learning 参数", px, py, self.font_sm,
                   (180, 180, 255)); py += 20
        self._text(f"大鱼 ε : {big_agent_eps:.3f}", px, py,
                   self.font_xs, COLOR_BIG_MALE); py += 17
        self._text(f"小鱼 ε : {small_agent_eps:.3f}", px, py,
                   self.font_xs, COLOR_SMALL_MALE); py += 17
        self._text(f"平均奖励: {avg_reward:+.2f}", px, py,
                   self.font_xs, COLOR_TEXT); py += 22

        # ---- 折线图：种群数量 ----
        self._divider(py); py += 8
        self._text("种群数量趋势", px, py, self.font_xs,
                   (180, 180, 255)); py += 16
        self._draw_graph(px, py, self.big_history,   self.small_history)
        py += self.GRAPH_H + 8

        # ---- 折线图：平均奖励 ----
        self._text("平均奖励趋势", px, py, self.font_xs,
                   (180, 255, 180)); py += 16
        self._draw_reward_graph(px, py)
        py += self.GRAPH_H + 8

        # ---- 操作说明 ----
        self._divider(py); py += 8
        self._text("操作说明", px, py, self.font_sm, (180, 180, 255)); py += 18
        tips = [
            "P  —  暂停 / 继续",
            "R  —  重置游戏",
            "S  —  保存模型",
            "T  —  切换训练/游戏",
            "+/- — 加速 / 减速",
            "ESC — 退出",
        ]
        for tip in tips:
            self._text(tip, px, py, self.font_xs, (160, 160, 160)); py += 16

    # ------------------------------------------------------------------ #
    #  折线图（种群数量）
    # ------------------------------------------------------------------ #
    def _draw_graph(self, x: int, y: int,
                    data_a: list, data_b: list):
        gw = self.PANEL_WIDTH - 20
        gh = self.GRAPH_H
        rect = pygame.Rect(x, y, gw, gh)
        pygame.draw.rect(self.screen, (20, 30, 60), rect)
        pygame.draw.rect(self.screen, (60, 60, 100), rect, 1)

        if len(data_a) < 2:
            return

        max_val = max(max(data_a, default=1),
                      max(data_b, default=1), 1)

        def to_screen(idx, val):
            sx = x + int(idx / (self.MAX_HISTORY - 1) * gw)
            sy = y + gh - int(val / max_val * (gh - 4)) - 2
            return sx, sy

        n = len(data_a)
        # 大鱼线（橙）
        pts_a = [to_screen(i, data_a[i]) for i in range(n)]
        if len(pts_a) >= 2:
            pygame.draw.lines(self.screen, COLOR_BIG_MALE, False, pts_a, 1)

        # 小鱼线（浅蓝）
        pts_b = [to_screen(i, data_b[i]) for i in range(n)]
        if len(pts_b) >= 2:
            pygame.draw.lines(self.screen, COLOR_SMALL_MALE, False, pts_b, 1)

        # 图例
        pygame.draw.line(self.screen, COLOR_BIG_MALE,
                         (x + gw - 60, y + 8), (x + gw - 46, y + 8), 2)
        self._text("大鱼", x + gw - 44, y + 3,
                   self.font_xs, COLOR_BIG_MALE)
        pygame.draw.line(self.screen, COLOR_SMALL_MALE,
                         (x + gw - 60, y + 20), (x + gw - 46, y + 20), 2)
        self._text("小鱼", x + gw - 44, y + 15,
                   self.font_xs, COLOR_SMALL_MALE)

    # ------------------------------------------------------------------ #
    #  折线图（平均奖励）
    # ------------------------------------------------------------------ #
    def _draw_reward_graph(self, x: int, y: int):
        gw  = self.PANEL_WIDTH - 20
        gh  = self.GRAPH_H
        data = self.reward_history
        rect = pygame.Rect(x, y, gw, gh)
        pygame.draw.rect(self.screen, (20, 30, 60), rect)
        pygame.draw.rect(self.screen, (60, 60, 100), rect, 1)

        if len(data) < 2:
            return

        min_v = min(data)
        max_v = max(data)
        span  = max(max_v - min_v, 1)

        def to_screen(idx, val):
            sx = x + int(idx / (self.MAX_HISTORY - 1) * gw)
            sy = y + gh - int((val - min_v) / span * (gh - 4)) - 2
            return sx, sy

        n   = len(data)
        pts = [to_screen(i, data[i]) for i in range(n)]
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, (100, 255, 150), False, pts, 1)

        # 零线
        if min_v < 0 < max_v:
            zero_y = y + gh - int((0 - min_v) / span * (gh - 4)) - 2
            pygame.draw.line(self.screen, (100, 100, 100),
                             (x, zero_y), (x + gw, zero_y), 1)

    # ------------------------------------------------------------------ #
    #  工具方法
    # ------------------------------------------------------------------ #
    def _text(self, text: str, x: int, y: int,
              font: pygame.font.Font, color: tuple):
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))

    def _divider(self, y: int):
        pygame.draw.line(
            self.screen, (60, 60, 100),
            (SCREEN_WIDTH - self.PANEL_WIDTH + 5, y),
            (SCREEN_WIDTH - 5, y), 1
        )