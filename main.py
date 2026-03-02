# ============================================================
#  main.py  —  程序主入口
#  用法：
#    python main.py          → 启动游戏（默认训练模式）
#    python main.py play     → 游戏模式（加载已训练模型）
#    python main.py train    → 无界面快速训练
# ============================================================
import sys
import os
import pygame
from config import *


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "train_gui"

    # -------------------------------------------------------- #
    #  无界面训练
    # -------------------------------------------------------- #
    if mode == "train":
        from trainer import HeadlessTrainer
        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else TRAIN_EPISODES
        HeadlessTrainer().run(episodes)
        return

    # -------------------------------------------------------- #
    #  Pygame 游戏界面
    # -------------------------------------------------------- #
    pygame.init()
    pygame.display.set_caption(TITLE)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # 设置任务栏图标
    icon_surf = pygame.Surface((32, 32), pygame.SRCALPHA)
    pygame.draw.circle(icon_surf, COLOR_BIG_MALE, (20, 16), 12)
    pygame.draw.circle(icon_surf, COLOR_SMALL_MALE, (10, 20), 7)
    pygame.display.set_icon(icon_surf)

    training = (mode != "play")

    from game import GameCore
    game = GameCore(screen, training=training)
    game.run()


if __name__ == "__main__":
    main()