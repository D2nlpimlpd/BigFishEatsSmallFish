# -*- coding: utf-8 -*-
# ============================================================
#  config.py  —  全局配置
# ============================================================

# ---------- 窗口 ----------
SCREEN_WIDTH  = 1200
SCREEN_HEIGHT = 800
FPS           = 60
TITLE         = "大鱼吃小鱼 - Q-Learning 生态模拟"

# ---------- 颜色 ----------
COLOR_BG          = (10,  40,  80)    # 深海蓝背景
COLOR_BIG_MALE    = (255, 140,   0)   # 雄大鱼：橙色
COLOR_BIG_FEMALE  = (255, 200,  50)   # 雌大鱼：金黄
COLOR_SMALL_MALE  = (100, 200, 255)   # 雄小鱼：浅蓝
COLOR_SMALL_FEMALE= (180, 230, 255)   # 雌小鱼：淡蓝
COLOR_SEAWEED     = (34,  139,  34)   # 海草：绿色
COLOR_UI_BG       = (0,   20,  60)    # UI 背景
COLOR_TEXT        = (220, 220, 220)
COLOR_HP_BAR_BG   = (80,  80,  80)
COLOR_HP_BAR_FG   = (50,  205,  50)
COLOR_HUNGER_FG   = (255, 165,   0)

# ---------- 大鱼参数 ----------
BIG_FISH_INIT_COUNT   = 6          # 初始数量
BIG_FISH_LIFESPAN     = 3000       # 寿命（帧）
BIG_FISH_HUNGER_MAX   = 600        # 最大饥饿值（帧），超过则饿死
BIG_FISH_SPEED        = 6.5        # 基础速度
BIG_FISH_SIZE         = 28         # 半径
BIG_FISH_BREED_AGE    = 800        # 可繁殖年龄（帧）
BIG_FISH_BREED_CD     = 500        # 繁殖冷却（帧）
BIG_FISH_BREED_RANGE  = 80         # 繁殖感应距离
BIG_FISH_EAT_RANGE    = 35         # 吃鱼距离
BIG_FISH_VISION       = 200        # 视野范围

# ---------- 小鱼参数 ----------
SMALL_FISH_INIT_COUNT  = 20
SMALL_FISH_LIFESPAN    = 2000
SMALL_FISH_SPEED       = 4.5
SMALL_FISH_SIZE        = 14
SMALL_FISH_BREED_AGE   = 600
SMALL_FISH_BREED_CD    = 400
SMALL_FISH_BREED_RANGE = 60
SMALL_FISH_EAT_RANGE   = 20
SMALL_FISH_VISION      = 150
SMALL_FISH_HUNGER_MAX  = 800       # 小鱼饥饿上限（不吃海草也不立即死，但扣HP）

# ---------- 海草参数 ----------
SEAWEED_INIT_COUNT    = 30
SEAWEED_SPAWN_INTERVAL= 60       # 每隔多少帧刷新一棵
SEAWEED_MAX_COUNT     = 80
SEAWEED_SIZE          = 10

# ---------- Q-Learning 参数 ----------
LR            = 0.001
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE    = 64
MEMORY_SIZE   = 10000
TARGET_UPDATE = 50              # 每隔多少 episode 更新目标网络

# 动作空间（8方向移动 + 停止）
ACTIONS = [
    ( 0, -1),  # 上
    ( 0,  1),  # 下
    (-1,  0),  # 左
    ( 1,  0),  # 右
    ( 1, -1),  # 右上
    ( 1,  1),  # 右下
    (-1, -1),  # 左上
    (-1,  1),  # 左下
    ( 0,  0),  # 停止
]
ACTION_SIZE  = len(ACTIONS)      # 9
STATE_SIZE   = 20                # 状态向量维度

# ---------- 模型保存路径 ----------
BIG_FISH_MODEL_PATH   = "big_fish_dqn.pth"
SMALL_FISH_MODEL_PATH = "small_fish_dqn.pth"

# ---------- 训练模式 ----------
TRAINING_MODE  = True   # True = 训练模式，False = 游戏模式
TRAIN_EPISODES = 500