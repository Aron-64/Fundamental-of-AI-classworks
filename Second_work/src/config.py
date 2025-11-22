import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# 环境
env = gym.make('FrozenLake-v1')

# 参数
RENDER_MODE = "human"   # 或 "rgb_array"（无 GUI 时）
MAX_STEPS = 200
FROZEN_ENV_ID = "FrozenLake-v1"

# 初始化
state_values = defaultdict(float)
policy = defaultdict(lambda: 0)
alpha = 0.5
gamma = 0.9
epsilon = 0.1
num_episodes = 1000

# 训练
for episode in range(num_episodes):
    state = env.reset()
    for step in range(MAX_STEPS):
        action = policy[state]
        next_state, reward, done, _ = env.step(action)
        state_values[state] += alpha * (reward + gamma * state_values[next_state] - state_values[state])
        policy[state] = action
        if done:
            break
    if episode % 100 == 0:
        print(f"Episode {episode} complete")

# 输出
print("State values:", state_values)
print("Policy:", policy)