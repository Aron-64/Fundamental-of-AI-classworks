# frozenlake_llm_rescue.py
# 组员：刘航程(22404150104) 
# 作业02：智语破冰——大模型赋能的Frozen Lake挑战

import os
import sys

# 确保 src 目录可被导入（运行时追加路径）
sys.path.append(os.path.dirname(__file__))

import gymnasium as gym
from envs.math_rescue import MathRescueWrapper
from agent.llm_agent import get_action_from_llm, ACTIONS
from viz.plotting import plot_trajectory

def play_game():
    env = gym.make('FrozenLake-v1', is_slippery=True, render_mode="rgb_array")
    env = MathRescueWrapper(env)

    obs, info = env.reset()
    env.update_desc()
    print("🤖 欢迎来到由刘航程构建的<带有LLM救援的Frozen Lake游戏>!")
    print("="*50)

    trajectory = [obs]
    done = False
    step_count = 0
    max_steps = 100
    reward = 0.0

    while not done and step_count < max_steps:
        try:
            _frame = env.render()
        except Exception:
            _frame = None

        if env.in_question_mode:
            print("🧠 LLM is solving the rescue question...")
            obs, reward, terminated, truncated, info = env.answer_question()
            done = terminated or truncated
            if isinstance(obs, tuple):
                obs = obs[0]
            print(f"Back to state {obs}.")
        else:
            action = get_action_from_llm(obs, env.env.unwrapped.desc)
            print(f"State {obs} → LLM chose: {ACTIONS.get(action, action)}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if isinstance(obs, tuple):
                obs = obs[0]
            trajectory.append(obs)
            step_count += 1

    print(f"Game Over! Final reward: {reward}")
    plot_trajectory(trajectory, env.env.unwrapped.desc)

if __name__ == "__main__":
    play_game()