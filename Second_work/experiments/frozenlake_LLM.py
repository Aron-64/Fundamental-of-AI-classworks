# frozenlake_LLM.py
# 组员：*****************************************
# 作业02：智语破冰——大模型赋能的Frozen Lake挑战

import gymnasium as gym
from gymnasium import Wrapper
import random
import numpy as np
import requests
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches



# ========================
# Qwen API 
# ========================
QWEN_API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
if not QWEN_API_KEY:
    raise ValueError("Please set QWEN_API_KEY/DASHSCOPE_API_KEY environment variable.")

def call_qwen(prompt, max_tokens=50):
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen3-max",
        "input": {
            "messages": [
                {"role": "system", "content": "You are a precise AI assistant."},
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {"max_tokens": max_tokens, "temperature": 0.1}
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        result = response.json()
        return result["output"]["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ Qwen API error: {e}")
        return ""

# ========================
# 自定义包装器
# ========================
class MathRescueWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.holes = [1, 4, 5, 7, 11, 12]
        self.goal = 15
        self.llm_question = None
        self.llm_answer = None
        self.in_question_mode = False
        self.last_state = 0
        self.question_hole = None
        self.update_desc()

    def update_desc(self):
        desc = []
        mapping = {0: b'S', 15: b'G'}
        for i in range(4):
            row = []
            for j in range(4):
                state = i * 4 + j
                if state in self.holes:
                    row.append(b'H')
                else:
                    row.append(mapping.get(state, b'F'))
            desc.append(row)
        self.env.unwrapped.desc = np.array(desc)

    def step(self, action):
        # 安全读取底层环境的内部状态（兼容不同实现）
        try:
            last = getattr(self.env.unwrapped, "s", None)
            self.last_state = int(last) if last is not None else self.last_state
        except Exception:
            pass

        # 兼容 gymnasium 的 step 签名：obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 处理 obs 可能为元组 (obs, info)
        if isinstance(obs, tuple):
            state = int(obs[0])
        else:
            try:
                state = int(obs)
            except Exception:
                state = obs
        if not self.in_question_mode:
            # 掉入初始 holes 才触发问题（避免重复触发）
            if state in [1, 4, 5, 7, 11, 12] and state in self.holes:
                print(f"🕳️ Uh-oh! You fell into a hole at state {state}!")
                self.llm_question, self.llm_answer = self.ask_math_question()
                print(f"❓ {self.llm_question}")
                self.in_question_mode = True
                self.question_hole = state
                obs = self.last_state
                self.env.unwrapped.s = obs
                terminated = False
                truncated = False
                reward = 0
            elif state == 15:
                if self.goal == 15:
                    print("🎉 You reached the goal! You win!")
                    reward = 1
                    terminated = True
                    truncated = False
                else:
                    print("💀 The goal has turned into a hole! You lose!")
                    reward = -1
                    terminated = True
                    truncated = False
            else:
                # 普通移动或掉入已拯救的洞（应被阻止）
                pass
            return obs, reward, terminated, truncated, info
        else:
            # 不应在 step 中处理答题（由主循环调用 answer_question）
            return self.last_state, 0.0, False, False, {}

    def ask_math_question(self):
        """用 Qwen 生成数学/逻辑题"""
        prompt = """
Generate a simple math or logic question with a single numerical answer.
Format:
Question: <your question>
Answer: <number>

Example:
Question: What is the square root of 64?
Answer: 8
"""
        response = call_qwen(prompt)
        lines = response.strip().split('\n')
        q = lines[0].replace("Question:", "").strip() if len(lines) > 0 else "What is 2+2?"
        a_str = lines[1].replace("Answer:", "").strip() if len(lines) > 1 else "4"
        try:
            a = float(a_str)
        except:
            a = 4.0
        return q, a

    def answer_question(self):
        """让 Qwen 自己回答并判断"""
        if not self.in_question_mode:
            return self.last_state, 0, False, False, {}

        # 让 Qwen 回答
        answer_prompt = f"Question: {self.llm_question}\nGive only the numerical answer."
        pred_text = call_qwen(answer_prompt, max_tokens=10)

        # 让 Qwen 判断是否正确
        judge_prompt = f"""
Question: {self.llm_question}
Correct Answer: {self.llm_answer}
Predicted Answer: {pred_text}

Is the predicted answer correct? Respond ONLY with "YES" or "NO".
"""
        judgment = call_qwen(judge_prompt, max_tokens=5).strip().upper()

        is_correct = (judgment == "YES")

        if is_correct:
            print("✅ Correct! The goal (15) is restored.")
            self.goal = 15
            if self.question_hole in self.holes:
                self.holes.remove(self.question_hole)
                self.update_desc()
        else:
            print("❌ Wrong! The goal (15) has turned into a hole!")
            if 15 not in self.holes:
                self.holes.append(15)
            self.goal = None

        self.env.unwrapped.s = self.last_state
        self.in_question_mode = False
        self.llm_question = None
        self.llm_answer = None
        return self.last_state, 0, False, False, {}

# ========================
# 动作选择：LLM 决策
# ========================
ACTIONS = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

def get_action_from_llm(obs, desc):
    map_str = "\n".join(["".join([cell.decode() for cell in row]) for row in desc])
    prompt = f"""
You are an AI agent in a 4x4 FrozenLake game.
Current state index: {obs}
Map (S=start, G=goal, H=hole, F=frozen):
{map_str}

Choose the safest next move to reach G (state 15).
Respond ONLY with: LEFT, DOWN, RIGHT, or UP.
"""
    text = call_qwen(prompt, max_tokens=10).upper()
    action_map = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}
    return action_map.get(text, random.choice([0,1,2,3]))

# ========================
# 轨迹可视化函数
# ========================
def plot_trajectory(states, desc, save_filename="trajectory.gif", edge_width=3, fps=2):
    """
    生成动画 GIF（不使用颜色填充，仅绘制格子边框并在格子中心标注 S/G/H/F，同时动态展示轨迹）。
    优先使用 imageio；若缺失则回退到 Pillow。
    针对不同 backend 的 canvas，优先尝试 tostring_rgb；若不支持则使用 tostring_argb 并转换为 RGB。
    """
    # 尝试导入 imageio 或 Pillow（PIL）
    have_imageio = False
    have_pil = False
    try:
        import imageio
        have_imageio = True
    except Exception:
        try:
            from PIL import Image
            have_pil = True
        except Exception:
            have_pil = False

    if not (have_imageio or have_pil):
        raise ModuleNotFoundError(
            "Neither imageio nor Pillow (PIL) is installed. Install with:\n"
            "  python -m pip install imageio pillow"
        )

    results_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, save_filename)

    frames = []
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_frame(step_idx):
        ax.clear()
        ax.set_facecolor('white')
        for i in range(4):
            for j in range(4):
                rect = patches.Rectangle((j, i), 1, 1,
                                         linewidth=edge_width,
                                         edgecolor='k',
                                         facecolor='white',
                                         joinstyle='miter')
                ax.add_patch(rect)
                cell = desc[i][j].decode() if isinstance(desc[i][j], (bytes, bytearray)) else str(desc[i][j])
                label = cell if cell in ['S', 'G', 'H', 'F'] else cell.upper()[:1]
                ax.text(j + 0.5, (3 - i) + 0.5, label, fontsize=14, fontweight='bold',
                        ha='center', va='center', color='k')

        sub_states = states[:step_idx + 1]
        coords = [(s // 4, s % 4) for s in sub_states]
        xs = [c for r, c in coords]
        ys = [3 - r for r, c in coords]
        cx = [x + 0.5 for x in xs]
        cy = [y + 0.5 for y in ys]
        if len(cx) > 0:
            ax.plot(cx, cy, '-o', color='black', linewidth=2, markersize=8)
            ax.text(cx[0] - 0.25, cy[0], 'S', fontsize=12, fontweight='bold', va='center')

        g_pos = None
        for i2 in range(4):
            for j2 in range(4):
                cellv = desc[i2][j2]
                if (isinstance(cellv, (bytes, bytearray)) and cellv == b'G') or (str(cellv).upper().startswith('G')):
                    g_pos = (j2 + 0.5, (3 - i2) + 0.5)
                    break
            if g_pos:
                break
        if g_pos:
            ax.text(g_pos[0] - 0.25, g_pos[1], 'G', fontsize=12, fontweight='bold', va='center')

        if len(cx) > 0:
            cur_x, cur_y = cx[-1], cy[-1]
            ax.plot([cur_x], [cur_y], 'o', color='red', markersize=10)

        outer = patches.Rectangle((0, 0), 4, 4, linewidth=edge_width * 1.2, edgecolor='k', facecolor='none')
        ax.add_patch(outer)

        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Agent Trajectory in FrozenLake (labels only)")
        plt.tight_layout()

        # 将当前图转为 RGB ndarray，兼容不同 canvas backend
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()  # width, height

        try:
            # 大多数 Agg backend 支持 tostring_rgb
            buf = fig.canvas.tostring_rgb()
            img = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
        except Exception:
            # 部分 backend（如 TkAgg）提供 tostring_argb：需要转换 ARGB -> RGB
            buf = fig.canvas.tostring_argb()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
            # arr format: A,R,G,B -> 取 R,G,B
            img = arr[:, :, [1, 2, 3]].copy()

        return img

    for t in range(len(states)):
        img = draw_frame(t)
        frames.append(img)

    # 保存 GIF：优先 imageio，否则使用 Pillow
    if have_imageio:
        import imageio
        imageio.mimsave(save_path, frames, fps=fps)
    else:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000 / fps) if fps > 0 else 500
        pil_frames[0].save(save_path, save_all=True, append_images=pil_frames[1:], duration=duration_ms, loop=0)

    plt.close(fig)
    print(f"✅ Trajectory GIF saved to {save_path}")

# ========================
# 主游戏循环
# ========================
def play_game():
    # 指定 render_mode，避免依赖 pygame（可改为 "human" 并安装 pygame）
    env = gym.make('FrozenLake-v1', is_slippery=True, render_mode="rgb_array")
    env = MathRescueWrapper(env)

    # gymnasium reset -> (obs, info)
    obs, info = env.reset()
    env.update_desc()
    print("🤖 欢迎来到由刘航程构建的<带有LLM救援的Frozen Lake游戏>!")
    print("规则如下:")
    print("- 到达状态<15>即获胜.")
    print("- 不小心掉入洞穴？只要正确回答问题就能拯救你的目标状态<15>！")
    print("- 回答正确: 洞穴消失，目标状态复原.")
    print("- 回答错误: 目标状态<15>也变成了一个洞穴!")
    print("="*50)

    trajectory = [obs]  # 记录轨迹
    done = False
    step_count = 0
    max_steps = 100  # 防止无限循环
    reward = 0.0

    while not done and step_count < max_steps:
        # rgb_array 模式下 render() 返回帧（ndarray），安全调用
        try:
            _frame = env.render()
        except Exception:
            _frame = None

        if env.in_question_mode:
            print("🧠 LLM is solving the rescue question...")
            # 注意：answer_question 返回 (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = env.answer_question()
            done = terminated or truncated
            if isinstance(obs, tuple):
                obs = obs[0]
            print(f"Back to state {obs}.")
        else:
            action = get_action_from_llm(obs, env.env.unwrapped.desc)
            print(f"State {obs} → LLM chose: {ACTIONS.get(action, action)}")
            # 注意：step 返回 (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if isinstance(obs, tuple):
                obs = obs[0]
            trajectory.append(obs)
            step_count += 1

    print(f"Game Over! Final reward: {reward}")
    
    # 可视化轨迹
    plot_trajectory(trajectory, env.env.unwrapped.desc)

# ========================
# 启动
# ========================
if __name__ == "__main__":
    play_game()
