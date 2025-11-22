import random
from api.qwen_api import call_qwen

ACTIONS = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

def get_action_from_llm(obs, desc):
    """
    根据当前观测和地图让 LLM 选择下一个动作（返回动作索引 0-3）。
    desc 为 4x4 字节矩阵（与 gym FrozenLake desc 对齐）。
    """
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

def ask_answer_from_llm(question):
    prompt = f"Question: {question}\nGive only the numerical answer."
    resp = call_qwen(prompt, max_tokens=10)
    try:
        return float(resp.strip())
    except Exception:
        return None