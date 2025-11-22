import numpy as np
from gymnasium import Wrapper
from api.qwen_api import call_qwen

class MathRescueWrapper(Wrapper):
    """
    把你原来的 MathRescueWrapper 放在此模块，保持逻辑不变。
    """
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
        try:
            last = getattr(self.env.unwrapped, "s", None)
            self.last_state = int(last) if last is not None else self.last_state
        except Exception:
            pass

        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, tuple):
            state = int(obs[0])
        else:
            try:
                state = int(obs)
            except Exception:
                state = obs

        if not self.in_question_mode:
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
            return obs, reward, terminated, truncated, info
        else:
            return self.last_state, 0.0, False, False, {}

    def ask_math_question(self):
        prompt = """
Generate a simple math or logic question with a single numerical answer.
Format:
Question: <your question>
Answer: <number>
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
        if not self.in_question_mode:
            return self.last_state, 0, False, False, {}

        answer_prompt = f"Question: {self.llm_question}\nGive only the numerical answer."
        pred_text = call_qwen(answer_prompt, max_tokens=10)

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