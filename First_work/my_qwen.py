import os
from openai import OpenAI

# 设置工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    # 读取 prompt.txt 文件内容
    with open("prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    client = OpenAI(
        api_key="sk-0b6427348d234443aac3e8c69ed6bc9f",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {'role': 'system', 'content': 'You are a genius poet.'},
            {'role': 'user', 'content': prompt}
        ]
    )
    poem = completion.choices[0].message.content
    print(poem)

    # 新增：保存诗句到 results/poem.txt
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "poem.txt"), "w", encoding="utf-8") as f:
        f.write(poem)

except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")