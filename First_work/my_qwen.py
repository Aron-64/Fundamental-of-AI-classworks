import os
from openai import OpenAI

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
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")