import os
from openai import OpenAI

# 设置工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    # 读取 API 密钥
    api_key_path = os.path.join("..", "API", "qwen_api.txt")
    with open(api_key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    # 读取 ../prompt/poem_prompt.txt 文件内容
    prompt_path = os.path.join("..", "prompt", "poem_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    client = OpenAI(
        api_key=api_key,
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

    # 保存诗句到 ../results/poem.txt
    results_dir = os.path.join("..", "results")
    os.makedirs(results_dir, exist_ok=True)
    poem_path = os.path.join(results_dir, "poem.txt")
    with open(poem_path, "w", encoding="utf-8") as f:
        f.write(poem)

    # 新增：读取诗歌内容作为文生图主题
    with open(poem_path, "r", encoding="utf-8") as f:
        poem_theme = f.read()

    # 假设有 images.generate 接口，实际请参考你的 API 文档
    image_response = client.images.generate(
        prompt=poem_theme,
        n=1,
        size="512x512"
    )
    image_url = image_response.data[0].url

    # 下载图片并保存到 ../results/poem_image.png
    import requests
    img_data = requests.get(image_url).content
    image_path = os.path.join(results_dir, "poem_image.png")
    with open(image_path, "wb") as handler:
        handler.write(img_data)

except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")