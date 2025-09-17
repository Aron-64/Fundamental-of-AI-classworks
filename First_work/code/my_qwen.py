import os
from openai import OpenAI
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis

# 设置工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    # 读取 API 密钥
    api_key_path = os.path.join("..", "API", "qwen_api.txt")
    with open(api_key_path, "r", encoding="utf-8") as f:
        qwen_api_key = f.read().strip()

    # 读取 ../prompt/poem_prompt.txt 文件内容
    poem_prompt_path = os.path.join("..", "prompt", "poem_prompt.txt")
    with open(poem_prompt_path, "r", encoding="utf-8") as f:
        poem_prompt = f.read()

    print(f"正在生成古诗中...")
    client = OpenAI(
        api_key=qwen_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {'role': 'system', 'content': 'You are a genius poet.'},
            {'role': 'user', 'content': poem_prompt}
        ]
    )
    poem = completion.choices[0].message.content
    print("生成古诗:\n" + poem)

    # 保存诗句到 ../results/poem.txt
    results_dir = os.path.join("..", "results")
    os.makedirs(results_dir, exist_ok=True)
    poem_path = os.path.join(results_dir, "poem.txt")
    with open(poem_path, "w", encoding="utf-8") as f:
        f.write(poem)
    
    print("正在根据古诗生成配图中...")

    # 新增：从 results/poem.txt 读取诗歌内容作为图片生成主题
    with open(poem_path, "r", encoding="utf-8") as f:
        poem_for_image = f.read()

    image_prompt = f"中国古典画风格，意境优美，描绘以下诗句场景：\n{poem_for_image}\n画面富有诗意，留白恰当，艺术感强。"

    rsp = ImageSynthesis.call(
        api_key=qwen_api_key,
        model="qwen-image",
        prompt=image_prompt,
        n=1,
        size='928*1664',      # 推荐标准尺寸
        prompt_extend=True,    # 自动扩写提示词，提升画面质量
        watermark=False        # 无水印（如商用请确认授权）
    )

    if rsp.status_code == HTTPStatus.OK:
        print("图片生成成功,正在下载图片...")
        for i, result in enumerate(rsp.output.results):
            url_path = unquote(urlparse(result.url).path)
            file_name = PurePosixPath(url_path).name
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_name = "poem_image.png"  # 默认命名

            image_path = os.path.join(results_dir, file_name)
            img_data = requests.get(result.url).content
            with open(image_path, 'wb') as f:
                f.write(img_data)
            print(f"图片已保存: {image_path}")
    else:
        print(f'图片生成失败: {rsp.status_code} - {rsp.code} - {rsp.message}')


except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")