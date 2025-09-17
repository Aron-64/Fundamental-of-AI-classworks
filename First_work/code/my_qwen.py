import os
import requests
from openai import OpenAI
from dashscope import ImageSynthesis
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY_PATH = os.path.join(BASE_DIR, "..", "API", "qwen_api.txt")
PROMPT_PATH = os.path.join(BASE_DIR, "..", "prompt", "poem_prompt.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
POEM_PATH = os.path.join(RESULTS_DIR, "poem.txt")

def read_api_key(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def read_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_poem(poem, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(poem)

def generate_poem(api_key, prompt):
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
    return completion.choices[0].message.content

def generate_image(api_key, poem, results_dir):
    image_prompt = f"按照油画风格，意境优美，描绘以下诗句场景：\n{poem}\n画面富有诗意，留白恰当，艺术感强。"
    rsp = ImageSynthesis.call(
        api_key=api_key,
        model="qwen-image",
        prompt=image_prompt,
        n=1,
        size='928*1664',
        prompt_extend=True,
        watermark=False
    )
    if rsp.status_code == HTTPStatus.OK:
        print("图片生成成功,正在下载图片...")
        for result in rsp.output.results:
            print("图片链接：", result.url)
            url_path = unquote(urlparse(result.url).path)
            file_name = PurePosixPath(url_path).name
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_name = "poem_image.png"
            image_path = os.path.join(results_dir, file_name)
            img_data = requests.get(result.url).content
            with open(image_path, 'wb') as f:
                f.write(img_data)
            print(f"图片已保存: {image_path}")
    else:
        print(f'图片生成失败: {rsp.status_code} - {rsp.code} - {rsp.message}')
        print(rsp)

def main():
    try:
        api_key = read_api_key(API_KEY_PATH)
        prompt = read_prompt(PROMPT_PATH)
        print("正在生成古诗中...")
        poem = generate_poem(api_key, prompt)
        print("生成古诗:\n" + poem)
        save_poem(poem, POEM_PATH)
        print("正在根据古诗生成配图中...")
        poem_for_image = read_prompt(POEM_PATH)
        generate_image(api_key, poem_for_image, RESULTS_DIR)
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

if __name__ == "__main__":
    main()