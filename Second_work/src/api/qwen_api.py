# 调用 Qwen 的轻量封装（不在导入时抛错）
import os
import requests

def call_qwen(prompt, max_tokens=50, timeout=10):
    """
    调用 Qwen 文本生成接口，返回纯文本响应。
    在环境变量 QWEN_API_KEY 或 DASHSCOPE_API_KEY 未设置时抛出明确错误。
    """
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("Please set QWEN_API_KEY or DASHSCOPE_API_KEY environment variable.")
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {api_key}",
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
        resp = requests.post(url, headers=headers, json=data, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        return result["output"]["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # 返回空字符串以便上层可用回退策略
        print(f"⚠️ Qwen API error: {e}")
        return ""