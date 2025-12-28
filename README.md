# AI_works 汇总说明（First_work + Second_work）

项目概览
- 本仓库包含两份作业/实验工程：
  - First_work：轻量 Qwen 调用与示例（诗歌生成 demo）。
  - Second_work：基于 Gymnasium FrozenLake 的 LLM 驱动实验（MathRescue + 轨迹可视化）。

目录结构（重要子目录）
- First_work/
  - requirements.txt          # First_work 依赖（如有）
  - API/                      # API 密钥示例与文本文件（api_key_example.txt, qwen_api.txt）
  - code/my_qwen.py           # 第一份作业的主代码示例（调用 Qwen）
  - prompt/                   # prompt 示例
  - results/                  # 输出（如生成的诗）
- Second_work/
  - requirements.txt          # Second_work 依赖（gymnasium, numpy, requests, matplotlib, imageio, Pillow 等）
  - README.md                 # FrozenLake + LLM 的单独说明（可直接阅读）
  - experiments/frozenlake_LLM.py  # 演示脚本（可直接运行）
  - src/                      # 建议的模块化实现（api/, envs/, agent/, viz/）
    - main.py                   # 运行入口（使用 src 模块化代码）
  - results/                  # FrozenLake 轨迹 GIF 输出等

快速开始（Windows）
1. 安装依赖（Second_work 为示例）：
   python -m pip install -r Second_work\requirements.txt
   （First_work 若含 requirements.txt，请另行安装）
2. 设置 QWEN API Key（PowerShell 示例）：
   $env:QWEN_API_KEY="your_key_here"
3. 运行示例：
   - First_work 示例（如 poem 生成）：
     python First_work\code\my_qwen.py
   - Second_work FrozenLake 演示：
     python Second_work\experiments\frozenlake_LLM.py
   或使用模块化入口：
     python Second_work\src\main.py
4. 输出文件：
   - First_work/results/poem.txt
   - Second_work/results/trajectory.gif（轨迹动画）

注意事项
- API 调用可能产生成本：开发/调试时建议 mock 或用极低频率请求。
- Matplotlib 后端差异：若遇到 canvas 方法缺失，可在脚本顶部强制使用非交互后端：
  ```python
  import matplotlib
  matplotlib.use("Agg")
  ```
- 如果 Git push 被拒绝，先 pull/rebase 远端改动再推送（参见项目内 process.txt 或使用 git pull --rebase origin main）。
- .gitignore 中原先可能忽略了 Second_work/experiments；若需要将实验脚本纳入版本控制，请确保 .gitignore 已更新（见项目根 .gitignore）。

开发建议
- 推荐将 experiments 中的脚本拆分到 Second_work/src 下（api/qwen_api.py, envs/math_rescue.py, agent/llm_agent.py, viz/plotting.py），experiments 保留为演示用脚本。
- 在提交或推送前，先检查并运行单元或简单功能测试，避免无意中提交敏感密钥文件（API 密钥请放入环境变量或受控配置文件）。

作者与许可证
- 作者/组员信息见各实验文件头部。请根据课程要求保留署名信息。
- 本仓库用于课程作业演示与学习，使用时遵守相应第三方 API 服务条款。