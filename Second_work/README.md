# FrozenLake + LLM（frozenlake_LLM.py）说明

简介
- 该脚本实现了一个基于 OpenAI 风格大模型（Qwen）控制的 FrozenLake 游戏变体。
- 在玩家/智能体掉入指定洞（holes）时，环境会向 LLM 提出一道简单数学/逻辑题，LLM 回答后判断真伪来决定是否“拯救”目标状态（state 15）。
- 可生成不使用颜色填充、在格子中心标注 S/G/H/F 的轨迹 GIF。

主要功能
- MathRescueWrapper：对原始 FrozenLake 环境的包装，管理洞/目标、生成题目并触发问答流程。
- LLM 动作选择：通过向 Qwen 请求文本，返回动作（LEFT/DOWN/RIGHT/UP）。
- 问题生成与判定：使用 Qwen 生成问题、回答并使用 Qwen 判定回答是否正确（示范多次 API 调用）。
- 可视化：生成仅边框与文本标注（S/G/H/F）的轨迹动画 GIF，兼容 imageio 或 Pillow，并兼容不同 Matplotlib canvas（tostring_rgb/tostring_argb）。

项目文件映射（Second_work 目录结构说明）
- experiments/frozenlake_LLM.py
  - 当前脚本，包含完整的实验逻辑与可视化。适于快速运行与调试。
- src/api/qwen_api.py
  - 建议位置：将 Qwen 调用逻辑封装到此模块，负责与 Qwen 接口交互、配置超时与错误处理。
- src/envs/math_rescue.py
  - 建议位置：把 MathRescueWrapper 的实现迁移到此模块，便于复用与测试。
- src/agent/llm_agent.py
  - 建议位置：把 get_action_from_llm 等动作决策逻辑放入此模块，负责构建 prompt、解析动作。
- src/viz/plotting.py
  - 建议位置：把 plot_trajectory 的实现放入该模块，负责所有可视化相关代码。
- results/
  - 输出目录。轨迹 GIF 会保存为 results/trajectory.gif（脚本中路径：相对于 experiments 文件夹的 ../results）。
- requirements.txt
  - 项目依赖（见下方）。在运行前通过 pip 安装。

依赖（已写入 Second_work/requirements.txt）
- gymnasium>=0.28.1,<1.0
- numpy>=1.22.0,<2.0
- requests>=2.28.0
- matplotlib>=3.5.0
- imageio>=2.28.0
- Pillow>=9.4.0

运行说明（Windows）
1. 安装依赖：
   python -m pip install -r requirements.txt
2. 配置 Qwen API Key（PowerShell 示例）：
   $env:QWEN_API_KEY="your_key_here"
3. 运行脚本：
   python experiments\frozenlake_LLM.py
4. 运行结果：
   - 控制台显示游戏过程与 LLM 的问题/判定信息。
   - 轨迹 GIF 保存到 results 文件夹（脚本运行目录的 ../results 下），文件名默认 trajectory.gif。

注意事项
- Qwen API 调用可能产生费用并受配额限制；调试时可用本地回退逻辑或 mock 返回减少调用次数。
- Matplotlib 后端差异：脚本已兼容大多数后端（tostring_rgb / tostring_argb）。若遇到显示/保存问题，可在脚本顶部强制非交互后端：
  ```python
  import matplotlib
  matplotlib.use("Agg")
  ```
- 若缺少 imageio 或 Pillow，脚本会提示安装相应包并可回退到 Pillow 保存 GIF。

建议的代码组织（短）
- 将 experiments 中脚本按功能拆分到 src/ 下对应模块（api, envs, agent, viz），保持 experiments 用途为“实验/演示”，src 为可复用模块。

作者
Aron-64(LiuHangcheng_0318)
