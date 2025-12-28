# 🚀 AI_works 汇总说明（First_work + Second_work + Project_work）

## 📋 项目概览
本仓库汇聚了三份精彩的作业/实验工程，涵盖AI应用的不同领域：

- **🎨 First_work**：轻量 Qwen 调用与示例（诗歌生成 demo）——探索AI创作的无限可能！
- **🏔️ Second_work**：基于 Gymnasium FrozenLake 的 LLM 驱动实验（MathRescue + 轨迹可视化）——智能决策与可视化之旅。
- **🗺️ Project_work**：中国旅行规划项目（基于LLM的旅行代理和环境模拟）——AI助力智能旅行规划。

## 📁 目录结构（重要子目录）

| 项目 | 子目录 | 描述 |
|------|--------|------|
| **First_work/** | requirements.txt | First_work 依赖（如有） |
| | API/ | API 密钥示例与文本文件（api_key_example.txt, qwen_api.txt） |
| | code/my_qwen.py | 第一份作业的主代码示例（调用 Qwen） |
| | prompt/ | prompt 示例 |
| | results/ | 输出（如生成的诗） |
| **Second_work/** | requirements.txt | Second_work 依赖（gymnasium, numpy, requests, matplotlib, imageio, Pillow 等） |
| | README.md | FrozenLake + LLM 的单独说明（可直接阅读） |
| | experiments/frozenlake_LLM.py | 演示脚本（可直接运行） |
| | src/ | 建议的模块化实现（api/, envs/, agent/, viz/）<br>&nbsp;&nbsp;- main.py | 运行入口（使用 src 模块化代码） |
| | results/ | FrozenLake 轨迹 GIF 输出等 |
| **Project_work/** | ChinaTravel/ | 主要项目文件夹 |
| | &nbsp;&nbsp;- requirements.txt | 项目依赖 |
| | &nbsp;&nbsp;- run_exp.py | 实验运行脚本 |
| | &nbsp;&nbsp;- eval_exp.py | 评估脚本 |
| | &nbsp;&nbsp;- chinatravel/ | 核心模块 |
| | &nbsp;&nbsp;&nbsp;&nbsp;- agent/ | 代理相关代码（包括LLM代理、验证器等） |
| | &nbsp;&nbsp;&nbsp;&nbsp;- data/ | 数据加载模块 |
| | &nbsp;&nbsp;&nbsp;&nbsp;- environment/ | 环境模拟（世界环境、数据库、工具） |
| | &nbsp;&nbsp;&nbsp;&nbsp;- evaluation/ | 评估工具（约束检查、偏好评估等） |
| | &nbsp;&nbsp;- cache/ | 缓存文件（LLM响应等） |
| | &nbsp;&nbsp;- results/ | 实验结果输出 |
| | &nbsp;&nbsp;- logs/ | 日志文件 |
| | &nbsp;&nbsp;- TPC@AIC2025/ | 相关竞赛或数据集 |

## ⚡ 快速开始（Windows）
1. **📦 安装依赖**（以 Second_work 为例）：
   ```bash
   python -m pip install -r Second_work\requirements.txt
   ```
   *（First_work 若含 requirements.txt，请另行安装）*

2. **🔑 设置 QWEN API Key**（PowerShell 示例）：
   ```powershell
   $env:QWEN_API_KEY="your_key_here"
   ```

3. **▶️ 运行示例**：
   - **First_work 示例**（如诗生成）：
     ```bash
     python First_work\code\my_qwen.py
     ```
   - **Second_work FrozenLake 演示**：
     ```bash
     python Second_work\experiments\frozenlake_LLM.py
     ```
     或使用模块化入口：
     ```bash
     python Second_work\src\main.py
     ```
   - **Project_work 中国旅行实验**：
     ```bash
     cd Project_work\ChinaTravel
     python run_exp.py  # 运行实验（需先安装依赖）
     ```

4. **📄 输出文件**：
   - First_work/results/poem.txt
   - Second_work/results/trajectory.gif（轨迹动画）
   - Project_work/results/（实验结果和日志文件）

---

## ⚠️ 注意事项
- 💰 **API 调用成本**：开发/调试时建议使用 mock 或低频率请求，避免不必要的费用。
- 🎨 **Matplotlib 后端差异**：若遇到 canvas 方法缺失，可在脚本顶部添加：
  ```python
  import matplotlib
  matplotlib.use("Agg")
  ```
- 🔄 **Git 操作**：若 push 被拒绝，先 pull/rebase 远端改动（参见项目内 process.txt 或使用 `git pull --rebase origin main`）。
- 🚫 **.gitignore 更新**：若需纳入 Second_work/experiments，请确保 .gitignore 已更新。

## 💡 开发建议
- 🏗️ **模块化改进**：推荐将 experiments 中的脚本拆分到 Second_work/src 下（api/qwen_api.py, envs/math_rescue.py, agent/llm_agent.py, viz/plotting.py），保留 experiments 为演示用。
- 🔒 **安全提交**：推送前检查并运行测试，避免提交敏感密钥（API 密钥请放入环境变量或受控配置文件）。

## 👥 作者与许可证
- 📝 **作者/组员信息**：见各实验文件头部。请根据课程要求保留署名。
- 📜 **许可证**：本仓库用于课程作业演示与学习，使用时遵守相应第三方 API 服务条款。

---

*🌟 祝您在AI探索之旅中收获满满！如有问题，欢迎交流。*