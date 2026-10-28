# LLM：从零训练 TinyStories 小型语言模型

面向单台笔记本的语言模型学习项目：使用 TinyStories，从自定义 Tokenizer 和随机初始化 Transformer 开始，逐步实现预训练、监督微调、偏好优化、评估和本地推理。

**当前状态：仅完成软件架构与目录骨架，尚无算法实现。** 所有 Python 文件都只有文件头用途注释，没有导入、函数、类或可执行语句。YAML 文件也是注释占位；运行这些 Python 文件不会执行训练。

请先阅读根目录的 [ARCHITECTURE.md](ARCHITECTURE.md)，其中包含模块职责、阶段依赖、数据约定、硬件预算和逐步实现顺序。

| 项目 | 设计基线 |
| --- | --- |
| 数据 | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)，英文短故事 |
| 硬件 | i9-14900HX、RTX 4070 Laptop 8GB 专用显存、32GB 内存 |
| 模型 | 随机初始化的 Decoder-only Transformer，约 1750 万参数 |
| 首轮预算 | 50M 有效预训练目标 Token，最大上下文 512 |
| 优先路线 | 数据 → Tokenizer → 预训练 → SFT → DPO → 评估与导出 |
| 后续实验 | 奖励模型＋PPO；可验证奖励与专项训练 |

模型规模、批量和训练预算将在实现后通过本机短时基准调整。本项目的目标是理解完整训练过程，并观察英文短故事生成行为。

后续从 `scripts/prepare_data.py` 和 `llm/data/` 开始实现。原始数据、模型权重、训练日志和导出结果存放在已预留目录中，由 `.gitignore` 排除；仓库保留代码、配置和文档。
