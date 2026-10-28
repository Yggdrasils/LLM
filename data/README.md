# 数据目录约定

当前只有目录占位，没有下载或生成任何数据。

| 目录 | 未来内容 |
| --- | --- |
| `raw/` | 固定 revision 的官方 TinyStories 数据及来源信息 |
| `processed/` | 轻量校验后的故事、训练/开发/测试分区与数据清单 |
| `tokens/` | 冻结 Tokenizer 生成的 Token 分片、边界索引与指纹 |
| `sft/` | 故事续写指令和目标回答 |
| `preferences/` | 带标注来源的 chosen/rejected 偏好对及 DPO 参考概率缓存 |
| `prompts/` | RL 在线采样提示、开发提示和隔离的最终评估提示 |

先按原始故事划分集合，再生成衍生样本；同一故事及其派生样本不得跨训练和评估分区。Tokenizer 只使用训练分区学习。

数据来源：[TinyStories 官方数据页](https://huggingface.co/datasets/roneneldan/TinyStories)。固定具体版本时记录上游许可和来源声明；代码仓库的许可选择不替代数据本身的许可。

数据字段、来源追踪和去重规则见 [整体架构](../ARCHITECTURE.md)。
