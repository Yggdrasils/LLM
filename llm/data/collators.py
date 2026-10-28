# 将各阶段样本组织成批次，构造 padding、attention mask 和 loss mask。
# 统一完成自回归输入与目标的单次移位；保留回答边界，剔除没有有效回答 Token 的样本。
