# 仅用训练分区学习 byte-level BPE Tokenizer，负责保存、加载、编码与解码。
# 预先分配特殊 Token 并冻结词表；检查编码回环、特殊 Token ID 和缓存兼容性。
