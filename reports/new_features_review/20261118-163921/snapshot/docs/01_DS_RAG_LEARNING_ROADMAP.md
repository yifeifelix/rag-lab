# DS RAG 学习路线：从可解释基线到真实文档实验

本项目是学习实验室。先理解实现的简化，再引入真实模型和交付功能。

## 1. 跑通最小链路

执行 `scripts/workbench start`。关闭缓存，问一个有答案和一个无答案的问题。记录问题、召回证据、回答和拒答理由。

阅读 `data_gen.py → chunking.py → embeddings.py → retrieval.py → pipeline.py`。默认回答是抽取式，embedding 是 TF-IDF + SVD，不是预训练大模型。

练习：手动计算一条查询的 BM25、余弦相似度和 RRF 排名；说明它们的分数为什么不能直接比较。

## 2. 分块与证据

比较 fixed、recursive、semantic、parent_child。当前主要按字符切分，不能把字符数当 token 数。

查询页不支持动态切换分块。修改 `RAGConfig` 并重建 pipeline，或运行架构对照脚本。

练习：保持语料、embedding 和检索方式不变，只改变分块；检查证据是否保留回答所需的限制条件和表格信息。

## 3. 检索与重排

阅读 `retrieval.py`、`reranker.py`。区分 BM25、局部语料 SVD 向量、预训练 bi-encoder，以及人工特征重排和神经 cross-encoder。

执行 `scripts/workbench prepare-domain` 和 `scripts/workbench evaluate-domain --split dev`。这里四组受控比较保持分块和 embedding 一致。

练习：给出 BM25 比复杂管线更好的问题，分析复杂度是否值得。固定测试集只用于最终记录；当前标签仍需独立人工审核。

## 4. 回答质量和实验设计

阅读 `evaluation.py` 和逐题 `pending_human_review.jsonl`。检索命中文档不等于找到了正确证据，更不等于回答正确。

`lexical_grounding_proxy` 是词项覆盖代理指标，不能识别关系颠倒、数字误用和错误推理。拒答不参与其平均值。

练习：人工标记正确性、引用支持和应该拒答与否；找出高 Recall、低答案质量的样例。扩展到真实业务问题，并邀请独立评审者核对。

bootstrap CI 表达当前样本下的不确定性，不能解决样本偏差或测试集调参。现有 bootstrap nonpositive fraction 不当作校准后的显著性 p 值。

## 5. 缓存与请求控制

阅读 `cache.py`、`resilience.py` 和回归测试。观察 L1、L2 和 singleflight 的区别。

练习：相同问题重复提问，再改变编号或数值；验证不会复用不匹配的语义缓存。修改/删除文档，验证旧答案不再命中。

执行 `scripts/workbench synthetic --requests 10000`。明确这是单进程、模拟等待、构造流量实验，记录尾延迟，不把本地缓存中位数写成云端延迟。

## 6. 完成一个可展示交付闭环

使用 Documents 页签导入真实且获准使用的 Markdown/TXT，填写来源、版本。演示导入、提问、查来源、更新、删除、重启。

随后接入可选预训练 embedding 和真实 LLM，记录成本、回答质量、P95 和降级情况。真实模型未跑通时不能把抽取式回退包装为成功。

进一步交付私有数据时，再补用户级 ACL、权限范围缓存、数据保留策略、监控、备份恢复和验收。不要为了学习先搭建大规模 Kubernetes 集群。
