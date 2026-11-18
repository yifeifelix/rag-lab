# Data Scientist 企业级 RAG 实战学习路线图 (6-Phase Mastery Roadmap)

本指南专为 **Data Scientist（数据科学家）** 设计，将本仓库 `/Volumes/Data/Projects/Rag` 作为**可运行、可实验、可量化验证**的工程实验室，帮你系统完成从 RAG 基础原理 → 检索与重排数学 → 统计学评估 → 10,000 并发高可用架构 → 求职面试与 Freelance 交付的全链路突破。

---

## 学习前置检查（确保所有环境与缓存留在 Data 盘）

```bash
cd /Volumes/Data/Projects/Rag
source .venv/bin/activate
# 验证语料与 Golden Benchmark 数据集
python -m rag_platform.data_gen
# 启动交互式 Web Workbench（浏览器访问 http://localhost:8000）
uvicorn rag_platform.server:app --host 0.0.0.0 --port 8000
```

---

## Phase 1：企业级知识库设计、分块策略与多域语料工程（Week 1）

### 核心问题
为什么把 PDF/Markdown 简单切分为固定 400 字符会导致严重的检索失败？
1. **语义断裂与表格撕裂**：固定滑动窗口会在表格行中间、公式中间强行切断。
2. **上下文稀释 vs 检索精度矛盾（Precision-Context Tradeoff）**：大块（900+ 字符）包含太多无关话题，导致向量均值稀释；小块（260 字符）检索极其精准，却丢失了前置条件与完整上下文。

### 本项目源码精读与动手实验
- 阅读 [`src/rag_platform/data_gen.py`](../src/rag_platform/data_gen.py)：观察 65 篇跨 5 个企业域（`cloud_architecture`, `security_compliance`, `ml_operations`, `financial_risk`, `sla_governance`）的文档结构。
- 阅读 [`src/rag_platform/chunking.py`](../src/rag_platform/chunking.py) 中的 4 种分块器：
  1. `FixedWindowChunker`（`fixed_chunk_size=260, fixed_overlap=40`）
  2. `RecursiveMarkdownChunker`（按 `##` 标题与段落递归切分）
  3. `SemanticBreakpointChunker`（计算相邻句子余弦距离并在第 80 百分位处切分）
  4. `ParentChildChunker`（索引 260 字符子节点用于精确匹配，返回 900 字符父节点上下文用于生成）

### 实操任务
- 在 Web UI 的 **Tab 1 (RAG Playground)** 测试同一条 `multi_hop` 问题，对比不同 `chunking_strategy` 的检索内容完整度。
- 尝试修改 `config.py` 中的 `parent_max_chars` 与 `child_max_chars`，观察 `scripts/run_ablation_study.py` 输出的 `Chunks` 数量与 `Recall@5` 变化。

---

## Phase 2：混合检索（Dense + Okapi BM25）、RRF 融合与跨特征重排（Week 2）

### 核心数学直觉（面试高频考点）
1. **为什么纯 Dense Bi-Encoder 会在企业检索中翻车？**
   - Bi-Encoder 将文本压缩进低维稠密空间，擅长捕捉同义改写（例如 *"8-bit integers"* ↔ *"INT8 scalar quantization"*），但对于生僻词（OOV）、错误码（`ERR_QDRANT_5032`）、合规条款（`SOC2-CC6.7`、`PCI-DSS-3.5.1`）很容易产生表示模糊。
2. **Okapi BM25 公式与工程调优**
   - 源码位于 [`src/rag_platform/retrieval.py`](../src/rag_platform/retrieval.py) 中的 `BM25Index`（使用 `scipy.sparse.csr_matrix` 向量化实现）：
   - `k1 = 1.5` 控制词频饱和度（防止高频词刷榜），`b = 0.75` 控制文档长度惩罚。
   - **关键细节**：`tokenize_technical()` 保留连字符/下划线/小数点复合编号，防止 `SOC2-CC6.7` 被切碎。
3. **Reciprocal Rank Fusion (RRF)**
   $$\text{RRF}(d) = w_{\text{dense}} \frac{1}{k + r_{\text{dense}}(d)} + w_{\text{sparse}} \frac{1}{k + r_{\text{sparse}}(d)}, \quad k = 60$$
   - **DS 必懂原因**：Dense 余弦相似度（0.2~0.8）与 BM25 分数（0~30+）量纲完全不同，直接线性加权对极值极其敏感；RRF 仅依赖名次（rank），无需标定分布即可稳定融合两路信号。
4. **Two-Stage Reranker（[`src/rag_platform/reranker.py`](../src/rag_platform/reranker.py)）**
   - `CrossFeatureReranker` 对 Top-36 候选进行精细化交叉特征打分：RRF 融合信号 + Dense 分数 + BM25 相对强度 + 词元与词根覆盖率 + 标题匹配率 + Bigram 邻近度 + 精确编号奖励，并通过 `deduplicate_and_select(..., max_per_doc=2)` 保证多跳问题不被单一文档刷屏。

---

## Phase 3：OOV 幻觉防线、安全护栏与置信度标定（Week 3）

### 真实踩坑复盘（可直接当作面试 STAR 故事！）
在本平台开发过程中，我们发现了极具代表性的 **OOV 单向量坍缩陷阱（Out-of-Vocabulary Cosine Collapse）**：
- 当用户问一个完全超出企业知识库的问题（例如 *"traditional French croissant lamination butter-to-dough ratio recipe"*）时，只有 `"ratio"` 一个词在语料词表中！L2 归一化会把这个 1 维非零向量放大到长度为 1，导致它和金融风控文档 `"Liquidity Coverage Ratio"` 产生高达 **`0.9315`** 的虚假余弦相似度！
- **解决方案**：在 `retrieval.py` 和 `reranker.py` 中引入 **词表覆盖率（`in_vocab_ratio`）与片段词元命中阻尼因子（`grounding_factor`）**，将不可回答问题的 `Refusal Accuracy` 从最初的 `16.7%` 直接提升至 **`100.0%` (`6/6`)**。

### 安全护栏（[`src/rag_platform/pipeline.py`](../src/rag_platform/pipeline.py)）
- **PII 自动脱敏**：在检索、日志与缓存之前剥离 SSN、银行卡 PAN、邮箱、Bearer Token。
- **Prompt Injection 拦截**：对越狱/套取 KMS 密钥类请求直接 Fail-Closed 返回 `ERR_GUARD_4007`。

---

## Phase 4：Data Scientist 级别的统计学评估体系（Week 4）

不要在简历或汇报中只写一句“Accuracy 提高了”！阅读 [`src/rag_platform/evaluation.py`](../src/rag_platform/evaluation.py)：
1. **将检索评估与生成评估解耦**：
   - 检索指标：`Recall@3`, `Recall@5`, `MRR`（首条命中倒数排名）, `NDCG@5`, `ContextPrecision@5`（RAGAS 位置加权精准度）。
   - 生成指标：`Answer Faithfulness`（回答词元在召回父上下文中的蕴含比例）、`Keyword Coverage`、`Citation Precision`、`Refusal Accuracy`。
2. **40 道分层 Golden QA 基准（`data/eval/golden_qa.jsonl`）**：
   - `exact_lookup` (12题) · `conceptual_semantic` (12题) · `multi_hop` (10题) · `unanswerable_guardrail` (6题)
3. **非参数 Bootstrap 置信区间与配对检验**：
   - `bootstrap_ci(..., n_boot=500)` 给出 95% 置信区间；
   - `paired_bootstrap_delta` 证明旗舰架构比 Naive Baseline 在 `Recall@5` 提升 `+0.0735 (p = 0.0480)`，在 `NDCG@5` 提升 `+0.0576 (p = 0.0460)`，统计显著！

---

## Phase 5：10,000 并发系统架构与性能压测（Week 5）

阅读 [`docs/02_ARCHITECTURE_10K_CONCURRENCY.md`](./02_ARCHITECTURE_10K_CONCURRENCY.md) 与 [`src/rag_platform/benchmark.py`](../src/rag_platform/benchmark.py)：
- 理解 **Little's Law（利特尔法则）** $L = \lambda \cdot W$ 如何指导缓存与并发槽位规划。
- 掌握三层防线：**L1 Exact Cache** → **L2 Semantic Vector Cache ($\cos \ge 0.90$)** → **Singleflight 请求合并** → **Token Bucket 限流 + Async Semaphore 背压 + Tri-State Circuit Breaker 熔断降级**。
- 运行 `python scripts/run_10k_benchmark.py --requests 10000`，并在 Web UI **Tab 3** 实时压测观察 P50/P95/P99。

---

## Phase 6：云原生部署、面试表达与 Freelance 商业化（Week 6）

- 学习 [`deploy/docker-compose.yml`](../deploy/docker-compose.yml)、[`deploy/k8s-production-10k.yaml`](../deploy/k8s-production-10k.yaml) 与 [`deploy/locustfile.py`](../deploy/locustfile.py)。
- 结合 [`docs/03_INTERVIEW_AND_FREELANCE_PLAYBOOK.md`](./03_INTERVIEW_AND_FREELANCE_PLAYBOOK.md) 进行模拟面试与客户提案演练；尝试切换 `.env` 中的 `RAG_LLM_MODE=openai` 或 `gemini` 跑通真实云端大模型端到端链路。
