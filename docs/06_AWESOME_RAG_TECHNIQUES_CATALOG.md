# GitHub Awesome-RAG 技术借鉴与落地应用指南

本文整理了 GitHub 上最有代表性的 **Awesome-RAG 生态仓库**，并将其中最适合 **Data Scientist 求职作品集** 与 **企业 Freelance 交付** 的核心技术直接吸收落地到了本仓库 [`src/rag_platform/awesome_rag.py`](../src/rag_platform/awesome_rag.py) 中。

---

## 一、GitHub 值得精读的 Awesome-RAG 仓库清单

| 仓库 | 定位与特点 | 最值得借鉴的模块 |
| --- | --- | --- |
| **[NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)** | 目前 GitHub 最火的 RAG 技术实战 Notebook 合集（30+ 独立实验） | **Contextual Chunk Headers**、**CRAG (Corrective RAG)**、**HyDE**、**Contextual Compression**、**Self-RAG**、**RAPTOR** |
| **[Danielskry/Awesome-RAG](https://github.com/Danielskry/Awesome-RAG)** | 按架构演进分类（Naive → Advanced → Modular → Agentic RAG）的工程索引 | 模块化路由（Query Routing）与企业架构模式对照 |
| **[jxzhangjhu/Awesome-LLM-RAG](https://github.com/jxzhangjhu/Awesome-LLM-RAG)** | 偏论文与前沿方法的系统综述清单 | 检索增强评估基准、检索-生成协同训练思路 |
| **[DEEP-PolyU/Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG)** | 知识图谱 + RAG（GraphRAG）专题综述与项目索引 | 实体抽取、多跳子图传播与社区摘要 |
| **[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)** | 轻量级双层知识图谱 RAG（比微软 GraphRAG 快得多、成本低得多） | **低延迟实体共现图 + 向量双路融合检索** |
| **[infiniflow/ragflow](https://github.com/infiniflow/ragflow)** | 企业级开源 Deep Document Understanding RAG 引擎 | 复杂 PDF/表格版面解析（DeepDoc）、可控分块模板、引用高亮 |
| **[Marker-Inc-Korea/AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG)** | 面向 DS 的 AutoML 式 RAG 自动网格调优框架 | **基于 Golden QA 的参数自动寻优（AutoRAGTuner）** |
| **[explodinggradients/ragas](https://github.com/explodinggradients/ragas)** | 事实性无参考评估行业标准框架 | Context Precision@K、Faithfulness（NLI 蕴含验证）、Answer Relevance |

---

## 二、已落地整合进本项目的 5 项 Awesome-RAG 核心技术

所有新增实现位于 [`src/rag_platform/awesome_rag.py`](../src/rag_platform/awesome_rag.py)，并已接入 `EnterpriseRAGPipeline`（[`src/rag_platform/pipeline.py`](../src/rag_platform/pipeline.py)）、7 架构消融矩阵（[`src/rag_platform/evaluation.py`](../src/rag_platform/evaluation.py)）以及调优接口 `POST /api/v1/autorag/tune`（[`src/rag_platform/server.py`](../src/rag_platform/server.py)）：

1. **Contextual Chunk Headers（源自 Anthropic / `NirDiamant/RAG_Techniques`）**
   - 类名：`ContextualHeaderEnricher`
   - 原理：在块索引前注入 `[Doc: <doc_id> | Domain: <domain> | Title: <title>]` 上下文头，解决子块脱离标题后的语义指代不清问题。
2. **Domain Query Expansion & Pseudo-HyDE（源自 `NirDiamant/RAG_Techniques` / `Awesome-LLM-RAG`）**
   - 类名：`QueryExpanderHyDE`
   - 原理：针对同义改写与术语鸿沟（如 `"8-bit integers"` ↔ `"INT8 scalar quantization"`、`"vocabulary shifts"` ↔ `"terminology changes / PSI drift"`、`"statistically real vs noise"` ↔ `"bootstrap confidence intervals / paired hypothesis"`）补充假设文档词汇。
   - **关键安全约束**：只对已通过原始词表覆盖度校验（`orig_in_vocab >= 0.35`）且超过最低置信度的查询生效，且最终拒答判定仍锚定原始查询置信度，避免越界问题被扩展词误放行。
3. **LightRAG 实体知识图谱检索（源自 `HKUDS/LightRAG` / `Awesome-GraphRAG`）**
   - 类名：`LightRAGEntityGraph` + `fuse_trimodal_rrf`
   - 原理：从语料抽取技术实体、错误码、合规条目与核心概念，构建 1-hop 实体共现图；检索时同时激活直接匹配实体（权重 `1.0`）和 1-hop 邻接实体（权重 `0.28 * edge_w`），输出第 3 路图排名，通过 **Tri-Modal RRF（`Dense 0.45 + BM25 0.35 + Graph 0.20`）** 融合。
4. **CRAG 自纠错门控（Corrective RAG，源自 `NirDiamant/RAG_Techniques`）**
   - 类名：`CRAGEvaluator`
   - 原理：对首轮检索做出三态判别：
     - `CORRECT`（`conf >= 0.48`）：直接放行，免除多余扩展开销；
     - `AMBIGUOUS`（`0.22 <= conf < 0.48`）：自动触发 `QueryExpanderHyDE` + `LightRAGEntityGraph` 纠错重检（`crag_status = "AMBIGUOUS_CORRECTED"`）；
     - `INCORRECT`（`conf < 0.22` 或 `in_vocab_ratio < 0.35`）：直接拒答，杜绝幻觉。
5. **Contextual Compression & AutoRAG 超参数自动寻优（源自 `NirDiamant/RAG_Techniques` / `Marker-Inc-Korea/AutoRAG`）**
   - 类名：`ContextualCompressor`、`AutoRAGTuner`
   - 原理：`ContextualCompressor` 按句级相关性压缩父上下文以减少喂给 LLM 的 Token 消耗；`AutoRAGTuner.tune_rrf_weights` 自动网格搜索 `(dense_weight, sparse_weight, graph_weight, rrf_k)` 组合，以 `NDCG@5` 与 `MRR` 为目标选出最优参数（已暴露为 `POST /api/v1/autorag/tune`）。

---

## 三、实测消融提升数据（`reports/ablation_report.md`）

在 65 篇企业文档 + 40 道分层 Golden QA 基准上的实测对照（`python scripts/run_ablation_study.py`）：

| 架构配置 | Recall@5 | MRR | NDCG@5 | Context Prec@5 | Refusal Acc | 单次检索耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. `Fixed Chunk + Naive Dense`（基线） | 0.897 | 0.892 | 0.879 | 0.869 | 1.000 | 0.38 ms |
| 6. `Parent-Child + Hybrid RRF + Reranker` | 0.971 | 0.934 | 0.936 | 0.920 | 1.000 | 1.62 ms |
| **7. `Awesome-RAG: LightRAG + CRAG + Reranker`** | **1.000** | **1.000** | **0.992** | **0.968** | **1.000** | **2.67 ms** |

- 尤其是原本较弱的语义改写类题目（`conceptual_semantic`, 12题），由第 6 组的 `Recall@5=0.917 / MRR=0.812` 提升至 **`Recall@5=1.000 / MRR=1.000 / NDCG@5=1.000`**。

---

## 四、下一步可继续引入的外部工具（面向 Freelance 真实客户项目）

当接手包含**复杂扫描件 PDF、跨页合并单元格财报、多模态图表**的真实商业单子时，建议搭配以下成熟开源工具做前端解析，再喂入本平台的检索与评估管线：
- **复杂 PDF / 表格解析**：`infiniflow/ragflow`（DeepDoc）、`DS4SD/docling`（IBM 开源文档结构化转 Markdown/JSON）或 `VikParuchuri/marker`。
- **真实 LLM 裁判评估**：将 `ragas`（需配置 `OPENAI_API_KEY` / `GEMINI_API_KEY`）接在 [`reports/`](../reports) 导出的 `pending_human_review.jsonl` 之上，把目前的 `lexical_grounding_proxy` 升级为真正的 NLI 语义忠实度评估。
- **Agentic RAG / 路由编排**：若客户端需要多步 SQL 查询 + 向量检索 + 外部 API 混合调用，可在 [`EnterpriseRAGPipeline`](../src/rag_platform/pipeline.py) 前方加上工具路由层（借鉴 `Danielskry/Awesome-RAG` 中的 Modular/Agentic RAG 范式）。
