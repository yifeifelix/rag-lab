# 用户审计意见复查

日期：2026-11-18。对象：14:46 UTC 复现时的源码快照，具体版本以 SHA-256 为准。范围：源码检查、隔离数据复现、Mock HTTP 请求检查和当时的现有测试。**本轮没有修改业务实现、生产语料或上一轮审计结果。**

收尾核对发现，其他进程随后修改了 `config.py`、`evaluation.py`、`pipeline.py`、`server.py`，加入 CRAG/Graph RAG 等可选功能。本轮没有覆盖这些外部改动。新代码中仍能看到本报告指出的 PAN、入库改写、同步检索、URL key、逐请求 client 和宽泛异常回退等路径；但不能把本次 24 项测试通过视为后来新增功能的完整验收。

## 结论

审计方向和主要优先级成立。确认了原文被不可逆改写、同步计算阻塞事件循环、重建锁阻塞读取、远程调用缺少错误可观测性，以及评测区分度不足等问题。

上一轮把查询脱敏函数复用于入库，是实施上的不当选择。修复“原始查询进入缓存”不代表 PII 识别正确，更不能成为修改原始资料的依据。上一轮报告的“优先级 0 已落实”不能理解为这些新发现已解决。

本轮完整运行原有测试仍是 **24 passed in 4.97s**。这些测试没有覆盖订单号误伤、原文保真、短查询错误拒答和小写/点号引用等问题，因此通过率不能替代这些场景的验证。

## 本轮证据

- [复现脚本](../reports/recheck/2026-11-18/reproduce.py)：所有写入都在 Data 盘临时目录，结束后清理；远程模型全部使用假密钥与 MockTransport。
- [结构化结果](../reports/recheck/2026-11-18/evidence.json)：包含源码 SHA-256、逐项结果及复现执行期间的源码一致性校验。
- [精确源码快照](../reports/recheck/2026-11-18/reproduced_source/rag_platform/) / [快照来源](../reports/recheck/2026-11-18/snapshot_origin.json)：18 个文件均与复现哈希一致；主要从此前构建的 wheel 恢复到审计目录，没有覆盖工作目录源码。
- [并行变更记录](../reports/recheck/2026-11-18/concurrent_changes.json)：区分复现版本和随后外部修改的版本。
- [复现输出](../reports/recheck/2026-11-18/reproduce.log)。
- [测试输出](../reports/recheck/2026-11-18/pytest.log) / [JUnit XML](../reports/recheck/2026-11-18/pytest.xml)。
- [本轮文件清单](../reports/recheck/2026-11-18/manifest.json)：仅对应本轮资料，不覆盖上次审计封存的哈希清单。

## 高优先级问题

### 1. PII 误伤及原文改写：确认，属于数据完整性问题

位置：`pipeline.py:53`、`knowledge_base.py:95`。

| 输入 | 当前脱敏结果 | Luhn |
| --- | --- | --- |
| `Order 1234567890123` | `Order [REDACTED_PAN]` | 不通过 |
| `+1 415 555 0100 2345` | `+[REDACTED_PAN]` | 不通过 |
| `Order 1234567890128` | `Order [REDACTED_PAN]` | 通过，但这里仍是订单号 |

将前两个样例写入隔离知识库后，落盘正文确实已经变成脱敏结果。原始编号和电话不再存在于正文中。

修法需分两层：

1. 保留原始文档，不在正常导入流程中直接覆盖事实内容；如需脱敏索引或展示，使用明确标识的派生视图。
2. PAN 候选增加 Luhn、合理长度及上下文等判断。Luhn 只能筛掉部分误报，不能把通过校验的订单号认定为银行卡。

“仅查询和日志侧脱敏”还需加一个边界：如果原文含 PII，检索证据会进入远程 LLM 请求和前端响应。应分别配置这些外发边界的脱敏/授权策略，不能认为查询已脱敏就意味着文档内容不会外发。

已被覆盖成 `[REDACTED_PAN]` 的原文不能靠新校验恢复；如已有真实资料受影响，应从原始来源重新导入，不猜测恢复数字。

### 2. 同步 CPU 和重建锁：确认，但要区分阻塞位置

位置：`pipeline.py:414`、`:426`、`:661`；`knowledge_base.py:71`；`server.py:70`。

- 查询 embedding、检索、重排、抽取式生成等仍在 async 请求执行路径中直接运行同步代码。并发 semaphore 不会自动将它们移出事件循环。
- 在现有同步检索入口注入 80ms 的受控 CPU 计算后，5ms 心跳的最大间隔从约 6ms 上升到约 85ms，证明该入口没有执行隔离。**这不是现有数据规模下的真实延迟 benchmark。**
- `KnowledgeBase.current()` 已由 `asyncio.to_thread` 调用，因此它的文件读取/重建不能直接描述成“仍在 API 事件循环里同步执行”。
- 但它确实持有 `_lock` 和文件锁完成整库重建。复现中，重建未完成时其他 reader 不能拿到锁，即使旧管线仍可用。`/health` 也依赖 current，存在同样等待。

建议：计算任务进入有容量限制的执行器或独立推理 worker；构建新快照时保留旧快照可读，只在短暂的发布阶段交换指针，并校验版本，避免较早完成的旧任务覆盖新版本。

线程隔离优先解决事件循环响应性，不保证 CPU 吞吐线性增长。Python 的 GIL 和本地计算库行为仍需测量；重 CPU 场景可能需要进程或独立服务。[Python asyncio 文档](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread)

### 3. 远程 LLM：主要问题确认，不能把风险检查当作已证实的模型攻击

位置：`pipeline.py:332`、`:348`、`:375`、`:378`、`:482`；未接入的云 embedding 适配器也在 `embeddings.py:177` 使用 URL key。

Mock 请求证实：

- 合法用户问题没有触发注入正则，但检索文档中的恶意指令和伪造的 `</retrieved_context>` 分隔文本原样进入模型 prompt。
- Gemini key 在 query string 中，没有 `x-goog-api-key` 请求头。
- 两次不使用缓存的请求构造了两个 AsyncClient，不能跨请求复用该 client 的连接池。
- 人为制造上游 timeout 后，响应有 `degraded_mode=true` 和 `generation_mode=extractive_fallback`；但没有应用日志，也没有错误原因分类。

因此，“完全静默”需要限定：**用户可见降级标志，运营侧仍看不到上游错误原因**。Uvicorn 默认访问日志不等于应用的模型错误日志。

这里确认的是不可信文档进入了指令边界，未调用真实 LLM，不能声称已经验证某个模型必然服从攻击，也没有证明可读取服务器密钥。[OWASP 对间接提示注入的定义](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)

建议合并处理：

1. 按供应商接口使用正式系统指令通道，把文档明确作为不可信数据，并进行来源/权限和输出引用验证。分隔标签或几条正则不能保证彻底阻断间接注入。
2. Gemini 使用 `x-goog-api-key`，请求和异常日志同时去除 query、认证头和敏感正文。**先处理 URL key，再增加异常日志**，避免把带 key 的异常 URL 写入日志。[官方 key 使用说明](https://aistudio.google.com/docs/api-key)
3. 按应用及事件循环生命周期管理共享 AsyncClient，关闭时显式释放。后台实验当前另起事件循环，不应不加区分地复用全局 async client。[HTTPX 连接池建议](https://www.python-httpx.org/async/#opening-and-closing-clients)
4. 对 timeout、HTTP 状态错误、响应格式错误、配置错误分别记录安全的事件类型、请求 ID、耗时和降级原因；不要广泛吞掉所有编程错误。重试应限定条件、次数和总时限。

### 4. 评测天花板：方向确认，“全部指标为 1”需要纠正

真实资料为 9 篇文档，测试题 13 道，其中 11 道可回答、2 道不可回答。当前各架构 Recall@3 和 Recall@5 都是 1.000，确实无法用这些指标判断进一步的检索改进。

但不是全部指标都为 1：

| 架构 | Recall@5 | MRR | NDCG@5 | 关键词覆盖 | 文档级 Citation Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 1.000 | 1.0000 | 1.0000 | 0.3636 | 0.3788 |
| Dense | 1.000 | 0.9545 | 0.9664 | 0.3636 | 0.3788 |
| Hybrid | 1.000 | 1.0000 | 1.0000 | 0.3636 | 0.3788 |
| Hybrid + 特征重排 | 1.000 | 1.0000 | 1.0000 | 0.3182 | 0.3788 |

这些词项和文档级引用指标也不是人工正确率。现有评测同时存在“检索指标饱和”和“回答质量尚未验证”的问题。

扩充不应只是增加题数。应增加近似编号、同主题不同条件、过期版本、矛盾事实、语义改写、多文档共同支持、证据不足和短查询，并加入对应干扰文档。明确标注支持的证据片段，避免只要命中正确文档的任意片段就算成功。保留独立审核、冻结测试集和人工答案/引用核查。

## 中优先级及容易遗漏的细节

### 5. 魔法数和短查询：确认；“至少 3 个词”不是硬条件

位置：`retrieval.py:310`、`reranker.py:137`、`:147`。

实际公式含 `min(1, hits/3)`，一两个匹配词仍有非零分数。这是软性降权，不是必须命中三词才能回答。不过当前确实造成了错误拒答：

| 问题 | 第一条证据 | 分数 | 是否拒答 |
| --- | --- | ---: | --- |
| `Semaphore` | `PY-SEMAPHORE` | 0.2161 | 是 |
| `Queue` | `PY-QUEUE` | 0.1874 | 是 |
| `gather` | `PY-GATHER` | 0.1570 | 是 |
| `What does Semaphore do?` | `PY-SEMAPHORE` | 0.2068 | 是 |
| `How does an asyncio Semaphore manage its internal counter?` | `PY-SEMAPHORE` | 0.6176 | 否 |

阈值为 0.22。短问题已经检索到对应文档，仍被门控拒绝。这可以作为下一批修复的明确验收例，而不是简单把全局阈值调低。

权重、缩放和阈值应配置化，并在足够的开发集上校准。校准目标应包含可回答性、证据完整性、错误拒答和错误作答的代价；神经 reranker 的原始分数也不能直接当作正确概率。多跳题还要检查所有必要证据，不能只看最有信心的一篇文档。

### 6. 引用剥离：确认，可尽早修复

位置：`pipeline.py:131`。同一段内容的函数级复现：

```text
alpha beta gamma                  -> 1.0000
alpha beta gamma [DOC-V1-2]       -> 1.0000
alpha beta gamma [doc.v1.2]       -> 0.4286
```

合法 doc_id 允许小写和点号，当前剥离规则却不支持。应统一引用解析与 doc_id 规则，识别实际引用并单独报告未知引用；不要为了抬高代理指标把所有方括号内容都删除。不同分块策略的上下文是否包含 ID 会影响具体降幅，因此不能断言所有这类 ID 都固定降到 0.43。

### 7. 全量重建和元信息扫描：确认，Qdrant 前有一个依赖

位置：`knowledge_base.py:102`、`pipeline.py:263`、`:282`、`:455`。

- 每次变更构建完整 pipeline，重做分块、索引、BM25 等。`source` 和 `version` 为每个选中 chunk 分别扫描 documents，开销为 O(KN)，可用 doc_id 映射或索引 payload 解决。
- **TF-IDF 和 SVD 会随语料重新拟合**。增加一个文档可能改变所有旧文档的权重和投影；若仅将新向量 upsert 到 Qdrant，就可能混合不同的向量空间。
- 应先采用固定版本的预训练 encoder，再以稳定的文档/chunk 标识实现增量 upsert/delete、旧版本清理、元信息和缓存失效。切换模型或向量维度时仍需重建或迁移索引。[Qdrant point 更新与删除](https://qdrant.tech/documentation/manage-data/points/)
- Qdrant 不会自动解决当前 BM25 的全量 CSR 构建，也不会把 JSONL 文档存储变成事务系统。稀疏检索和文档存储需要各自的更新设计。

## 工程项复核

| 项目 | 结果 | 建议 |
| --- | --- | --- |
| Git | `git rev-parse` 失败，当前不是仓库 | 在实施修复前建立版本基线；本轮没有初始化或提交 |
| 日志 | 无应用 logging/logger；有 CLI 输出、报告和服务框架日志 | 增加结构化事件及敏感字段治理 |
| Lint | 当前没有 ruff/flake8 等配置；AST/JS 语法检查不等于 lint | 增加低噪声 lint 与 CI 检查 |
| 依赖锁定 | 没有依赖解析锁文件；环境版本快照不等于 resolver lock | 生成并提交锁文件，记录可选模型依赖与版本 |
| ignore 规则 | `.gitignore:13` 为 `*.lock` | 该规则也会忽略 `uv.lock`；应缩小到数据运行时锁文件 |
| import 副作用 | `config.py:11` 导入即读 `.env`、建目录、改环境及 tempfile | 移到显式启动/CLI 初始化，库导入只定义配置 |
| 路径 | `runtime.py` 硬性要求 `/Volumes/Data` | Data 盘是用户原有明确要求，应保留默认策略；将路径作为可配置数据根和启动检查，避免库导入硬绑定 |

## 学习路线与修复顺序

1. **先保护资料和修准可测行为**：原文保真、改进 PAN 判断、外发脱敏策略、引用解析、短查询回归测试；安全地处理 key 和错误日志。同时建立 Git、锁文件和 lint 基线。
2. **整理运行时边界**：有界计算执行器、后台构建与短暂快照发布、模型 client 生命周期、明确的错误分类及文档注入测试。
3. **建立能区分方案的英文评测**：更有挑战的资料和独立问题，冻结测试集，加入人工答案与证据审核。英语资料使用一个英语预训练 encoder 即可；先跑通已配置的 MiniLM，再决定是否比较 BGE，不假定换模型必然提升。[BGE 英文模型卡](https://huggingface.co/BAAI/bge-small-en-v1.5)
4. **做真实神经重排与开发集校准**：在相同候选集上比较当前特征重排与 cross-encoder，同时记录质量和 P95；固定模型后再校准接受/拒答策略。[Cross-Encoder 两阶段检索说明](https://sbert.net/examples/cross_encoder/applications/README.html)
5. **最后接入增量 Qdrant 路径**：使用固定 encoder 和版本策略，验证增、改、删、重启、重复任务、旧 chunk 清理及缓存失效。学习目标是可靠更新闭环，而不是先扩大并发宣传。

本次只完成复查与证据归档。上述问题仍需后续业务修复；本报告不把复现测试通过标记为缺陷已修复。

如需对相同版本再次复现，使用保存的源码快照，并将结果写入新目录：

```bash
RAG_PROJECT_ROOT="$PWD" \
PYTHONPATH="$PWD/reports/recheck/2026-11-18/reproduced_source" \
AUDIT_OUTPUT_DIR="$PWD/reports/recheck/replay-$(date +%Y%m%d-%H%M%S)" \
RAG_LLM_MODE=local RAG_EMBEDDING_BACKEND=local_svd \
TMPDIR="$PWD/.cache/tmp" PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" \
.venv/bin/python reports/recheck/2026-11-18/reproduce.py
```

该命令是隔离源码复现，不是针对不断变化的工作目录重新验收；它也不会重跑真实模型攻击。
