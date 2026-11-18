# RAG Learning & Evaluation Workbench

面向 RAG 学习、Applied Data Scientist 求职作品集和小型 Freelance PoC 的本地实验项目。

当前定位：**可运行的单用户 RAG 工作台，有文档生命周期、可复现实验和审计记录**。不是经过生产验收的企业级万并发服务。

## 已实现

- 四类分块、TF-IDF + SVD 离线 embedding、BM25、RRF、人工特征重排。
- Awesome-RAG 进阶特性（[`src/rag_platform/awesome_rag.py`](src/rag_platform/awesome_rag.py)）：LightRAG 实体共现图三路 RRF、CRAG 自纠错门控、Pseudo-HyDE 查询扩展、上下文压缩与 AutoRAG 权重调优（`POST /api/v1/autorag/tune`），详见 [Awesome-RAG 技术借鉴指南](docs/06_AWESOME_RAG_TECHNIQUES_CATALOG.md)。
- 可选 Sentence Transformers 预训练 embedding，默认只读本地模型缓存；显式命令才允许下载。
- 本地抽取式回答，及可选 OpenAI/Gemini 生成适配器。响应明确标识模型、降级、token 用量和可选成本估算。
- L1/L2 缓存、singleflight、限流、并发门控、熔断。缓存属于单个 Python 进程。
- 文档导入、查询、更新、删除；原子保存 JSONL；JSON/NumPy 索引持久化；更新清空答案缓存；文档版本事件日志。
- 文档来源、版本、证据展示；Markdown/TXT 文件编辑器；单用户 owner token。
- 合成数据上的六组架构对照，真实 CPython 文档案例上的受控检索对照，以及待人工审核的逐题回答导出。

## 启动

```bash
cd /Volumes/Data/Projects/Rag
scripts/workbench start
# http://127.0.0.1:8000
```

已有 `.venv` 可直接使用。新环境需要 Python 3.12+ 和 uv：

```bash
mkdir -p .cache/tmp
UV_CACHE_DIR="$PWD/.cache/uv" TMPDIR="$PWD/.cache/tmp" uv venv .venv
scripts/workbench install dev
```

先创建 `.cache/tmp` 再创建新环境。可复制 `.env.example` 为 `.env`；Python 配置会自动读取，已导出的环境变量优先。不要提交真实密钥。

启动器在 Python 启动前设置 Data 盘上的临时目录、依赖和模型缓存。项目 `.venv`、数据、索引与报告位于 Data 盘；Docker 镜像/虚拟磁盘、浏览器缓存和操作系统本身的空间使用不在这项保证内。

## 文档演示

在第 5 个 Documents 页签导入 `.md` / `.txt`，填写文档 ID、标题、来源、版本和正文。保存后提问，再修改文档并复问，最后删除并复问，可观察答案与证据变化。

API：

- `GET /api/v1/documents`：列表和语料版本。
- `GET /api/v1/documents/{id}`：正文与来源。
- `POST /api/v1/documents`、`PUT /api/v1/documents/{id}`：导入/更新。
- `DELETE /api/v1/documents/{id}`：删除。
- 写入可携带 `If-Match: <corpus_revision>`；版本冲突返回 409。

更新先完成索引构建，再原子替换语料。新请求使用新版本；已在途的请求可以完成其旧版本快照。其他本地 worker 在下次请求时发现语料变化并刷新。此机制不等于跨机器分布式一致性。

默认可编辑语料位于 `data/workspaces/demo/`；工作区和 `reports/private/` 已加入 `.gitignore`。客户资料的评估输出应显式指定到私有报告目录，分享前检查是否含正文和证据。

默认仅允许 loopback 访问 API。配置 `RAG_ADMIN_TOKEN` 后，所有 `/api/v1` 请求需要 `Authorization: Bearer ...`，浏览器可在顶部输入 token。不建议把这个单用户工作台当作多租户产品公开部署；尚无用户级/文档级 ACL。

## 真实文档学习案例

```bash
scripts/workbench prepare-domain
scripts/workbench evaluate-domain --split dev
scripts/workbench evaluate-domain --split test
```

从本机 CPython 官方 asyncio docstrings 提取 9 篇真实资料，保留本机源码位置、Python 版本和 SHA-256。9 道开发题和 13 道固定测试题分别保存；**题目由代理编写，尚未经过独立人工审核**，不能作为已验证的泛化成绩。

固定评估资料放在 `data/python_support/`，可编辑演示副本放在 `data/workspaces/python_support/`，二者分离。使用演示副本启动：

```bash
RAG_CORPUS_PATH="$PWD/data/workspaces/python_support/documents.jsonl" \
RAG_QA_PATH="$PWD/data/workspaces/python_support/qa_dev.jsonl" \
scripts/workbench start
```

每次实验输出到 `reports/python_support/<embedding>/<llm>/<split>/`，包含配置/源码/数据哈希、逐题回答、证据、P50/P95/P99，以及 `pending_human_review.jsonl`。**优先阅读失败例子，不只看 Recall。**

## 可选真实模型

预训练 embedding 需要额外依赖、网络和模型下载：

```bash
scripts/workbench install dev,pretrained
scripts/workbench download-model
RAG_EMBEDDING_BACKEND=sentence_transformers scripts/workbench start
scripts/workbench evaluate-domain --embedding sentence_transformers --split dev
```

默认示例模型为英文 `sentence-transformers/all-MiniLM-L6-v2`。多语言能力取决于所选模型和独立评估，不能由接口支持推断。

远程生成在 `.env` 设置 `RAG_LLM_MODE=openai` 或 `gemini`，配置对应 API key 和账号可用的模型名称。首次验证应只跑少量问题；云端费用由所用服务产生。

`RAG_INPUT_USD_PER_MILLION` / `RAG_OUTPUT_USD_PER_MILLION` 可填运营者核实后的生成 token 单价；未配置时云端费用为 `null`，不会猜测价格。这个估算不含 embedding、托管或网络费用。缓存命中不重复计生成 token。

本次修改期间没有可用密钥，也无法在沙箱下载模型：**真实模型尚未实测**。接口契约使用 MockTransport 验证；云端失败会显示 `extractive_fallback`，不计作真实模型验证成功。

## 实验与验证

```bash
scripts/workbench test -q
scripts/workbench ablation
scripts/workbench synthetic --requests 10000
# 服务在独立终端启动后再运行真正的 HTTP 客户端：
scripts/workbench http-benchmark --requests 200 --concurrency 10
```

`synthetic` 是单进程 `asyncio.gather` 实验：固定流量比例、预热缓存、本地抽取回答和模拟等待；不是 HTTP、真实 LLM 或生产容量证明。吞吐、平均在途请求、用户数分别报告，不能互换。

`lexical_grounding_proxy` 只计算回答词项在证据中的覆盖率，不判断事实是否正确；拒答时为 `null`。它不是 RAGAS Faithfulness。bootstrap nonpositive fraction 是描述性重采样结果，不称作校准后的显著性 p 值。

SSE 当前是**完整回答生成后的分段回放**，响应头标注 `X-Streaming-Mode: buffered-response-replay`；没有声称模型首 token 流式延迟。

## 审计与边界

- [首次审查](docs/04_PROJECT_REVIEW.md)：修复前发现与原始依据。
- [实施审计报告](docs/05_IMPLEMENTATION_AUDIT.md)：修复矩阵、测试结果、未验证事项和后续步骤。
- `reports/audit/2026-11-18/before/`：原始源码、文档、配置、数据及报告快照，不含 `.env` 和凭据。
- `reports/audit/2026-11-18/`：基线哈希、最终哈希、测试 XML 和修改证据。
- `data/.../document_events.jsonl`：创建/更新/删除事件，含版本与内容摘要，不记录正文或用户查询。它是普通本地文件，不是不可篡改合规审计存储。

`deploy/` 中 Redis/Qdrant/Prometheus 和 Kubernetes 是参考拓扑；应用尚未接入这些服务。合成语料中的企业制度、错误码、客户案例和合规叙述是测试素材，不是本项目部署事实。

API 1.1 变更：原 `faithfulness` 字段改为 `lexical_grounding_proxy`，拒答返回 `null`；实验统计字段改为 `bootstrap_nonpositive_fraction`；查询回显已脱敏。
