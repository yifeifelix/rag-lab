# RAG Learning & Evaluation Workbench

面向 RAG 学习、Applied Data Scientist 求职作品集和小型 Freelance PoC 的本地实验项目。

当前定位：**可运行的单用户 RAG 工作台，有文档生命周期、可复现实验和审计记录**。不是经过生产验收的企业级万并发服务。

本地部署使用 OrbStack，GitHub 用于代码备份、开源分享和 CI。部署与发布步骤、实际验证状态见 [GitHub / OrbStack 实施审计](docs/09_GITHUB_ORBSTACK_CICD.md)。原创代码采用 [MIT License](LICENSE)，第三方材料见 [许可证说明](THIRD_PARTY_NOTICES.md)。

```bash
scripts/orbstack up        # 本机构建、启动并检查 HTTP；需要 OrbStack 已运行
scripts/orbstack token     # 查看本地 owner token，输入网页顶部
scripts/publish-github     # 登录 GitHub，创建公开仓库并推送；默认 rag-learning-workbench
```

更新本地代码后再次运行 `scripts/orbstack up`。GitHub Actions 每次 push / PR 自动检查、测试、构建 Python 包和容器，不连接本机 Docker，不保存工作区、密钥或模型缓存。工作区数据需要自行备份；GitHub 的代码备份不包含它们。

1.2 修复状态见 [最新实施审计](docs/10_REMEDIATION_AND_USER_TESTING.md)，邀请用户前阅读 [测试手册](docs/11_USER_TEST_GUIDE.md)。历史审计保留原始证据，以新报告为准。

默认资料包含明确标注的虚构政策和 CPython 文档摘录，不能当作真实企业制度。

```bash
scripts/orbstack tester-token # 分享给测试者，提供查询、阅读与反馈权限
scripts/orbstack up-models    # 可选：联网锁定依赖、下载 MiniLM/BGE，启用真实模型和 Qdrant
```

## 已实现

- 四类分块、TF-IDF + SVD 离线 embedding、BM25、RRF、人工特征重排。
- 教学策略：技术标识符共现图、启发式检索路由、显式领域词典扩展、保留限制条件的抽取压缩，以及开发集 RRF 网格搜索（`POST /api/v1/rrf/tune`）。它们是简化实现，没有生成 HyDE 假设文档，也没有集成官方 LightRAG / AutoRAG 框架。默认扩展词典为空。
- 可选 Sentence Transformers 预训练 embedding，默认只读本地模型缓存；显式命令才允许下载。
- 本地抽取式回答，及可选 OpenAI/Gemini 生成适配器。响应明确标识模型、降级、token 用量和可选成本估算。
- L1/L2 缓存、singleflight、限流、并发门控、熔断。缓存属于单个 Python 进程。
- 文档导入、查询、更新、删除；原文保真、原子 JSONL、版本事件。SVD 基线重新拟合；固定 encoder 路径复用未改文档的分块、向量和 BM25 postings，可增量 upsert/delete Qdrant。
- 文档来源、版本、证据展示；Markdown/TXT 文件编辑器；单用户 owner token。
- 历史合成集七组诊断对照；新版挑战集 81 篇资料、82 道开发题、83 道测试题，按问题组隔离，包含证据片段标签、难负例、多跳、短查询和无答案题。题目仍由代理编写，独立用户审核尚未完成。
- 可选真实 cross-encoder。开发集按组交叉拟合置信度、按错误作答/误拒答代价选择阈值；目标是必要检索证据是否齐全，不是答案事实正确率。代码、语料或配置改变后校准失效。

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

默认可编辑语料位于 `data/workspaces/user_testing_v2/`；工作区和 `reports/private/` 已加入 `.gitignore`。客户资料的评估输出应显式指定到私有报告目录，分享前检查是否含正文和证据。

默认仅允许 loopback 访问 API。配置 `RAG_ADMIN_TOKEN` 后，浏览器顶部输入 owner token；`RAG_TESTER_TOKEN` 只允许查询、读取共享工作区资料和提交反馈，不能编辑文档或启动实验。不建议把这个单用户工作台当作多租户产品公开部署；尚无用户级/文档级 ACL。

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

## 开发、校准与冻结测试

```bash
scripts/workbench study tune
scripts/workbench study calibrate --output reports/private/calibration.json
scripts/workbench study compare
# 可选依赖与模型已下载后，保持同一候选集，对照真正的 cross-encoder：
scripts/workbench study compare --embedding sentence_transformers --include-neural
# 选定配置、重新校准后冻结；不根据最终测试结果反复改参数：
scripts/workbench study freeze --output reports/private/freeze.json
scripts/workbench study final-test --freeze reports/private/freeze.json
```

调参只读取显式 dev 文件，不访问测试题。校准和最终报告需使用相同的环境配置；若设置 `RAG_CALIBRATION_PATH`，冻结记录还会绑定该校准文件。配置变化后重新校准。

可以通过 `RAG_ENABLE_GRAPH=true` / `RAG_ENABLE_ROUTER=true` / `RAG_ENABLE_COMPRESSION=true` / `RAG_ENABLE_HEADERS=true` 分项开启教学模块。不要把它们默认组合成一个优胜方案。规则扩展需要带来源记录的显式词典，不能根据测试题编写默认规则。

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

`deploy/` 中 Redis/Prometheus/Kubernetes 仍是参考拓扑。Qdrant 已通过 REST 接入固定 encoder 路径，配置见根目录 `compose.models.yaml`；本轮真实容器与模型运行受执行环境限制，尚未验证。合成语料中的企业制度、错误码、客户案例和合规叙述是测试素材，不是本项目部署事实。

API 1.1 变更：原 `faithfulness` 字段改为 `lexical_grounding_proxy`，拒答返回 `null`；实验统计字段改为 `bootstrap_nonpositive_fraction`；查询回显已脱敏。
