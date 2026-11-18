# RAG 工作台实施审计报告

日期：2026-11-18。实施版本：1.1.0。项目：`/Volumes/Data/Projects/Rag`。

## 结论与交付边界

已完成本地可验证的缺陷修复、评估口径校准、文档生命周期闭环、真实资料学习案例和审计证据归档。项目现在更适合作为可解释、可复现的 RAG 学习与求职作品集，以及单用户 Freelance PoC 的基础。

**这不是企业生产验收报告。** 真实模型、独立人工评审、浏览器渲染、真实网络负载和多租户权限尚未完成验证。初次建议中的优先级 0 已落实；优先级 1 完成了本地闭环和案例基础，真实模型与独立质量验证仍待完成；优先级 2 的生产交付能力没有被宣称完成。

## 证据入口

| 证据 | 用途 |
| --- | --- |
| [实施前审查](04_PROJECT_REVIEW.md) | 原始问题及定位；现标记为基线 |
| [修改前快照](../reports/audit/2026-11-18/before/) | 34 个源码、文档、配置、数据和报告文件；不含密钥、`.env`、虚拟环境 |
| [基线哈希](../reports/audit/2026-11-18/baseline_manifest.json) | 校验修改前快照 |
| [最终文件哈希与差异清单](../reports/audit/2026-11-18/final_manifest.json) | 文件级追溯；包含本报告和验证证据的摘要 |
| [离线验证结果](../reports/audit/2026-11-18/validation.json) | 检查范围、命令、结果与明确未验证项 |
| [测试输出](../reports/audit/2026-11-18/pytest.log) / [JUnit XML](../reports/audit/2026-11-18/pytest.xml) | 最终 24 个测试通过 |
| [外部验证限制](../reports/audit/2026-11-18/external_validation.json) | DNS、端口监听、Orca 连接限制和模型凭据缺失 |
| [实验命令与退出码](../reports/audit/2026-11-18/experiment_commands.json) | 合成对照、开发/测试集实验、构建均为退出码 0 |
| [依赖版本](../reports/audit/2026-11-18/installed_dependencies.txt) | 当前虚拟环境的软件版本快照 |
| [wheel 构建记录](../reports/audit/2026-11-18/wheel_build.log) / [包内冒烟验证](../reports/audit/2026-11-18/wheel_smoke.json) | 包含静态页面，包内导入/页面服务/文档导入和问答通过 ASGI 检查 |

这些是普通本地文件和 SHA-256 摘要，可用于核对变更；不是第三方签名或不可篡改的合规审计系统。项目没有 Git 仓库，本次没有创建提交或发布部署。

## 修改与验证矩阵

| 编号 | 原问题 | 实施结果 | 主要依据 |
| --- | --- | --- | --- |
| A01 | L2 淘汰改变列表索引，其他 scope 可能错答 | 矩阵持有稳定条目引用；过期后重新选择仍有效的结果 | `cache.py`；跨 scope 淘汰、过期最优项回归测试 |
| A02 | 脱敏后的 response 仍保存原始查询 | 返回、L1、L2 和 singleflight 仅保留脱敏查询；命中时更新本次脱敏标志 | `pipeline.py`；响应/缓存邮箱不存在性测试 |
| A03 | 近似语义缓存混用技术编号/数字 | L2 scope 加入语料版本和编号/数值约束 | 不同 SLA 数值不能语义命中的测试；不保证所有相似问题都语义等价 |
| A04 | 词覆盖分数误称 Faithfulness | 更名 `lexical_grounding_proxy`；拒答为 null；没有观测值的指标为 null，并提供样本数 | 关系颠倒反例、空指标测试；页面/脚本/报告同步 |
| A05 | bootstrap 比例被称为显著性 p 值 | 更名 `bootstrap_nonpositive_fraction`，明确描述性含义 | `evaluation.py`、README、实验报告 |
| A06 | 客户文档接入闭环缺失 | Markdown/TXT 编辑器、列表、导入、更新、删除；原子 JSONL 保存；来源与版本展示 | CRUD、缓存失效、删除后重启、跨本地 worker 刷新、版本冲突测试 |
| A07 | 每次启动重新构建且没有索引快照 | JSON/NumPy 持久化，无 pickle；指纹覆盖数据、分块、embedding、相关源码和库版本 | 重启加载并得到同样回答；删除清理旧快照 |
| A08 | 私有数据与访问边界不清楚 | 默认可编辑工作区独立；workspace/private reports 忽略提交；单 owner token；无 token 仅 loopback | 401/403、授权查询、普通查询限流 429 测试 |
| A09 | 只有合成资料 | 导入本机官方 CPython asyncio 文档字符串，记录来源、版本、源码位置及哈希 | 9 篇真实资料、9 道开发题、13 道固定测试题 |
| A10 | 无预训练 embedding 主管线接线 | 增加 Sentence Transformers backend、显式下载入口、默认本地缓存加载 | 缺依赖明确失败，不静默回退；真实权重未下载验证 |
| A11 | 云端失败和费用不透明 | 返回生成模式、降级、token 用量；按运营者提供单价估算；缓存不重复计生成用量；只列出回答实际提及的引用 ID | MockTransport 验证脱敏 prompt、用量、成本与引用；失败回退不缓存 |
| A12 | “万并发”外推过度 | 固定为本地合成协程实验；新增真正 HTTP 客户端脚本；写明计时与工作负载范围 | 合成 10,000 请求完成；真实 socket 检查被沙箱拒绝 |
| A13 | `.env.example` 不会自动加载，缓存路径未约束 | 实现无 shell 执行的 `.env` 读取；启动器在解释器启动前设置 Data 盘路径 | 环境优先级及 TMPDIR 路由测试；不覆盖既有 `.env` |
| A14 | 文档夸大成熟度、UI 直接插入证据文本 | 重写 README/学习/面试/架构说明；转义证据文本；UI 说明缓冲 SSE 回放和启发式分数 | JS 语法、静态资源打包及页面接口检查；浏览器渲染未验证 |

真实神经 cross-encoder 未实现；当前 reranker 仍是人工加权特征。原有 `CloudEmbeddingAdapter` 也没有被当作已接入的云 embedding 功能宣传；本次接入的是可选本地预训练 encoder。

## 文档与索引变更语义

- 默认可编辑知识库：`data/workspaces/demo/documents.jsonl`。原始合成数据保留用于追溯。
- 导入/更新/删除由 POSIX 锁协调；先构建新索引，成功后原子替换语料和运行时管线。构建失败保持旧语料可用。
- 新管线具有新版本标识和空答案缓存；旧索引文件被删除。已在途请求允许完成旧版本快照，不能据此宣称强一致删除或合规擦除。
- 其他本地 worker 在后续请求发现文件变化后刷新；不提供跨机器数据库事务。
- `document_events.jsonl` 记录操作、时间、文档 ID、版本摘要和内容哈希，不保存文档正文或用户查询。事件日志与语料文件不是单一数据库事务，也没有不可篡改保护。
- 导入内容和查询只使用已实现的 PII 正则做有限脱敏；文档 ID 必须使用非敏感标识。此能力不是通用 DLP，也不替代私有数据权限管理。

## 重新运行的实验结果

### 合成资料：用于机制学习和历史对照

65 篇合成文档、40 道 QA 的六组对照已重新生成。完整配置的 Recall@5 为 0.971；BM25 的 NDCG@5 为 0.940，仍高于完整配置的 0.936。没有把复杂配置包装成所有指标的优胜者。

当前统计字段只表示这套资料上的观察结果；没有独立审核、测试集代表性或真实业务泛化保证。完整逐题回答与证据保存在 [对照结果](../reports/ablation_results.json)。

### 真实资料：CPython asyncio 小领域案例

资料来自已安装的 CPython 3.12.15，来源和许可记录在 [数据集清单](../data/python_support/dataset_manifest.json)。固定资料保存在 `data/python_support/`；可编辑演示副本在 `data/workspaces/python_support/`，防止演示修改污染固定资料。

四组受控检索比较保持父子分块和 embedding 不变：BM25、Dense、Hybrid RRF、Hybrid + 特征重排。当前实际运行使用本地 SVD 和抽取式生成。

| 测试集架构 | Recall@5 | 关键词覆盖率 | 人工正确性 |
| --- | ---: | ---: | --- |
| BM25 | 1.000 | 0.3636 | 待审核 |
| Dense | 1.000 | 0.3636 | 待审核 |
| Hybrid RRF | 1.000 | 0.3636 | 待审核 |
| Hybrid RRF + 特征重排 | 1.000 | 0.3182 | 待审核 |

这暴露了当前最需要继续改进的地方：检索命中了文档，回答仍可能不完整。关键词覆盖也不是准确率，不能把上表当作模型正确性成绩。

开发题没有不可回答样本，因此该部分 Refusal Accuracy 为 null；测试题只有两个不可回答样本，2/2 拒答只能描述这些样本。

限制：资料只有 9 篇、竞争检索难度较低；22 道题由同一实施代理编写，尚无独立人工审核；测试结果已被实施代理看到，不能称为盲测；文档字符串可能与在线文档或运行时行为不同。没有为了提高测试分数改写这些冻结题目。

逐题评审入口：

- [开发集评审表](../reports/python_support/local_svd/local/dev/pending_human_review.jsonl)
- [测试集评审表](../reports/python_support/local_svd/local/test/pending_human_review.jsonl)
- [当前人工审核汇总](../reports/python_support/local_svd/local/test/human_review_summary.json)：全部待审核，正确率为 null。

每行填写 `review.correctness` 为 `correct` / `partial` / `incorrect`；`citation_support` 为 `supported` / `unsupported` / `not_applicable`，并填写 reviewer。不要把代理自动评分当作独立人工审核。

```bash
.venv/bin/python scripts/summarize_human_review.py \
  reports/python_support/local_svd/local/test/pending_human_review.jsonl \
  --output reports/python_support/local_svd/local/test/human_review_summary.json
```

### 10,000 请求实验

最新准确数字见 [合成负载报告](../reports/benchmark_10k_report.md) 和 [JSON](../reports/benchmark_10k.json)。本次最终复跑完成 10,000/10,000，请求峰值后端并发 256，缓存加合并比例为 74.86%。

加入编号/数值约束后，语义缓存命中减少，部分请求进入合并/后端路径；总体 P50 约 312ms，原先宣传的 0.017ms 中位数不再适用。完整 P95/P99 及分层计数均保留，不用旧数字替代新结果。

这是一次有限时长、单 Python 进程、构造流量、预热缓存、模拟 2ms 等待的实验；没有真实网络或 LLM。完成率不是回答正确率，也不是生产 SLA。真实服务还没有经过持续负载、CPU/内存监测和费用验证。

## 验证过程与失败记录

最终完整测试：**24 passed**。涵盖原有机制、额外缺陷回归、文档闭环、访问控制、模型失败、Mock API 契约及空指标处理。测试在隔离的 Data 盘临时目录运行，不使用客户工作区或真实模型。

Python AST 与页面 JavaScript 语法通过。构建出的 1.1.0 wheel 与当前包源码一致，包含静态页面，未打包 `.env`、运行时语料或缓存。额外从解包后的 wheel 导入并进行 ASGI 页面服务、文档导入和问答冒烟验证，均通过。

没有隐藏首次失败：

1. 初次隔离全测有 1 项失败，原因是旧分块测试隐式依赖仓库已有数据。已改为逐测试生成隔离 fixture。失败输出保留在 [attempt_1](../reports/audit/2026-11-18/attempt_1/)，最终全测通过。
2. 首次离线 wheel 构建因虚拟环境缺少 setuptools 而失败；随后使用 Data 盘中既有缓存的 setuptools 84.2.1 构建，未联网安装或写入系统目录。
3. 构建后发现旧 editable 安装的发行元数据仍是 1.0.0，已离线更新本项目 editable 安装到 1.1.0，并核对代码与元数据版本一致。未升级其他运行依赖。

真实 HTTP 监听失败的准确错误为 `Errno 1 / operation not permitted`。Orca 浏览器运行时连接失败为 `runtime_access_denied / EPERM`。终端外网下载失败为 DNS 无法解析。没有通过其他工具绕过这些限制；因此没有把页面语法或 ASGI 检查标记为真实浏览器/网络验收。

## 未完成事项及后续验收

| 事项 | 状态 | 下一步 |
| --- | --- | --- |
| 预训练 encoder 真实运行 | 未验证 | 在可联网环境安装可选依赖，下载并固定模型版本，重复开发集比较 |
| 真实云端生成与成本 | 未验证 | 配置可用模型和密钥，先跑少量问题；确认没有 fallback，再记录用量和质量 |
| 独立回答正确性评审 | 待人工 | 审核来源、QA、引用支持与拒答；扩展有代表性的业务问题集 |
| 浏览器导入/编辑/删除与渲染 | 未验证 | 正常本地终端启动，用浏览器逐项操作并保存截图/控制台证据 |
| 真实网络持续压测 | 未验证 | 运行 HTTP 脚本，明确负载、持续时长、缓存条件、资源、错误率与成本 |
| 用户/文档级 ACL、多租户 | 未实现 | 私有多用户交付前，实现身份、检索前权限过滤及权限范围缓存 |
| Redis/Qdrant 分布式状态 | 未接入 | 有实际容量需求时实现和测试；现有部署文件仅作参考 |
| 神经重排、真实 token 流式生成 | 未实现 | 分别验证质量/延迟收益；当前 SSE 明确为完整回答后的分段回放 |
| 监控、备份恢复、完整审计事务 | 未完成 | 根据客户部署环境和验收要求补齐 |

相似度加编号/数值约束仍不能证明两个自然语言问题语义等价；否定、条件、时间范围等缓存误命中仍需专门评估。正则注入拦截也不是完整安全防线。

## 后续审计与复核命令

```bash
cd /Volumes/Data/Projects/Rag
# 离线检查；默认生成新的时间戳目录，不覆盖这次报告。
scripts/workbench audit

# 对照本次变更，不会自动恢复或覆盖客户数据。
diff -ru reports/audit/2026-11-18/before/src/rag_platform src/rag_platform

# 文档/人工评审修改后，仅刷新本次哈希。
scripts/workbench audit --manifest-only --output reports/audit/2026-11-18

# 正常终端中启动服务，另一终端执行真实 HTTP 负载。
scripts/workbench start
scripts/workbench http-benchmark --requests 200 --concurrency 10
```

后续若修改业务资料、配置、模型或源码，应生成新的评估报告和审计目录，不沿用本次成绩。客户数据报告可能包含完整回答及证据，应放在私有目录并审查分享范围。恢复旧代码前应先评估已修复缺陷的回归风险，不自动回滚工作区和客户数据。
