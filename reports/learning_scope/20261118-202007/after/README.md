# RAG Learning Workbench & Lab

以学习路径为主的本地 RAG 实验台：**看懂链路 → 一次改变一个变量 → 核查证据 → 分析失败。** 求职作品集和 Freelance 演示是学习成果的用途。

打开应用先进入“学习路径”，每课都有概念、动手任务、完成标准和代码阅读入口。“链路观察”用于提问和查引用，“单变量 Lab”用于左右对照，“文档工作区”用于导入、修改和删除资料。

## 开始

已有 `.venv` 时：

```sh
cd /Volumes/Data/Projects/Rag
scripts/workbench start
# 打开 http://127.0.0.1:8000
```

也可以使用 OrbStack：

```sh
scripts/orbstack up
scripts/orbstack token
```

把 owner token 填入页面后连接。OrbStack 是运行方式，无须先学习容器、Qdrant 或 CI/CD。当前代理环境仍不能访问 Docker socket，容器运行状态需要在正常本机终端确认。

默认教学组件为 **BM25 / TF-IDF + SVD、特征重排和本地抽取回答**。学习页初始关闭重排与答案缓存；容器不自动校准，不下载模型，不调用云端 API。基础拒答和原文/引用正确性修复继续保留。

新建 Python 环境需要 Python 3.12 和 uv：

```sh
mkdir -p .cache/tmp
UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/uv-python" TMPDIR="$PWD/.cache/tmp" uv venv --python 3.12 .venv
UV_CACHE_DIR="$PWD/.cache/uv" TMPDIR="$PWD/.cache/tmp" uv pip sync --require-hashes requirements/ci.lock
UV_CACHE_DIR="$PWD/.cache/uv" TMPDIR="$PWD/.cache/tmp" uv pip install --no-deps --no-build-isolation -e .
```

默认使用 `/Volumes/Data`，源码外的可写数据和缓存也需位于配置的数据根；其他机器显式设置 `RAG_DATA_ROOT` 和 `RAG_PROJECT_ROOT`。启动时可读取 `.env`，示例为 `.env.example`，已导出的变量优先。学习主线不需要 API key。

## 七步学习主线

| 课 | 动手任务 | 能解释什么 |
| --- | --- | --- |
| 1. 最小链路 | 提问，打开完整证据，找到支持答案的原句 | 原文、检索块、回答和引用的关系 |
| 2. 分块 | 预览同一文档，再比较 fixed / parent-child | 匹配粒度、边界、限制条件和字符预算 |
| 3. 检索 | 固定分块，对照 BM25 / SVD 向量检索 | 精确术语与改述为何可能有不同结果 |
| 4. 融合 | 只切换 BM25 / RRF | 为何融合名次，不能直接相加不同算法分数 |
| 5. 重排 | 在相同检索方案上开关特征重排 | 召回候选与最终排序的区别 |
| 6. 拒答 | 对已知、未知、短问题比较两个阈值 | 错误作答和误拒答的取舍，分数不等于正确概率 |
| 7. 评估 | 改 top-k，核对多跳证据，人工标注自己写的问题 | Recall 的边界、测试集隔离和失败分析 |

详细任务见 [学习路线](docs/01_DS_RAG_LEARNING_ROADMAP.md)。每课的“完成”勾选是自评，只在浏览器保存课号，不代表自动验收，也不保存 token。

## Lab 如何工作

选择一个问题、一个变量、A/B 两个值；先写假设，再运行并检查两边的证据、回答与拒答。Lab 固定为本地 SVD / 抽取回答，关闭缓存、校准和其他高级开关，不改变服务配置、工作区文档或原有索引。

分块变化会引起局部 SVD 表示重新拟合，这是该实验选择的后果。一次查询的耗时受初始化与运行环境影响，不是速度基准；单条问题也不能证明某方案普遍更好。

owner 可以勾选保存运行，记录写入 `reports/private/learning_lab/`，包含假设、配置、语料版本和两边结果。观察笔记仅留在当前页面，需要时另行记录。tester 可以运行不写入文件的基础 Lab，不能编辑文档或保存运行。

默认资料包含虚构政策和 CPython 文档摘录。不要把虚构内容当作企业事实。可编辑文档在 `data/workspaces/user_testing_v2/`；它与冻结评测资料分离。

## 主线完成后再选专题

- **真实模型**：比较一个英文预训练 encoder，再比较特征重排与真正的 cross-encoder；模型与运行成本另行记录。
- **检索策略**：上下文压缩、显式词典扩展、技术词项共现图，逐项开关，不默认组合成“最佳方案”。
- **评估方法**：开发集调参、置信度校准、冻结配置与最终测试。历史题目已被实现者看过，不能当独立盲测。
- **运行机制**：缓存、singleflight、限流、合成并发实验。
- **交付展示**：Qdrant 增量更新、OrbStack、GitHub。Kubernetes、Redis 等参考拓扑不作为学习前置条件。

这些模块保留在代码中；默认页面收起进阶入口，初始化不自动跑评估。真实模型和真实 Qdrant 尚未在本次代理环境验证。学习主线只依赖基础 Python 环境。

```sh
scripts/workbench study compare       # 可选：开发集上的批量对照
scripts/workbench study calibrate     # 可选：观察校准取舍，不会自动应用
scripts/orbstack up-models            # 可选：需要联网的模型 + Qdrant 模式
```

## 记录与分享

完成一个实验后，保留问题、假设、唯一变量、支持原句、观察、失败例子与局限，再用于面试或 PoC 演示。项目仍是共享工作区的学习原型，不承诺生产容量或答案正确率。SSE 是完整答案生成后的回放，词项覆盖代理不是事实验证。

GitHub 用于源码备份和开源分享，保留简单自动检查；不会自动部署到本机。`.venv`、密钥、工作区、缓存和私有报告不提交。Docker 镜像/VM 磁盘与浏览器本身的空间位置由相应软件控制。

- [学习主线调整审计](docs/12_LEARNING_FIRST_SCOPE.md)：当前产品范围与默认行为。
- [此前修复审计](docs/10_REMEDIATION_AND_USER_TESTING.md)：保留历史结果；其中自动校准启动说明以 12 号报告为准。
- [用户测试手册](docs/11_USER_TEST_GUIDE.md)、[GitHub / OrbStack](docs/09_GITHUB_ORBSTACK_CICD.md)：可选运行与分享。
- [MIT License](LICENSE)、[第三方材料说明](THIRD_PARTY_NOTICES.md)。
