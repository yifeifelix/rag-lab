# GitHub 与 OrbStack 实施审计

日期：2026-11-18。需求：本地使用 OrbStack；GitHub 用于备份、开源分享及自动验证。

## 部署边界

- 本地源码仍位于 `/Volumes/Data/Projects/Rag`。`scripts/orbstack up` 构建当前本地源码、等待容器健康、检查 HTTP，适合本地更新部署。
- GitHub Actions 在 GitHub 的临时 runner 上检查 push / PR，生成测试 XML、wheel 和源码包。不部署到云服务器，也不控制本机 OrbStack。
- 默认公开仓库名 `rag-learning-workbench`，使用运行发布脚本时登录的 GitHub 账号。仓库创建和推送成功后以脚本输出 URL 为准。
- `.venv`、`.env*`（示例除外）、`.orbstack.env*`、缓存、索引、可编辑语料和私有报告不提交。GitHub 不是客户工作区或运行数据备份。

## 已增加的实现

| 文件 | 用途 |
| --- | --- |
| `Dockerfile` / `.dockerignore` | Python 3.12 多阶段镜像；安装带哈希锁定依赖；非 root 用户；只复制指定演示材料 |
| `compose.yaml` | OrbStack 单应用、本机端口、Data 盘目录挂载、单 worker |
| `scripts/orbstack` | 本地构建、健康等待、HTTP 验证、日志、停止；自动生成权限 0600 的 owner token |
| `scripts/smoke_http.py` | 检查健康、未授权访问返回 401、授权本地问答 |
| `.github/workflows/ci.yml` | 源码分享检查、Ruff 正确性规则、pytest、构建包、检查 wheel、构建容器和 HTTP 测试 |
| `requirements/{runtime,ci,build}.lock` | uv 解析得到的 Python 3.12 版本与包哈希锁；不包含可选深度学习模型依赖 |
| `scripts/publish-github` | 正常本机终端中登录、初始化 main、扫描待提交内容、提交、创建公开仓库并推送；不强推 |
| `scripts/check_repository.py` | 检查提交范围中的私有运行路径、常见 token / 私钥格式和大文件；不是完整的敏感内容审计 |
| `LICENSE` / `THIRD_PARTY_NOTICES.md` | 原创代码 MIT；CPython 文档摘录保留其许可证 |

`runtime.configure_runtime()` 新增显式 `RAG_DATA_ROOT`：本机默认 `/Volumes/Data`；CI 显式指定 checkout 根目录；容器指定 `/app`。已有 import 配置副作用没有在这次部署工作中重构。

## 正常本机终端操作

```bash
cd /Volumes/Data/Projects/Rag
scripts/orbstack up
scripts/orbstack token
# 打开 http://127.0.0.1:8000，把 token 填入网页顶部。

scripts/publish-github
# 可传入另一仓库名：scripts/publish-github your-repository-name
gh run list --workflow ci.yml
gh run watch
```

发布脚本首次登录可能打开 GitHub 授权页面。已有同名远程仓库不会被覆盖；创建失败时先检查归属和内容。后续备份：`git add <本次需要的文件>`、`git commit`、`git push`。已有私有仓库不会被脚本自动改为公开。

本地改代码后重新执行 `scripts/orbstack up`；停止执行 `scripts/orbstack down`，绑定目录保留。手动选定旧 commit 后可重新构建旧代码；数据格式迁移和数据回滚不由此脚本保证。

Compose 明确使用 `orbstack` Docker context，不切换全局 context。只监听 `127.0.0.1`，默认本地抽取回答与 SVD embedding，不产生云模型费用。浏览器和宿主机虚拟磁盘空间不受 Python 缓存路径限制；请在 OrbStack 设置中确认其磁盘位置。

运行目录：`data/workspaces/`、`reports/private/`、`.cache/orbstack/`，均位于项目目录。已有宿主机演示工作区会被继续使用；首次容器报告目录从随镜像附带的历史合成实验结果初始化，**这些是演示报告，不是本次部署的新测试成绩**。

## 验证记录

本次实际检查：

| 检查 | 结果 |
| --- | --- |
| pytest 回归与新增路径隔离用例 | **27 passed，8.18 秒**；XML：`reports/audit/2026-11-18/github_orbstack_junit.xml` |
| Compose 配置 | `docker --context orbstack compose config --quiet` 通过（仅解析，未启动容器） |
| YAML / shell 语法 | Compose、CI、Dependabot YAML 解析通过；四个 shell 启动脚本语法检查通过 |
| 带哈希 CI 依赖 | `uv pip sync --require-hashes --dry-run --offline` 解析 31 个包通过；未修改既有虚拟环境 |
| Python 构建 | wheel 与 sdist 构建通过；wheel 全部模块及静态网页与当前源码逐字节一致 |
| 独立 wheel 验证 | 解包 wheel 后从其路径导入；通过 ASGI 内存客户端验证网页、健康、401 鉴权、本地问答；没有使用网络端口 |
| 提交范围检查 | 当前可分享目录未发现被检查的常见凭据格式；不代表完整人工隐私审核 |
| Ruff | 未执行成功：本机无缓存，当前环境不能联网安装 |
| 真正的 HTTP 冒烟测试 | 未执行成功：当前环境禁止绑定和连接本机端口；不能以 ASGI 内存测试替代网络验证 |
| Docker 构建 / OrbStack 启动 / GitHub Actions | 未执行成功，见下方权限与认证限制 |

远端 CI 成功与实际 Docker 启动必须分别记录，不能由静态配置或本机 pytest 推断。

结构化结果和实施文件 SHA-256 见 [`github_orbstack_validation.json`](../reports/audit/2026-11-18/github_orbstack_validation.json)。其中发布范围计数记录保存审计文件之前的检查；新增审计文件后再次检查通过，共 252 个文件。

已确认的执行限制：当前代理环境禁止写 `.git`，`git init -b main` 返回 `Operation not permitted`；访问 OrbStack socket 返回 `permission denied`；`gh auth status` 报现有账号 `yifeifelix` 的令牌无效，GitHub API 网络访问也失败。因此当前没有可报告的 GitHub 仓库 URL、远端 CI run 或已启动的 OrbStack 容器。这些步骤需由正常本机终端运行上面的脚本完成。

锁文件在本机 Python 3.12 的离线 uv 缓存中解析并生成哈希。Linux 安装和容器运行仍需首次真实 CI / OrbStack 构建确认。Ruff 命令固定 `0.14.0`，首次执行需要网络下载。CI 使用官方 Actions 主版本标签，由 Dependabot 跟踪升级；setup-uv 固定到 commit。依赖哈希锁不保证 Python 基础镜像逐字节可重现。

## RAG 审计事项仍然开放

这次部署工作没有修复查询扩展与测试集重叠、AutoRAG 在评测题上调参、CRAG 拒答门控、置信度覆盖、压缩证据不一致及命名夸大等问题。详见 `docs/07_NEW_FEATURES_REVIEW.md`、`docs/08_EVALUATION_VALIDITY_RECHECK.md`。CI 通过说明回归测试与构建通过，不能证明检索泛化能力、真实模型效果或万并发容量。

## 配置依据

- [GitHub 官方 Python CI 文档](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
- [checkout](https://github.com/actions/checkout)、[setup-python](https://github.com/actions/setup-python)、[upload-artifact](https://github.com/actions/upload-artifact)
- [setup-uv](https://github.com/astral-sh/setup-uv)
