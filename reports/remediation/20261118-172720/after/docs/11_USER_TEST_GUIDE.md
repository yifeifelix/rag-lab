# 用户测试手册

这是一套共享工作区的 RAG 学习原型。邀请少量用户检查答案、引用、拒答与交互；尚未完成独立效果验收，也未验证生产容量。

当前离线基线在已见题集上仍有约 35% 的可回答题被保守门控拒答；测试时请特别记录“应该回答却拒答”的问题。检索满分不代表回答正确。真实模型与 Qdrant 路径尚待实际机器验证。

## 运行

在普通本机终端、OrbStack 已打开的情况下：

```sh
cd /Volumes/Data/Projects/Rag
scripts/orbstack up
scripts/orbstack token
scripts/orbstack tester-token
```

访问 `http://127.0.0.1:8000`。你使用 owner token；给参与测试的人 tester token。二者保存在权限 0600 的 `.orbstack.env`，不提交 GitHub。页面不持久保存 token，刷新后重新输入。

如果用户使用同一可信局域网的其他设备，显式开启局域网监听：

```sh
RAG_BIND_HOST=0.0.0.0 scripts/orbstack up
```

把本机局域网 IP 的 `http://<IP>:8000` 和 tester token 交给对方。此方式用于可信网络的演示；不应将带 token 的明文 HTTP 暴露到公网。恢复本机监听时执行 `scripts/orbstack up`。本轮代理环境无法执行真实端口或容器测试，因此 LAN 可达性需要在实际设备确认。

tester 可以读取整个共享演示工作区、提问、提交反馈；不能增删改文档、调参、校准或压测。这不是每个测试者各自隔离的文档空间，请使用演示资料。

## 测试任务

| 任务 | 操作 / 检查 |
| --- | --- |
| 精确问题 | `What is the current EU restore recovery target for Cedar?`；核对 current/2026、地区、单位，点击完整证据检查 |
| 条件干扰 | 把 EU 改成 US；答案和证据应反映另一地区，不能沿用原缓存 |
| 多跳 | `Under the 2026 Cedar EU policies, give both the restore recovery target and export retention period.`；核对两个不同操作的依据 |
| 短查询 | 试 `Semaphore`、`Queue`；记录误拒答或错误证据 |
| 无答案 | 询问资料未给出的续约费、退款保证；记录系统是否编造 |
| 自己的问题 | 阅读资料后自行编写问题及预期支持句，不使用页面样例的近似改写充当独立测试 |
| 文档变更（owner） | 导入一篇测试文档、提问、修改数值、再提问、删除后再问；观察版本、缓存和引用 |
| 原文保真（owner） | 保存带订单号、电话和测试卡号的虚构文本，读回确认原文保留；查询和远程外发使用脱敏策略 |
| 压力 / 交互 | 先用数名用户正常提问；记录延迟、429、界面错误，勿把一次合成压测当容量认证 |

页面显示完整生成证据。SSE 是完整答案生成后的回放，不代表真实模型 token 流式输出。切换检索模式、关闭重排或使用域过滤后，原校准可能不适用，页面会显示 `uncalibrated_relevance`。

每次回答后展开“Review this answer”，选择评分与问题类别。评论中写明去除敏感信息后的问题、期望答案和错误位置，有助于复现。反馈不自动保存查询或答案正文；请在有必要时主动补充到评论。

反馈保存在 `reports/private/user_feedback.jsonl`（模型模式为其 `models/` 子目录），包含 response ID、语料版本、评分、问题类型和脱敏评论，不自动加入训练/开发集。重启后旧 response ID 不再能提交反馈，需要重新提问。

## 真实模型与 Qdrant

```sh
scripts/orbstack up-models
```

首次执行会联网解析 Linux/当前 CPU 架构的完整带哈希依赖锁、构建模型镜像、下载英文 MiniLM 与 BGE cross-encoder、进行实际模型推理检查并记录 immutable revision，随后启动 Qdrant 与应用。缺少模型或依赖时会失败，不会把本地特征模型冒充成神经模型。

这一步可能下载较大的模型和 CPU PyTorch 包。网络、真实模型和真实 Qdrant 在本轮环境中未能实测；目前提供的是完成接线并经过模拟契约检查的路径。

两种模式使用独立的可编辑文档目录：`user_testing_v2` 与 `models_testing_v2`。模型缓存、Qdrant 存储和报告绑定到项目 Data 盘目录。OrbStack 自身的镜像/虚拟磁盘位置仍由 OrbStack 设置决定。

启动时只用开发集校准。默认本地抽取回答，不调用收费 LLM API。若工作区被改为与开发题无关的资料，自动校准可能不可用；应用显示未校准分数，owner 应准备匹配的新开发集后重新校准。

## 复测记录

```sh
scripts/orbstack status
scripts/orbstack smoke
scripts/orbstack logs
# 停止，保留绑定目录：
scripts/orbstack down
```

请记录应用版本、语料版本、所用模式、问题、预期依据、实际结果、response ID 和测试日期。独立用户题先由人工审核并冻结；若之后用它们改规则或阈值，它们应归入开发资料，再另收最终测试题。
