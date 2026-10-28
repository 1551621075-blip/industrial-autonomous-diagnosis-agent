# 使用说明

## 1. 建立数据和身份

先运行 README 的安装、init、build、publish。`init` 的管理员令牌来自安全随机数，数据库仅保存 hash；不要在 URL、文档、Git、截图中暴露令牌。

知识可导出查看：

```bash
uv run industrial-diagnosis fixture
uv run industrial-diagnosis fixture --reports-only
uv run industrial-diagnosis --help
```

知识 JSON 由 `KnowledgeBuildRequest` 严格验证。report 的 `synthetic` 固定 true；本工程不接收真实内部报告作为演示数据。人工审核应检查原始报告、条件单位、阈值与适用性，不可仅凭 HermiT success 就判断业务正确。

## 2. 工作台诊断

1. 选择一个明确运行模式启动本地服务，访问 `http://127.0.0.1:8003`。
2. 输入 `init` 时仅显示一次的本地管理员令牌，验证身份；“仅显示一次”不表示使用一次即失效。页面切换身份/退出后清除前一个身份的数据。
3. 如需构建新知识，选择本地合成 JSON；构建结果 checked 后再填写人工审核说明并发布精确版本。
4. 从授权合成事件中选择案例，输入调查描述。事件标签只用于演示下拉框，不作为模型的预期根因答案。
5. 建立任务并运行。模型会提议工具，程序核验并展示证据。页面等待时，不表示调用已失败；重复点击不会生成并发执行。
6. 遇人工补问时，只有确认已安全获得观察才作答。真实使用时不要为了让流程继续猜“是”；本工程的验收脚本会明确标注人工答案来自合成夹具。
7. 诊断提交后不可取消/改写。查看完整证据及审计；缺失或冲突不能当作正常。
8. 诊断结束后单独生成报告。模型草稿可能失败，不会改变之前诊断。对报告精确 hash 提交人工通过/驳回，不会启动设备措施。

## 3. 失败与恢复

| 状态/错误 | 含义与处理 |
|---|---|
| waiting_human | 等待声明的人工条件；不能用人工答案替代缺失信号 |
| blocked / model_unknown | 调用结果未知，不自动重发；原拥有者且管理员核查后对账 |
| knowledge_changed / snapshot_changed | 任务固定快照失效；建立新任务，不悄悄使用新数据 |
| budget_exhausted | 累计预算已耗尽，升级人工；重启/压缩不清零 |
| missing_signals | 已声明信号数据不完整，不能提交结论 |
| conflicting_signals / conflicting_candidates / unexplained_fault_signal | 完整信号无兼容路径、多路径冲突或有未解释正证据，转人工调查 |
| submitted | 程序提交校验通过的冻结规则结论，不是物理因果证明 |
| report failed/unknown | 报告生成问题，原诊断不受影响；同 action 不自动重试 |

未知结果对账 UI 支持明确确认“没有可用结果”，并可填写经核实的本次 token 总量；未知则留空，不要猜零。没有结果不代表没有消耗：未知用量留空会保守占用剩余预算，下一次运行将升级；已有用量不可减小。API 也支持补录完整模型 Action 和正整数 token 使用量。两者都是可审计人工声明，不等于自动验证外部事实。

## 4. API

除首页、静态资源、health 外，需要 `Authorization: Bearer <本地令牌>`。所有变更使用 JSON；拒绝跨源访问。OpenAPI 在线页默认关闭。

| 方法 / 路径 | 功能 |
|---|---|
| GET /api/me | 当前身份 |
| GET /api/cases | 授权合成事件列表 |
| POST /api/knowledge/extract | 从原创结构化报告片段提取草稿 |
| POST /api/knowledge/build | 真正 OWL/HermiT 构建，不自动发布 |
| POST /api/knowledge/publish | 管理员发布精确 version/hash |
| GET /api/knowledge | 授权知识版本 |
| POST /api/tasks | 建立幂等任务 |
| GET /api/tasks/{id} | 当前任务快照 |
| POST /api/tasks/{id}/run | 显式运行/恢复，不自动重发未知模型调用 |
| POST /api/tasks/{id}/answer | 绑定当前 question_id 的人工观察 |
| POST /api/tasks/{id}/reconcile | 拥有者管理员对账 |
| POST /api/tasks/{id}/cancel | 取消未结束且无活跃 worker 的任务 |
| GET /api/tasks/{id}/audit | 工具审计和模型调用状态 |
| POST /api/tasks/{id}/reports | 独立生成报告 |
| GET /api/tasks/{id}/reports | 查看该任务报告列表 |
| GET /api/reports/{id} | 报告详情 |
| POST /api/reports/{id}/review | 绑定 content_hash 的独立人工审签 |

动作需要稳定的 action_id；相同 ID 内容冲突返回错误。`run` 无动作 ID，它由持久化任务和 worker 租约保证不会并发重放；终态再次 run 只返回既有结果。

## 5. 合成案例的预期

| 案例 | 独立验收预期 |
|---|---|
| normal | 完整正常证据后 normal |
| jam | 电流/带速支持且有显式合成卡料观察后 jam |
| slip | 传动差值/电流支持且有显式合成打滑观察后 slip |
| sensor_fault | 编码器差异、正常带速/电流及独立测量确认后 sensor_fault |
| missing | 不提交诊断，保留缺失原因 |
| conflicting | 不提交单一故障，有未解释故障信号 |

这些预期不是传给模型的提示词，也不是企业设备验证。模型可以选择不同合法顺序，但不能改变最终证据条件。
