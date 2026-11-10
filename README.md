# 工业自主诊断 Agent（本地 Demo／测试版）

本仓库提供本项目的本地 Demo／测试版本，用于生产部署前在本地验证核心功能、业务流程与交互方式。

基于原创合成数据开展输送带诊断。**单 Agent ReAct + Heavy Harness**：本地模型选择下一步调查动作，程序独立执行信号验证、证据绑定、权限检查和提交校验。不连接生产接口、不控制设备，也不把规则满足冒充物理因果证明。

**使用范围：** 本版本使用合成数据与本地测试环境，不包含真实业务或设备数据、内部资料、生产模型及敏感配置。本地验证结果不代表生产环境的可用性、效果或业务收益。

本机验收：**1,064 项测试通过、1 项文件系统限制跳过；六组合成程序案例通过；真实本地 Qwen 卡料与正常两个案例的诊断及报告通过；Chrome 26 个检查组通过、0 失败**。测试提供器不等于真实模型，合成数据不等于生产验证；详细结果与未覆盖范围见 [验收记录](docs/validation.md)。

## 工程内容

- **真实 LangGraph 五节点**：`agent / tools / guard / compact / escalate`，不把固定工具流水线称为 ReAct。
- **六种工具**：意图分类、知识候选检索、时序验证、树节点展开、人工补问、诊断提交。
- **离线知识生命周期**：三份原创结构化合成报告 → 安全 YAML 提取 → Pydantic → OWL → 真正 HermiT 一致性/分类与重载核验 → 人工管理员发布。
- **在线诊断**：事件/设备/工厂/时间窗/知识版本/数据 hash 固定；数值证据与人工观察分别记录；连续路径、必检、AND/OR、缺失、反证和冲突均由程序检查。
- **持久化恢复**：SQLite、内容寻址制品、调用前登记、版本及租约隔离、未知结果不自动重发、明确人工对账。
- **提交后冻结**：独立报告生成失败不能改写诊断；模型草稿与确定性结论分区；人工审签不触发设备操作。
- **浏览器工作台 / API / CLI**：本地身份、知识审核、事件调查、证据链、人工补问、报告和审计。
- **合成案例**：卡料、打滑、传感器差异、正常、缺数据、冲突信号，及跨设备/工厂隔离样例。

## 快速运行

需要 Python **3.12**、uv；真实知识构建还需要显式指定 Java 17。默认不发现或启用云端服务，不自动下载模型。

```bash
uv sync --frozen
uv run industrial-diagnosis init ./.industrial-data
```

`init` 只接受**不存在的新目录**，生成原创合成时序并一次性输出本地管理员令牌。保存令牌到你自己的安全位置；不要提交到仓库。数据库只保存令牌 hash。此时没有自动发布知识。

```bash
# JAVA 指向你显式选择的 Java 17 可执行文件
uv run industrial-diagnosis build ./.industrial-data "$JAVA"
```

按提示输入本地令牌。检查构建结果中的 `draft`、`provenance`、`reasoner`、`version_id` 和 `content_hash`。**只有 `checked` 才可发布**；缺 JVM、推理失败/超时、不一致或命名类不可满足都会阻断。

```bash
uv run industrial-diagnosis publish ./.industrial-data VERSION_ID CONTENT_HASH ACTION_ID \
  "已检查原创合成规则、单位、来源和真实推理结果，仅用于本地演示"
```

生产式本地模型路径——这里的“本地模型”不代表生产验证：

```bash
uv run industrial-diagnosis serve ./.industrial-data --runtime-config /absolute/path/runtime.json --java-path "$JAVA"
```

或显式连接你自己启动的数值 loopback 模型端点：

```bash
uv run industrial-diagnosis serve ./.industrial-data --model-url http://127.0.0.1:18080 --java-path "$JAVA"
```

仅测试程序流程，不做模型能力验收：

```bash
uv run industrial-diagnosis serve ./.industrial-data --test-mode --java-path "$JAVA"
```

打开 `http://127.0.0.1:8003`，输入令牌。页面会明确显示 `TEST-ONLY` 或本地模型模式。令牌只在当前页面内存，不进入 URL、浏览器 localStorage/sessionStorage。

详细步骤：[使用说明](docs/usage.md) · [运行时](docs/runtime.md) · [架构与边界](docs/architecture.md) · [依赖说明](docs/dependencies.md)

## 验证

```bash
INDUSTRIAL_DIAGNOSIS_JAVA="$JAVA" uv run pytest
uv run ruff check src tests scripts

# 新目录，真实 HermiT + 明确 TEST 提供器，完整六案例
uv run python scripts/run_acceptance.py \
  --data-root /absolute/path/new-synthetic-acceptance \
  --java "$JAVA" --test-provider

# 新目录，真实 HermiT + 本地 Qwen，不回退到 TEST 提供器
uv run python scripts/run_acceptance.py \
  --data-root /absolute/path/new-real-acceptance \
  --java "$JAVA" --runtime-config /absolute/path/runtime.json --cases normal jam
```

验收脚本中人工观察是**显式的合成测试答案**，不是现场观察或真人审查。预期诊断只用于最后的独立断言，不传入模型。未设置 Java 时，特定真实推理测试会明确跳过，不能称为全部真实推理通过。

其他验证入口：

```bash
# Node 24 与本机 Chrome；创建独立临时 profile，不使用你的浏览器资料
node scripts/browser_smoke.mjs --java "$JAVA"

# 只扫描源码候选，不读取仓库外运行时、模型或业务数据
uv run python scripts/check_release.py --source-only

# 实际构建后核验源码、wheel、sdist 与 wheel RECORD 一致性
uv build
uv run python scripts/check_release.py \
  --wheel dist/industrial_autonomous_diagnosis_agent-0.1.0-py3-none-any.whl \
  --sdist dist/industrial_autonomous_diagnosis_agent-0.1.0.tar.gz
```

`scripts/check_install.py` 必须复制到源码树外，在已实际安装 wheel 的独立新虚拟环境中用 `python -I` 执行，传入 `--forbid-root` 指定源码目录。它检查实际安装文件、入口、资源及进程内 HTTP，不启动模型或 Java。源码候选扫描是有界的词法检查，不是完整的机密检测或供应链审计。

`check_release.py` 只扫描约定的源码候选，不检查整个 Git 索引；提交前仍须复核所有待提交路径。运行时配置、令牌文件必须保存在仓库外，不要因为文件名未被 `.gitignore` 匹配就提交。快速运行的数据目录 `./.industrial-data` 已整体忽略；也可改用仓库外的全新目录。

## 不包含

真实客户/车辆/设备数据、生产模型或接口、PLC 写入、自动维修、隐式云端回退、后台开机守护、未授权开源许可声明。运行时模型/JVM、数据、令牌、日志均不应加入 Git。
