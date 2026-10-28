# 本地运行时

## 明确选择提供器

`--runtime-config` 使用本工程的 ManagedLocalProvider，校验明确配置的 llama.cpp 可执行文件、邻近动态库及生成模型的 SHA256。同一次诊断图运行使用显式 `session()` 范围：第一次模型调用才启动 CPU 子进程，后续调用复用进程和引擎可用的输入前缀缓存；人工暂停、图结束或异常时回收，不跨任务或人工暂停保留进程。没有 session 的独立调用（包括提交后报告）仍各自启动并回收进程。线程锁和操作系统锁保证排他；只加载 generation，不加载 embedding/rerank。缓存仅加速输入处理，不减少每次请求实际报告的 total_tokens 记账。

`--model-url` 使用已运行的数值 loopback HTTP 端点。拒绝域名（包括 localhost）、非 loopback、URL 凭据、查询、fragment、重定向和环境代理。此模式不证明远端进程加载了预期权重；需要调用者自己配置可信的本地服务。

`--test-mode` 是明确的确定性测试提供器，不是模型。不允许未配置时自动切换此模式。

配置 JSON 结构如下（所有 path 必须绝对路径；hash/bytes 必须来自实际核验，不可复制占位符）：

```json
{
  "binary": {"path": "/absolute/runtime/bin/llama-server", "sha256": "实际64位小写SHA256"},
  "libraries": [{"path": "/absolute/runtime/bin/libllama.dylib", "sha256": "实际64位小写SHA256"}],
  "models": {
    "generation": {"path": "/absolute/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "sha256": "实际64位小写SHA256", "bytes": 2497281120}
  },
  "lock_path": "/absolute/runtime/llama.lock",
  "port": 18080
}
```

这是结构示例，不是可直接使用的有效配置。libraries 必须列全所用邻近引擎库；同目录位置、文件 hash 和非符号链接身份均受检查。本地可信配置不是恶意管理员隔离边界。也接受同形状的 generation/embedding/rerank 三角色配置，后两者不会读取或运行。

不读取 shell profile、云凭据或其他项目模型。不自动下载/自动更新，不使用全局代理环境，无推理重试。

## 本次参考引擎及模型

- llama.cpp `b6500`，commit `a7a98e0fffed794396b3fbad4dcdbbc184963645`，MIT。
- 原模型 `Qwen/Qwen3-4B-Instruct-2507`；GGUF 分发 `unsloth/Qwen3-4B-Instruct-2507-GGUF`。
- revision `0b0406b39725d752255ffeb48f102c66f98e14aa`。
- 文件 `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`，2,497,281,120 bytes。
- SHA256 `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`，Apache-2.0。
- 显式 CPU，4 threads，8192 context，temperature 0，seed 42，无 Metal / GPU 隐式回退。

复用已在仓库外校验的同一公开权重和引擎，不将其提交。此前本机 Metal 预热失败，因此明确选 CPU；CPU 不是云端兜底。当前引擎编译未启用 Accelerate/BLAS；固定输入前缀和运行范围内复用减少重复 CPU 处理，不代表生产吞吐能力，也不保证所有案例都在预算内成功。

## Java / HermiT

使用显式 Java 17 可执行文件，不修改系统 Java 注册、shell profile 或开机服务。

参考平台 macOS 13.2 / Apple M1 arm64 已验证便携 Azul Zulu JRE：

- Zulu `17.68+203-CA`，OpenJDK `17.0.20.1+1-LTS`。
- 源：`https://cdn.azul.com/zulu/bin/zulu17.68.203-ca-jre17.0.20.1-macosx_aarch64.tar.gz`。
- SHA256 `2348b7bdcad558d218700f60487b92e63b51f063b1fb44143ca56dd0a06c7ca6`。
- 发布校验元数据：`https://api.azul.com/metadata/v1/zulu/packages/9c335676-e4a8-449a-83b1-d775a8062867`。
- 实际归档 42,727,477 bytes；元数据 size 为 42,727,500，不能称 size 一致。发布 SHA256 校验和实际执行是核验依据。
- GPLv2 + Classpath Exception；保留包内第三方 notices。

owlready2 `0.51` 附带 HermiT `1.3.8.1099`，JAR SHA256 `97e53f519720a0a2e328556e03b806aa239347c85e57637a935e6433ed2ad3a8`。仅安装依赖不等于推理成功，必须运行实际 Java 检查。

```bash
export JAVA=/absolute/path/to/java
INDUSTRIAL_DIAGNOSIS_JAVA="$JAVA" uv run pytest tests/test_knowledge.py tests/test_knowledge_regressions.py
```

## 文件与安全

数据目录、runtime 配置、模型、Java、数据库、日志、令牌不入 Git。CLI 一次性令牌输出不要重定向到提交文件。浏览器固定同源 API，后端仅 loopback，无 CORS 开放。

生产化需要额外设计多用户部署、身份集成、数据库权限、日志/保留策略、受控模型服务、签名发布、可观测性和现场验证；本工程未宣称已经完成。
