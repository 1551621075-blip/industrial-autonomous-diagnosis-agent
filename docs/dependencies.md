# 依赖与许可边界

版本和 hash 以 `uv.lock` 为准。构建时使用 Python 3.12.0 arm64、uv 0.11.26，公开阿里 PyPI 镜像；锁文件保留实际来源和 hash，不伪称从其他索引下载。

| 组件 | 本次安装版本 | 已核查许可元数据 |
|---|---|---|
| LangGraph | 1.2.5 | MIT |
| FastAPI | 0.162.0 | MIT |
| Pydantic | 2.14.1 | MIT |
| Typer | 0.27.3 | MIT |
| HTTPX | 0.28.1 | BSD-3-Clause |
| Uvicorn | 0.54.0 | BSD-3-Clause |
| PyYAML | 6.0.3 | MIT |
| NumPy | 2.5.4 | BSD-3-Clause；分发元数据同时列 0BSD/MIT/Zlib/CC0-1.0 |
| Owlready2 | 0.51 | LGPL-3.0-or-later |
| pytest | 9.2.0 | MIT |
| Ruff | 0.17.3 | MIT |

Owlready2 附带 HermiT 及 Java 依赖，不能把整套运行时统称为 MIT。HermiT 的许可证及附带组件 notices 应按实际分发包核查和保留；本项目只声明依赖，不把 JAR/JRE 或第三方源码拷入仓库。Java、模型、llama.cpp 的固定来源、hash 和许可见 [运行时说明](runtime.md)。

`langsmith` 是 LangGraph/langchain-core 的传递依赖；图执行显式 `tracing_context(enabled=False)`、空 callbacks，不启用远程追踪，不把材料发送到第三方服务。LLM HTTP 禁用环境代理和重定向，JVM 使用受控本地本体。

此表不是完整法律意见或全供应链安全审计。最终交付需要再次扫描实际提交文件、包内容和许可证事实。

**没有为用户的整体原创工程擅自添加 MIT/Apache 等开源许可。** 私有上传不等于允许公开再分发，第三方依赖条款独立适用。

## 浏览器验证依赖

工作台本身不依赖外部 CDN 或 JavaScript 包。实际浏览器验收使用 Node 内建 WebSocket/CDP 与本机独立 Chrome profile，不使用用户真实 profile/cookie。参考环境 macOS 13.2，Node 24.18.0，Chrome 155；脚本平台前提以其 CLI/源码及最终验收记录为准，不称跨操作系统浏览器矩阵已经验证。
