# Novel-factory

Novel-factory 是面向中文长篇小说写作的本地 Web 应用。它既能使用 `llama.cpp` 加载 GGUF 模型，也能临时连接 DeepSeek API。小说原稿、章节摘要、人物卡、场景卡和对话记录均保存在本机 SQLite 数据库中。

## 主要功能

- 流式续写、改写与多候选重生成，满意后再选用版本。
- TXT 小说资料库：自动识别编码、拆分章节、新建空白小说、编辑摘要和导出最新原稿。
- 每章一次模型调用生成完整章节摘要，并据此生成长期前文总览。
- 手动人物卡：通过专用表单维护人物资料，也可从指定章节范围的已有摘要中提炼。
- JSON 场景卡：规划下一章、决定是否加入结尾钩子，逐场景写作后可直接采用或统一润色。
- 章节局部重写：选取一段正文，加入指导意见，预览满意后再替换原文。
- 本地模式提供 40K / 80K 上下文；API 模式使用独立的上下文与输出预算，默认分别为 1M / 384K。
- 内置 MCP 服务与写作 skill，外部 Agent 操作时前端会同步显示结果。
- 应用与模型请求日志自动轮转，便于排查流式连接中断和生成错误。

## 快速开始

### macOS / Linux

首次使用：

```bash
./scripts/setup.sh
```

把 GGUF 模型放入 `model/`，并在 `.env` 中填写模型路径：

```text
MODEL_PATH=model/your-model.gguf
```

日常启动：

```bash
./scripts/start.sh
```

也可以安装全局启动命令：

```bash
./scripts/install-launcher.sh
novel
```

默认页面是 [http://127.0.0.1:8000](http://127.0.0.1:8000)。停止服务时在启动终端输入 `exit` 并回车，也可以按 `Control+C`。普通 HTTP 访问记录不会持续刷屏，详细运行信息保存在 `data/novel-factory.log`。

### DeepSeek API 模式

安装 `novel` 启动命令后运行：

```bash
novel --api
```

页面打开后，在右上角手动填写 DeepSeek API Key。Key 只保存在当前后端进程内，不写入数据库、配置文件、浏览器存储或日志；每次重启后都需要重新填写。

API 模式会把本次生成所需的提示词发送给 DeepSeek。完整资料库仍保存在本机。章节批量总结可能消耗大量 token，请先选择必要的章节范围。

### Windows

PowerShell 初始化并启动：

```powershell
.\scripts\setup.ps1
.\scripts\start.ps1
```

也可使用 `start.bat`，或通过 `scripts\install-launcher.ps1` 安装 `novel` 命令。Windows 本地模式需要自行准备 `llama-server.exe` 并在 `.env` 中设置 `LLAMA_SERVER_BIN`。

## 推荐写作流程

1. 打开“小说资料库”，导入现有 TXT，或点击“新建 TXT”。新文件会自动创建“第一章”。
2. 选择需要处理的起止章节，点击“总结全部章节”。每章只生成一份完整摘要，不再拆成编号分片。
3. 手动编辑长期背景、短期背景和章节摘要。API 模式下只总结真正需要的范围。
4. 在“核心人物卡”中手动新建人物，或按上方章节范围从已有摘要提炼人物卡。
5. 打开“场景编排器”，输入下一章方向，选择是否需要结尾钩子，多次生成并选用合适的 JSON 场景卡。
6. 在主窗口点击“启动”，执行逐场景写作、完成度检查和分场景审阅。
7. 必要时重写单个场景；正文已经满意可直接采用，也可以点击“继续润色”做整章连续性检查和最终润色。
8. 将满意版本加入现有章节或新章节，也可以在资料库中选择一段正文做局部重写。
9. 定期“导出最新 TXT”保存独立原稿备份。

## 小说资料库

支持 UTF-8、GB18030、Big5 和 UTF-16 TXT。章节标题可识别“第一章”“第 2 章”“序章”“番外”等常见格式。导入、新建、追加或局部重写后都可以导出为一份最新 TXT。

资料注入具有独立开关：

- 长期背景
- 短期背景
- 最近 1–5 章摘要
- 人物卡
- 已选场景卡

“当前提示词快照”可以查看真正发送给模型的资料。章节摘要注入时只使用摘要正文，不会直接塞入内部 JSON。

## 人物卡

人物卡不会在章节总结时自动生成。创建方式只有两种：

- 点击“手动新建”，在独立窗口中填写名称、身份、外貌、核心性格、行为习惯和世界观。
- 选择章节起止范围，点击“按章节范围提炼”。模型只读取这些章节已经生成的摘要。

人物卡页面只展示整理后的可读内容，不显示内部存储格式。提炼后也可以随时打开同一个表单修改。

## 场景编排器

场景卡使用结构化 JSON 保存，但写作时会转换为可读指令。单场景 `max_tokens` 上限为 3600。编排流程包括：

- 逐场景写作
- 完成度检查
- 不完整时续写当前场景
- 偏离时局部重写
- 分场景审阅与手动重生成
- 可选的整章连续性检查与最终润色

场景编号和标题只用于流程状态，不会写入最终小说正文。

## 创作参数

本地模式支持随机性（`temperature`）、采样范围（`top_p`）、最大输出、重复惩罚和固定 seed。

DeepSeek API 模式会发送随机性、采样范围和最大输出；固定 seed 与 `llama.cpp` 的重复惩罚不会发送。DeepSeek 建议随机性与采样范围主要调整其中一个。当前 API 客户端关闭思考模式，因此随机性和采样范围会生效。

## 外部 Agent / MCP

MCP 服务入口：

```bash
python agent/novel_factory_mcp.py
```

服务通过本机 HTTP API 操作正在运行的 Novel-factory，因此用户可以同时在前端看到 Agent 的改动。写作流程说明位于 `skills/novel-factory-writing/`，可安装到 Codex、WorkBuddy 等支持 MCP 与 skills 的 Agent 软件。

Agent 应优先查询项目、TXT、章节和当前提示词，再执行生成、总结、人物卡维护或局部重写；不要直接修改 SQLite 数据库。

## 常用配置

```text
MODEL_MODE=local
MODEL_PATH=model/your-model.gguf
LLAMA_SERVER_BIN=llama-server
N_CTX=40960
API_CONTEXT_SIZE=1000000
API_MAX_OUTPUT_TOKENS=384000
DEEPSEEK_MAX_RETRIES=2
DEEPSEEK_RETRY_BASE_SECONDS=5
DEEPSEEK_READ_TIMEOUT_SECONDS=90
N_GPU_LAYERS=auto
CACHE_TYPE_K=q8_0
CACHE_TYPE_V=q8_0
DATABASE_PATH=data/novel-factory.db
APP_LOG_MAX_BYTES=10485760
APP_LOG_BACKUP_COUNT=5
LLAMA_LOG_MAX_BYTES=5242880
LLAMA_LOG_BACKUP_COUNT=3
```

本地 80K 上下文需要更多内存，速度也可能下降。API 模式默认使用当前 DeepSeek 模型的 1M 上下文和 384K 最大输出能力；如所选模型或兼容服务限制更低，请在 `.env` 中下调 `API_CONTEXT_SIZE` 与 `API_MAX_OUTPUT_TOKENS`。已有安装的 `.env` 不会因更新代码而自动覆盖，需要手动修改。`novel --api` 只临时切换当前进程的运行模式，不会修改 `.env`。

当在线模型返回长度上限时，应用会将本次候选标记为未完成并保留已生成内容，不会把截断正文误报为正常完成。

DeepSeek 瞬时断连时会自动重试两次，默认等待约 5 秒、10 秒。尚未向页面输出正文的请求可以安全重试；已经输出部分正文的普通生成不会自动重放，避免重复拼接。场景完成度检查、隐藏续写和连续性检查会先缓存单次结果，失败重试时丢弃残缺结果。单次流连续 90 秒没有收到数据时会主动结束并进入重试，以上参数均可在 `.env` 中调整。

## 日志与故障排查

应用日志：

```text
data/novel-factory.log
```

本地模型日志：

```text
data/llama-server.log
```

查看最近 200 行应用日志：

```bash
tail -n 200 data/novel-factory.log
```

也可在本机访问 `/api/runtime/logs?lines=200`。日志记录请求编号、耗时、模型流开始/完成、HTTP 错误和浏览器断开事件，不记录 API Key 或提示词正文。

## 数据与隐私

- 数据库默认位于 `data/novel-factory.db`。
- 本地模式下正文和资料不会发送到在线模型服务。
- API 模式只发送当前请求所需上下文。
- 不要把 `.env`、API Key、模型文件或私人小说原稿提交到 Git。
- 删除资料库内容不可恢复，重要原稿请先导出备份。
