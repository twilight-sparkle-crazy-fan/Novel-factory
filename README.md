# Novel-factory

一个基于 `llama.cpp` 和 GGUF 模型的本地优先小说创作助手。名字致敬传奇开源项目 `llama-factory`：如果说 `llama-factory` 是模型工厂，那 Novel-factory 就是一条给个人作者用的小说生产线。

它不是普通聊天壳子，而是围绕“本地写小说”做的工具：续写、重生成抽卡、TXT 前文整理、人物卡、结构化伏笔/物品/地点/关系账本、下一章大纲和最新稿导出。

## 功能特点

- 完全本地运行：通过 `llama-server` 加载 GGUF 模型，正文和资料库默认不上传。
- 类 ChatGPT 的对话界面：流式输出、多候选重新生成、手动选用满意版本。
- 长输出兜底：正文生成不足默认约 2000 token 时，会自动追加隐藏续写指令补足长度，目标值可在设置里调整。
- 词汇风格 / 词表白名单：可为单个对话设置措辞尺度、风格偏好和优先用词。
- 小说资料库：导入 TXT，自动拆章，按分片总结前文。
- 角色卡系统：人物观察独立生成，按角色 ID 与别名归一合并，减少“同一人多张卡”。
- 结构化事实账本：单独保存时间线事件、未回收伏笔、关键物品及持有人、地点状态、人物关系变化。
- 下一章大纲：支持像抽卡一样多次生成，默认不入库，手动保存/选用后才注入提示词。
- 增量写作：满意正文可加入现有章节或新章节，默认先保存，之后批量总结；也可立即更新摘要和人物卡。
- 提示词可视化：查看实际注入内容，排查隐藏资料泄漏。
- 多窗口隔离：不同窗口不会抢同一条对话上下文。

## 技术栈

- 推理引擎：`llama.cpp` / `llama-server`
- 模型格式：GGUF
- 后端：FastAPI + SQLite
- 前端：原生 HTML / CSS / JavaScript
- 默认地址：[http://127.0.0.1:8000](http://127.0.0.1:8000)

## 平台支持

| 平台 | 状态 | 说明 |
| --- | --- | --- |
| macOS | 推荐 | 支持 `scripts/setup.sh`、`scripts/start.sh` 和 `start.command` |
| Linux | 可用 | 需要自行安装 `llama-server` |
| Windows WSL2 | 推荐 | 基本按 Linux 路线运行 |
| Windows 原生 | 可用 | 提供 PowerShell 脚本，需要自行准备 Windows 版 `llama-server.exe` |

项目主体是 Python + SQLite + Web 前端，跨平台问题主要集中在 `llama.cpp` 安装和启动脚本上。

## 准备模型和 llama.cpp

1. 安装 `llama.cpp`，确保命令行能找到 `llama-server`。
2. 下载 GGUF 模型，放到 `model/` 目录。
3. 第一次运行会生成 `.env`，请把 `MODEL_PATH` 改成你的模型路径，例如：

```text
MODEL_PATH=model/your-model.gguf
```

如果 `llama-server` 不在 PATH，可以在 `.env` 中指定绝对路径：

```text
LLAMA_SERVER_BIN=/path/to/llama-server
```

Windows 示例：

```text
LLAMA_SERVER_BIN=C:\tools\llama.cpp\llama-server.exe
```

模型文件通常很大，仓库默认不会提交 `model/*.gguf`。

## macOS / Linux 运行

在项目目录执行：

```bash
./scripts/setup.sh
./scripts/start.sh
```

如果想像 Claude 一样用一个单词启动，可以安装 `novel` 命令：

```bash
./scripts/install-launcher.sh
novel
```

之后在任意终端输入 `novel`，会自动启动 Novel-factory 并打开浏览器。

macOS 也可以双击：

```text
start.command
```

如果 8000 端口被占用，可以临时换端口：

```bash
APP_PORT=8001 ./scripts/start.sh
```

停止应用时，在启动终端按 `Control+C`。应用会尝试一并清理自己启动的 `llama-server`。

## Windows 原生运行

先打开 PowerShell，进入项目目录：

```powershell
cd C:\path\to\Novel-factory
```

初始化：

```powershell
.\scripts\setup.ps1
```

启动：

```powershell
.\scripts\start.ps1
```

如果想安装一词启动命令：

```powershell
.\scripts\install-launcher.ps1
novel
```

也可以双击或运行：

```bat
start.bat
```

如果 PowerShell 阻止脚本执行，可以临时允许当前窗口运行脚本：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

如果端口被占用，可以临时换端口：

```powershell
$env:APP_PORT=8001
.\scripts\start.ps1
```

## 常用配置

第一次运行 `setup` 脚本会从 `.env.example` 创建 `.env`。常用项：

```text
MODEL_PATH=model/your-model.gguf
LLAMA_SERVER_BIN=llama-server
N_CTX=32768
CACHE_TYPE_K=q8_0
CACHE_TYPE_V=q8_0
N_GPU_LAYERS=auto
REASONING=off
MAX_CANDIDATES_PER_EXCHANGE=20
DATABASE_PATH=data/novel-factory.db
LLAMA_LOG_MAX_BYTES=5242880
LLAMA_LOG_BACKUP_COUNT=3
EXPERIMENTAL_MATERIAL_SYSTEM=false
```

说明：

- `N_CTX=32768` 是默认上下文长度，界面里也可切换 32K / 64K。
- `CACHE_TYPE_K=q8_0` 和 `CACHE_TYPE_V=q8_0` 用于降低长上下文 KV cache 内存压力。
- `REASONING=off` 会尽量避免推理型模型把输出额度消耗在长思考上。
- 64K 上下文需要更多内存，速度也可能下降。
- `LLAMA_LOG_MAX_BYTES` / `LLAMA_LOG_BACKUP_COUNT` 控制 `data/llama-server.log` 轮转，避免日志无限增长。
- `EXPERIMENTAL_MATERIAL_SYSTEM=true` 会启用实验性的 `.llm4pkg` 分析包 API，默认关闭。

## 基本使用

### 对话与重生成

每条模型回复下方都有“重新生成”按钮：

1. 点击“重新生成”会创建新候选，旧文本不会被覆盖。
2. 用左右按钮查看所有候选版本。
3. 点击“选用此版本”，该版本才会进入下一轮上下文。
4. 如果改选较早的回复，应用会提示创建分支，原对话保持不变。

### 导入小说前文

点击左下角“小说资料库”，选择 TXT 原稿：

1. 自动识别 UTF-8、GB18030、Big5 和 UTF-16。
2. 按“第一章”“序章”“番外”等标题拆章。
3. 可选择从第几章处理到第几章。
4. 长章节会按分片处理，每个分片完成后保存断点。
5. 可暂停、断点续行，也可批量处理待总结章节。

### 资料库注入

资料库中的信息不会无脑全部塞进提示词。当前支持独立开关：

- 前文总览
- 最近章节摘要
- 人物卡
- 结构化事实
- 下一章大纲

可以点击“查看实际注入内容”检查本轮到底带了哪些资料。

### 词汇风格和长输出

在“设置 → 写作指令”里可以填写“词汇风格”和“词表白名单 / 优先用词”，也可以从冷峻悬疑、古风雅致、对话口语、直白强表达等内置模板起步。这两项会作为当前对话的高优先级写作约束注入提示词，适合放文风尺度、专有称谓、固定表达和不希望被模型委婉替换的词。

正文生成时，如果模型提前结束且完成量不足设置里的“自动续写目标 token”，应用会自动补一轮隐藏续写指令，让模型从最后一句自然接着写。这个兜底只影响正文生成，不会让抽大纲候选自动入库；目标设为 0 可关闭长度兜底。

### 人物卡与结构化事实

人物卡不会和章节摘要抢同一次输出额度。系统会先从正文分片中提取人物观察，再合并成人物卡。新增正文立即总结时，只会更新新增观察涉及的人物。

结构化事实会单独保存：

- 时间线事件
- 未回收伏笔
- 关键物品及持有人
- 地点状态
- 人物关系变化

每条事实保留来源章节、首次出现章节和最近更新章节。

### 增量写作和导出

满意的正文可以加入当前 TXT 的现有章节或新建章节。默认只保存正文并标记为待总结，不会立刻占用模型；需要时也可以勾选“加入后立即总结”。

在“小说资料库”中点击“导出最新 TXT”，会按当前选中 TXT 的章节顺序导出最新稿件。

## 数据与隐私

- 对话、资料库、章节摘要、人物卡保存在 SQLite 数据库中。
- 默认数据库路径：`data/novel-factory.db`。
- 模型文件、数据库、日志、`.env` 都被 `.gitignore` 排除。
- 模型服务日志位于 `data/llama-server.log`，会按大小轮转，默认保留 3 个备份；不主动记录完整聊天内容。

### 实验性分析包

打开 `EXPERIMENTAL_MATERIAL_SYSTEM=true` 后，会启用 `/api/experimental/material-system` 下的实验 API，并在小说资料库里显示“导出分析包 / 导入分析包 / 重建实验资料 / 提示词预算 / 当前快照 / 预算设置 / 确认队列”。当前支持导出 `.llm4pkg`、导出前报告、导入前校验、纯新文档导入、合并到当前 TXT、替换当前实验资料层，并可按资料层选择只导入语义观察、时间线、人物/关系、确认队列、辅助账本或预算配置；合并导入还可填写章节范围，只导入该范围内有明确来源的资料记录。系统能把现有章节摘要、人物卡和结构化事实投影为实验时间线、人物实体、人物关系、确认队列、辅助账本和提示词预算报告。开启实验开关后，章节分析还会额外抽取统一事件并写入实验事件账本，确认队列可创建缺失人物实体并把事件或关系写回实验资料层，也会提示弱别名、人物合并候选、人物事实冲突和关系覆盖风险，位置、物件和悬念类辅助观察可从队列写入辅助账本和时间线，能力观察可写入人物经历；确认队列可按状态/类型筛选，支持勾选后批量确认普通项或批量忽略，人物事实冲突可确认后用新事实覆盖冲突旧事实，关系覆盖冲突可接受新关系并把旧冲突关系标记为 superseded；实验资料视图会显示语义观察账本，可按类型/状态筛选，并可把观察标记为 active/resolved/disabled；时间线节点会按全书、章节组、章节和手工父子节点树形展示，并可调整父节点、章节范围和排序；可新建、编辑、删除时间线节点、时间线事件、人物实体、人物事实、人物阶段档案、人物经历事件、关系边、关系事件和地点/物件/悬念辅助账本，时间线事件可维护章节、顺序和参与者，人物经历事件、关系事件和辅助账本可维护章节与顺序，人物事实和人物阶段档案可设置适用章节范围；支持人物别名、人物合并和人物拆分，拆分时可选择把关系边及其关系事件迁移到新人物；关系网络概览会统计并可视化人物节点、关系边、关系事件和核心人物，可查询单个人物的全部关系、两个人物之间的关系历史，也可按章节查看当时已有关系事件快照；删除人物前会检查关系引用并显示阻塞明细。人物事实、近期经历、人物关系历史和辅助账本会进入提示词预算段，当前快照可单独预览实际会注入的资料文本，单个人物也可按章节查看当时适用的阶段档案、事实和经历快照。正式生成会使用实验提示词预算器产出的资料段，预算设置会按分段裁剪资料注入，预算预览会显示裁剪前后 token 和剩余额度。导入前校验会显示包内文件 hash、documents 必填字段、documents 未知字段、资料层记录数、资料必填字段、资料未知字段、JSONL 实际记录数、章节/chunk 数量、章节/chunk 必填字段、章节/chunk 未知字段、document_id 和资料 document_id 一致性、章节/chunk 内容 hash、provenance 来源 hash、资料 provenance 引用、资料内部引用、目标 TXT hash 对照、本地相同原文匹配和记录级差异预览；合并导入会保留已人工编辑/确认的人物、时间线节点、时间线事件、关系边和辅助账本字段，把包内字段差异写入确认队列，并可在队列中勾选具体字段后应用包内冲突值。

建议仍然定期把重要小说原稿保存到独立文件中。

## 测试

```bash
AUTO_START_LLAMA=false .venv/bin/python -m pytest
```

Windows：

```powershell
$env:AUTO_START_LLAMA="false"
.\.venv\Scripts\python.exe -m pytest
```

## 常见问题

### 页面一直显示“正在加载模型”

查看日志：

```bash
tail -n 80 data/llama-server.log
```

Windows PowerShell：

```powershell
Get-Content .\data\llama-server.log -Tail 80
```

请确认：

- `MODEL_PATH` 指向真实存在的 GGUF 文件。
- `LLAMA_SERVER_BIN` 能找到 `llama-server` / `llama-server.exe`。
- 8080 端口没有被其他程序占用。
- 当前机器内存足够加载模型和上下文。

### 生成内容为空或很短

部分推理模型会先输出很长 reasoning。项目默认 `REASONING=off`。如果你改成 `auto`，请提高最大输出 token。

### Windows 上找不到 llama-server.exe

请下载或自行编译 Windows 版 `llama.cpp`，然后在 `.env` 中设置：

```text
LLAMA_SERVER_BIN=C:\path\to\llama-server.exe
```

### 可以上传模型和数据库吗？

不建议。仓库默认忽略：

- `model/*.gguf`
- `data/*.db`
- `data/*.log`
- `.env`

这些文件通常包含大模型、个人创作内容或本机配置。
