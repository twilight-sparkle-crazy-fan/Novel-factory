# Novel-factory

一个基于 `llama.cpp` 和 GGUF 模型的本地优先小说创作助手。名字致敬传奇开源项目 `llama-factory`：如果说 `llama-factory` 是模型工厂，那 Novel-factory 就是一条给个人作者用的小说生产线。

它不是普通聊天壳子，而是围绕“本地写小说”做的工具：续写、重生成抽卡、TXT 前文整理、人物卡、结构化伏笔/物品/地点/关系账本、下一章大纲和最新稿导出。

## 功能特点

- 完全本地运行：通过 `llama-server` 加载 GGUF 模型，正文和资料库默认不上传。
- 类 ChatGPT 的对话界面：流式输出、多候选重新生成、手动选用满意版本。
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
```

说明：

- `N_CTX=32768` 是默认上下文长度，界面里也可切换 32K / 64K。
- `CACHE_TYPE_K=q8_0` 和 `CACHE_TYPE_V=q8_0` 用于降低长上下文 KV cache 内存压力。
- `REASONING=off` 会尽量避免推理型模型把输出额度消耗在长思考上。
- 64K 上下文需要更多内存，速度也可能下降。

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
- 模型服务日志位于 `data/llama-server.log`，不主动记录完整聊天内容。

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
