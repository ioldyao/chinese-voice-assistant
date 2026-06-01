# 智能语音助手

<div align="center">

**中文语音助手 v2.7.0 - Pipecat 官方架构 + Agent Skills + WebSocket 实时语音合成**

双阶段语音识别 + 多 LLM 服务 + MCP 工具集成 + Agent Skills + 智能音频处理

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.7.0-green.svg)](https://github.com/yourusername/chinese-voice-assistant)

</div>

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd chinese-voice-assistant

# 安装依赖
uv sync
```

### 2️⃣ 配置 API Key

```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env，至少配置一个 LLM API Key
```

**推荐使用 Qwen**（有免费额度）：
```bash
LLM_SERVICE=qwen
QWEN_API_KEY=your-qwen-api-key-here
```

[获取 API Key →](https://dashscope.console.aliyun.com/)

### 3️⃣ 启动程序

```bash
uv run python main.py
```

### 4️⃣ 开始对话

说出唤醒词：
- "小智"
- "你好助手"
- "智能助手"

---

## 📚 文档导航

| 文档 | 说明 |
|------|------|
| **[快速开始指南](QUICKSTART.md)** | 5 分钟快速上手 |
| **[配置方案对比](docs/CONFIG_COMPARISON.md)** | 选择合适的配置方案 |
| **[完整配置文档](.env.example)** | 所有配置参数说明 |
| **[WebSocket TTS 指南](docs/QWEN_TTS_REALTIME.md)** | 实时语音合成详细说明 |
| **[TTS 迁移指南](docs/TTS_MIGRATION_GUIDE.md)** | 从 HTTP 切换到 WebSocket |

---

## ✨ 特性

### 🚀 核心功能
- **🎤 语音唤醒**:
  - **阶段1 - KWS**: 轻量级关键词检测（3.3MB），持续监听，CPU占用低
  - **阶段2 - ASR**: 唤醒后启动完整语音识别（120MB），准确率高
  - 支持自定义唤醒词（默认：小智、你好助手、智能助手）

- **🧠 多 LLM 服务支持**: 工厂模式，灵活切换
  - **Qwen** - 阿里云 DashScope（中文优化，Function Calling）
  - **DeepSeek** - DeepSeek API（强推理，低成本）
  - **OpenAI** - 官方 API（GPT-4o, o1 等）
  - **Anthropic Claude** - Claude Sonnet 4.5/Opus 4.6（支持 Thinking 模式）
  - 基于 Pipecat 官方框架（继承 OpenAILLMService）
  - **完全异步执行**，自动管理对话历史
  - **统一接口**，通过 `.env` 一键切换模型
  - 基于 MCP Python SDK 官方推荐模式

- **🎭 Playwright MCP**: 浏览器自动化操作
  - 网页导航、元素交互、截图、PDF生成等
  - 支持 Chrome/Firefox/Safari 浏览器控制
  - **完全异步**的工具调用（符合 MCP 官方最佳实践）

- **🔧 Agent Skills**: Claude Code 设计的技能系统
  - LLM 自主判断使用哪个技能（无需关键词匹配）
  - 技能描述注入到 system prompt
  - 统一的 `skill_execute` 函数接口
  - 支持自定义技能扩展
  - 零停用词、零关键词匹配、零干扰

- **🔊 语音合成**: 多引擎支持，低延迟流式播放 ⚡
  - **Piper TTS** - 本地超低延迟（<100ms，推荐开发）
  - **DashScope Realtime** - WebSocket 实时合成（~100-200ms，推荐生产）🆕
  - **DashScope HTTP** - HTTP 流式合成（~300ms，稳定可靠）
  - **Edge TTS** - 微软免费 API
  - **Azure TTS** - 高质量付费 API

- **👁️ 视觉理解**: 多模型支持（可配置切换）
  - **Moondream（本地）** - 完全离线，保护隐私
    - 硬件加速（CUDA/MPS/CPU）
    - 图片自动优化（缩放、格式转换）
    - 中英文智能提示
  - **Qwen-VL-Plus（API）** - 高精度识别
  - **Qwen-VL-Max（API）** - 最高精度
  - 通过 `.env` 一键切换模型
  - 统一接口，工厂模式设计
  - **完全异步化**（asyncio + PIL）

- **🔊 音频降噪**: RNNoise + soxr 高质量音频处理
  - **RNNoise** - 深度学习降噪（<5ms 延迟）
  - **soxr** - 高质量音频重采样
  - 可选降噪模式：rnnoise / noise_gate / pass_through
  - 仅影响 ASR，保持原始音频质量


- **🎧 音频设备配置**: 交互式设备选择
  - 启动时自动检测音频设备
  - 支持指定输入/输出设备
  - 避免立体声混音问题
  - 设备索引持久化保存

### 🎨 技术亮点（混合架构）
- ⚡ **多 LLM 服务工厂** - 支持 Qwen/DeepSeek/OpenAI/Anthropic 灵活切换✨
- 🎯 **官方 LLM Service** - 继承 OpenAILLMService（官方框架）✨
- 🔄 **自动对话管理** - LLMContextAggregator（官方框架）✨
- 🛠️ **Function Calling** - MCP 工具无缝集成（官方机制）✨
- 🧩 **Agent Skills 系统** - Claude Code 设计，LLM 自主判断✨
- 🚀 **保留自定义优势** - KWS + ASR + Piper TTS（本地、免费）
- 🛡️ **完全异步架构** - 纯异步，无线程开销
- 👁️ **多模型 Vision 系统** - 本地/云端模型可配置切换
- 🧠 **Smart Turn v3** - 智能对话完成检测（支持 23 种语言）✨
  - ✅ 理解语言上下文（语法、语调、语义）
  - ✅ 避免句子中间被打断
  - ✅ 本地 CPU 推理（<100ms 延迟）
- 🔇 **RNNoise 降噪** - 深度学习音频降噪（<5ms 延迟）✨
  - ✅ soxr 高质量重采样
  - ✅ 仅影响 ASR，保持原始音频
  - ✅ 可选降噪模式
  - ✅ 3分钟无活动自动结束
  - ✅ Pipeline 事件处理器
  - ✅ 可自定义超时时间
- 🎧 **音频设备配置** - 交互式设备选择✨
  - ✅ 启动时自动检测
  - ✅ 避免立体声混音
  - ✅ 设备索引持久化
- ⏸️ **标准中断机制** - 使用 Pipecat 官方 `InterruptionFrame`
  - ✅ 生态兼容：可与官方 TTS/LLM Processor 配合
  - ✅ 统一协调：`allow_interruptions` 全局管理
  - ✅ 事件明确：`TTSStoppedFrame` 通知停止状态

### 🏗️ 混合架构优势
**保留自定义（官方不支持）**：
- ✅ KWS 唤醒词检测（Sherpa-ONNX，本地）
- ✅ ASR 语音识别（Sherpa-ONNX，本地）
- ✅ Piper TTS（本地，超低延迟）
- ✅ 多模型 Vision（Moondream 本地 + Qwen-VL API，可切换）
- ✅ RNNoise 降噪（深度学习音频处理）
- ✅ Agent Skills 系统（Claude Code 设计）
- ✅ 音频设备配置（交互式选择）

**改用官方（享受生态）**：
- ✨ LLM Service（多服务支持：Qwen/DeepSeek/OpenAI/Anthropic）
- ✨ Context Aggregator（自动管理历史）
- ✨ Function Calling（MCP 工具集成）
- ✨ VAD + Smart Turn（Silero VAD + Smart Turn v3）

---

## 📦 安装

### 1. 环境要求
- Python 3.12+
- Windows 10/11
- 麦克风设备
- Node.js 18+（用于 Playwright MCP）

### 2. 克隆项目
```bash
git clone https://github.com/yourusername/voice-assistant.git
cd voice-assistant
```

### 3. 安装依赖
```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 4. 下载模型
```bash
# 下载 KWS + ASR + VAD 模型
python scripts/download_models.py

# 下载 Piper TTS 中文模型（推荐）
python download_piper_model.py
```

模型文件约 250MB，包括：
- **KWS 模型**（3.3MB）- Zipformer WenetSpeech（唤醒词检测）
- **ASR 模型**（120MB）- Paraformer 中文（语音识别）
- **Piper TTS 模型**（~50MB）- 中文语音合成（本地、超低延迟）
- **VAD 模型**（1MB）- Silero VAD（静音检测）

---

## 🔧 配置

### API Keys（使用 .env 文件）

1. 复制示例配置文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的 API 配置：
```bash
# ==================== LLM 配置 ====================
# 指定使用哪个 LLM 服务：qwen | deepseek | openai | anthropic
LLM_SERVICE=qwen

# Qwen (阿里云 DashScope) 配置组
QWEN_API_KEY=your-qwen-api-key-here
QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
# 可选模型：qwen-plus, qwen-max, qwen-turbo
# 本地部署示例：QWEN_API_URL=http://localhost:4000/v1, QWEN_MODEL=Local1-Qwen3-235B

# DeepSeek 配置组
# DEEPSEEK_API_KEY=your-deepseek-api-key-here
# DEEPSEEK_API_URL=https://api.deepseek.com/v1
# DEEPSEEK_MODEL=deepseek-chat
# 可选模型：deepseek-chat, deepseek-reasoner

# OpenAI 配置组
# OPENAI_API_KEY=your-openai-api-key-here
# OPENAI_API_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o
# 可选模型：gpt-4o, gpt-4, gpt-3.5-turbo, o1-preview, o1-mini

# Anthropic Claude 配置组
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
# ANTHROPIC_API_URL=https://api.anthropic.com
# ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
# 可选模型：claude-sonnet-4-5-20250929, claude-opus-4-6-20250514
# ANTHROPIC_ENABLE_THINKING=false  # 是否启用 Thinking 模式（Claude 扩展思考）
# ANTHROPIC_THINKING_EFFORT=medium  # Thinking 努力程度：high, medium, low

# ==================== Vision 服务配置 ====================
# 指定使用哪个 Vision 服务：moondream | qwen-vl-plus | qwen-vl-max
VISION_SERVICE=moondream
```

**重要提示**：
- ✅ `.env` 文件已被 `.gitignore` 忽略，不会被提交到 git
- ✅ 团队成员各自使用自己的 `.env` 配置
- ✅ `.env.example` 作为配置模板（已提交到 git）

获取 API Key：
- **Qwen**: [阿里云 DashScope](https://dashscope.console.aliyun.com/)
- **DeepSeek**: [DeepSeek 开放平台](https://platform.deepseek.com/)
- **OpenAI**: [OpenAI Platform](https://platform.openai.com/)

### 多 LLM 切换示例

#### 切换到 DeepSeek（强推理）
```bash
LLM_SERVICE=deepseek
DEEPSEEK_API_KEY=sk-xxxxx
DEEPSEEK_MODEL=deepseek-chat  # 或 deepseek-reasoner
```

#### 切换到 OpenAI（GPT-4o）
```bash
LLM_SERVICE=openai
OPENAI_API_KEY=sk-xxxxx
OPENAI_MODEL=gpt-4o  # 或 gpt-4, o1-preview
```

#### 切换回 Qwen（本地部署）
```bash
LLM_SERVICE=qwen
QWEN_API_KEY=your-key
QWEN_API_URL=http://localhost:4000/v1
QWEN_MODEL=Local1-Qwen3-235B
```

#### 切换到 Anthropic Claude（Sonnet 4.5）
```bash
LLM_SERVICE=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxx
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929  # 或 claude-opus-4-6-20250514
```

#### 启用 Claude Thinking 模式（扩展思考）
```bash
ANTHROPIC_ENABLE_THINKING=true
ANTHROPIC_THINKING_EFFORT=high  # high, medium, low
```

### 唤醒词配置
编辑 `config/keywords.txt`，使用以下格式：

```text
拼音音节(空格分隔) @中文
```

示例：
```text
x iǎo zh ì @小智
n ǐ h ǎo zh ù sh ǒu @你好助手
zh ì n éng zh ù sh ǒu @智能助手
```

### MCP Server 配置

本项目使用 **MCP (Model Context Protocol)** 协议集成各种工具能力（浏览器操作、系统控制等）。

#### 配置文件位置
```
config/mcp_servers.json
```

#### 配置文件结构
```json
{
  "servers": [
    {
      "name": "playwright",
      "description": "浏览器自动化",
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "timeout": 120,
      "enabled": true  // ← 设置为 true 启用
    },
    {
      "name": "windows",
      "description": "Windows 系统操作",
      "enabled": false  // ← 设置为 true 启用
    }
  ],
  "_comments": {
    "说明": "工具列表会在 Server 启动后通过 MCP 协议自动获取（session.list_tools()）"
  }
}
```

> **注意：** 工具列表由 MCP Server 运行时自动提供，无需在配置文件中指定。

#### 可用的 MCP Server

| Server | 功能 | 默认状态 |
|--------|------|---------|
| **playwright** | 浏览器自动化（导航、点击、输入、截图） | ✅ 启用 |
| **windows** | Windows 系统操作（鼠标、键盘、应用） | ⚪ 禁用 |
| **filesystem** | 文件系统操作（读取、写入、搜索） | ⚪ 禁用 |
| **github** | GitHub 操作（仓库管理、Issue、PR） | ⚪ 禁用 |

> **说明：** 工具列表在 Server 启动后自动获取，无需手动配置。

#### 启用/禁用 Server

编辑 `config/mcp_servers.json`，修改 `enabled` 字段：

```json
{
  "name": "windows",
  "enabled": true,  // false 改为 true
  ...
}
```

重启程序即可生效。

#### 添加自定义 Server

在配置文件中添加新条目：

```json
{
  "name": "custom-server",
  "description": "自定义功能描述",
  "command": "python",  // 或 npx, uvx 等
  "args": ["path/to/your/server.py"],
  "timeout": 60,
  "enabled": true,
  "env": {  // 可选：环境变量
    "API_KEY": "your-api-key"
  }
}
```

#### 配置优势

✅ **无需修改代码** - 通过 JSON 文件管理所有 Server
✅ **集中管理** - 所有配置在一个文件
✅ **灵活切换** - 通过 `enabled` 字段快速启用/禁用
✅ **向后兼容** - 配置文件不存在时自动使用默认配置
✅ **环境变量支持** - 敏感信息可通过 `env` 字段配置

### Vision 模型配置

本项目支持多种 Vision 模型，可通过 `.env` 配置一键切换：

**方案 1：Moondream 本地模型（推荐，隐私优先）**
```bash
VISION_SERVICE=moondream
MOONDREAM_USE_CPU=false  # 使用 GPU 加速（自动检测最佳设备）
```
- ✅ 完全本地化，无需 API 调用
- ✅ 完全离线，保护隐私
- ✅ 无 API 费用
- ⚠️ 首次运行会下载模型（~4GB）
- ⚠️ 对中文支持一般（自动翻译为英文提示）

**方案 2：Qwen-VL-Plus（高精度）**
```bash
VISION_SERVICE=qwen-vl-plus
QWEN_VL_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_VL_API_KEY=your-dashscope-api-key
```
- ✅ 高精度识别
- ✅ 原生中文支持
- ⚠️ 需要 API 调用费用

**方案 3：Qwen-VL-Max（最高精度）**
```bash
VISION_SERVICE=qwen-vl-max
QWEN_VL_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_VL_API_KEY=your-dashscope-api-key
```
- ✅ 最高精度识别
- ✅ 复杂场景理解能力强
- ⚠️ 需要 API 调用费用（比 Plus 稍贵）

**测试 Vision 模型：**
```bash
# 测试多模型切换
uv run test_vision_models.py
```

---

## 🚀 使用

### 启动助手
```bash
# 使用 uv（推荐）
uv run python main.py

# 或直接运行
python main.py
```

助手将自动启动，开始持续监听唤醒词。

### 交互流程
1. **唤醒**: 说出唤醒词（如"小智"）
2. **指令**: 听到提示音后，说出指令
3. **执行**: 系统自动理解并执行操作

### 支持的指令示例

#### 🌐 浏览器导航
```
"打开 B 站"
"访问百度"
"浏览器后退"
"刷新页面"
```

#### 🖱️ 网页交互
```
"点击搜索框"
"点击登录按钮"
"在输入框输入测试文本"
```

#### 📸 屏幕截图
```
"截取当前页面"
"保存页面为PDF"
```

#### 👁️ 视觉理解
```
"看看浏览器窗口显示了什么"
"分析当前屏幕内容"
```

---

## 📁 项目结构

```
chinese-voice-assistant/
├── src/voice_assistant/      # 核心源代码
│   ├── __init__.py           # 模块导出（延迟导入）
│   ├── config.py             # 配置管理（环境变量）
│   ├── wake_word.py          # 唤醒词系统
│   ├── mcp_client.py         # MCP 客户端（异步，多Server管理）
│   ├── llm_services.py       # LLM 服务工厂
│   │                         # - QwenLLMService, DeepSeekLLMService
│   │                         # - OpenAILLMServiceWrapper, AnthropicLLMService
│   │                         # - LLMFactory（工厂模式）
│   ├── qwen_llm_service.py   # MCP 工具转换器和函数注册
│   │                         # - MCP Tools → OpenAI 格式
│   │                         # - Function Calling 注册
│   │                         # - skill_execute, openmeteo_weather
│   ├── pipecat_main_v2.py    # Pipecat 主程序 v3.1
│   │                         # - 符合官方架构
│   │                         # - LLM 工厂模式
│   │                         # - Agent Skills 集成
│   ├── pyaudio_transport.py  # PyAudio Transport
│   │                         # - 标准 BaseTransport 实现
│   ├── audio_processors.py   # 音频处理器
│   │                         # - RNNoise 降噪
│   │                         # - soxr 高质量重采样
│   ├── audio_device_setup.py # 音频设备配置
│   │                         # - 交互式设备选择
│   ├── pipecat_adapters.py   # Pipecat Processors
│   │                         # - SherpaKWSProcessor (KWS)
│   │                         # - SherpaASRProcessor (ASR)
│   │                         # - VisionProcessor (Vision)
│   │                         # - PiperTTSProcessor (TTS)
│   ├── vision_services.py    # Vision 服务工厂
│   │                         # - Moondream, Qwen-VL
│   │                         # - VisionFactory（工厂模式）
│   ├── tts.py                # TTS 语音合成管理
│   ├── tts_realtime.py       # WebSocket 实时 TTS
│   ├── tts_realtime_adapter.py # Realtime TTS 适配器
│   ├── vision.py             # 视觉理解
│   └── skills/               # Agent Skills 系统
│       ├── base_skill.py     # 技能基类
│       ├── skill_loader.py   # 技能加载器
│       ├── skill_manager.py  # 技能管理器
│       └── skill_executor.py # 技能执行器
│
├── skills/                   # 技能目录
│   ├── browser/              # 浏览器技能
│   ├── calendar/             # 日历技能
│   └── weather/              # 天气技能
│
├── scripts/                  # 工具脚本
│   └── download_models.py    # 模型下载
│
├── tests/                    # 测试文件
│   └── test_pipecat_v2.py    # Pipecat v2 架构测试
│
├── docs/                     # 文档
│   ├── CONFIG_COMPARISON.md  # 配置方案对比
│   ├── QWEN_TTS_REALTIME.md  # WebSocket TTS 指南
│   └── TTS_MIGRATION_GUIDE.md # TTS 迁移指南
│
├── config/                   # 配置文件
│   ├── keywords.txt          # 唤醒词配置
│   └── mcp_servers.json      # MCP Server 配置
│
├── models/                   # 模型文件（需下载）
│   ├── piper/                # Piper TTS 模型
│   ├── sherpa-onnx-kws-*/    # KWS 模型 (3.3MB)
│   └── sherpa-onnx-paraformer-zh/ # ASR 模型 (120MB)
│
├── main.py                   # 主程序入口
├── pyproject.toml            # 项目配置 (v2.7.0)
└── README.md                 # 项目文档
```

### 代码统计
| 模块 | 代码行数 | 主要功能 |
|-----|---------|---------|
| `pipecat_main_v2.py` | 670 | Pipecat 主程序 v3.1（Pipeline 构建 + LLM 工厂） |
| `pipecat_adapters.py` | ~600 | Pipecat Processors（KWS/ASR/Vision/TTS） |
| `llm_services.py` | ~400 | LLM 服务工厂（Qwen/DeepSeek/OpenAI/Anthropic） |
| `qwen_llm_service.py` | ~500 | MCP 工具转换 + 函数注册 |
| `mcp_client.py` | ~400 | MCP 客户端（异步多 Server） |
| `tts.py` | ~350 | TTS 引擎管理（Piper/DashScope/Edge） |
| `tts_realtime.py` | ~300 | WebSocket 实时 TTS |
| `tts_realtime_adapter.py` | ~200 | Realtime TTS Pipecat 适配器 |
| `vision_services.py` | ~250 | Vision 服务工厂（多模型支持） |
| `pyaudio_transport.py` | ~330 | PyAudio Transport（VAD + Turn Detection） |
| `audio_processors.py` | ~300 | 音频处理器（RNNoise + soxr） |
| `audio_device_setup.py` | ~250 | 音频设备配置（交互式选择） |
| `vision.py` | ~150 | 视觉理解（异步） |
| `wake_word.py` | ~100 | 模型加载器（KWS + ASR） |
| `config.py` | ~225 | 全局配置（LLM + Vision + MCP + TTS） |
| `skills/` | ~500 | Agent Skills 系统（4 个模块） |
| `__init__.py` | 143 | 模块导出（延迟导入） |
| `main.py` | 49 | 主程序入口 |
| **总计** | **~5,800** | **v2.7.0 完整实现** |

---

## 🔧 开发

### 代码格式化
```bash
# 安装开发依赖
pip install -e ".[dev]"

# 格式化代码
black src/

# 代码检查
ruff check src/
```

### 架构说明

#### **Pipecat v3.1 架构（最新）**
```
Pipeline:
  PyAudioTransport.input() (音频输入 - 标准 BaseTransport + VAD + Turn Detection)
    ↓
  SherpaKWSProcessor (唤醒词检测 - 自定义)
    ↓
  NoiseReductionProcessor (RNNoise 降噪 - 自定义 ✨)
    ↓
  SherpaASRProcessor (语音识别 - 自定义，响应 VAD 事件)
    ↓
  UserAggregator (添加用户消息 - 官方 ✨)
    ↓
  VisionProcessor (视觉理解 - 自定义)
    ↓
  LLM Service (LLM + Function Calling - 官方 ✨)
    ↓
  PiperTTSProcessor (语音合成 - 自定义)
    ↓
  AssistantAggregator (保存助手响应 - 官方 ✨)
    ↓
  PyAudioTransport.output() (音频输出 - 标准 BaseTransport)
```

**关键特性**：
- ✅ Silero VAD（快速检测语音段，stop_secs=0.2）
- ✅ Smart Turn v3（智能判断对话完成，理解语言上下文）
- ✅ 避免句子中间被打断
- ✅ 支持 23 种语言
- ✅ **RNNoise 降噪**（深度学习音频处理，<5ms 延迟）✨
- ✅ **Agent Skills**（LLM 自主判断使用技能）✨
- ✅ **多 LLM 支持**（Qwen/DeepSeek/OpenAI/Anthropic）✨

### 核心改进

#### **1. 混合架构设计**
```python
# 保留自定义（官方不支持）
- KWS 唤醒词检测（Sherpa-ONNX）
- ASR 本地识别（Sherpa-ONNX）
- Piper TTS（本地、免费）
- Qwen Vision（保持现有 API）

# 改用官方（享受生态）
- QwenLLMService（继承 OpenAILLMService）
- LLMContextAggregator（自动管理历史）
- Function Calling（MCP 工具无缝集成）
```

#### **2. QwenLLMService 集成**
```python
# 初始化 Qwen LLM Service（完全兼容 OpenAI API）
llm = QwenLLMService(model="qwen-plus")

# 注册 MCP 函数处理器（统一调用所有 MCP 工具）
await register_mcp_functions(llm, mcp)

# 创建对话上下文（自动管理历史）
context = QwenLLMContext(messages, tools=tools_schema)
user_aggregator = LLMUserContextAggregator(context)
assistant_aggregator = LLMAssistantContextAggregator(context)
```

#### **3. 符合 Pipecat 官方最佳实践**
```python
# Pipeline 自动处理对话流程
Pipeline([
    ...,
    user_aggregator,      # 官方：自动添加用户消息
    llm,                  # 官方：LLM + Function Calling
    assistant_aggregator, # 官方：自动保存助手响应
    ...,
])
```

### 添加新功能
1. **添加新的 Pipecat Processor**: 在 `pipecat_adapters.py` 中继承 `FrameProcessor`
2. **添加新的 MCP 工具**: 工具会自动通过 Function Calling 集成到 LLM
3. **添加新的唤醒词**: 编辑 `config/keywords.txt`
4. **扩展 LLM Service**: 参考 `qwen_llm_service.py` 添加自定义功能

---

## 🛠️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| **语音识别** | | |
| 唤醒词检测 | Sherpa-ONNX (Zipformer) | 3.3MB 轻量级 KWS |
| 语音识别 | Sherpa-ONNX (Paraformer) | 120MB 中文 ASR |
| 静音检测 | Silero VAD | 1MB 语音活动检测 |
| 对话检测 | Smart Turn v3 | 智能判断对话完成（支持 23 种语言） |
| **语音合成** | | |
| 本地 TTS | Piper TTS | 超低延迟（推荐） |
| 云端 TTS | DashScope WebSocket | 实时流式播放 |
| 免费 TTS | Edge TTS | 微软免费 API |
| **智能决策** | | |
| LLM 框架 | Pipecat LLM Service | 官方框架 + 自动历史管理 |
| LLM 模型 | Qwen/DeepSeek/OpenAI/Claude | 多模型支持，工厂模式切换 |
| Function Calling | MCP Protocol | 工具无缝集成 |
| Agent Skills | Claude Code 设计 | LLM 自主判断技能使用 |
| **浏览器控制** | | |
| MCP 框架 | Model Context Protocol | 官方 Python SDK v1.25.0 |
| 浏览器自动化 | Playwright MCP | 跨浏览器支持 |
| **音频处理** | | |
| 实时框架 | Pipecat AI | Frame/Pipeline/Processor |
| 音频I/O | PyAudio | 录音播放 |
| 音频降噪 | RNNoise + soxr | 深度学习降噪 + 高质量重采样 |
| **视觉理解** | | |
| 多模态模型 | Qwen-VL-Max | 屏幕内容分析 |
| 截图工具 | PIL ImageGrab | 屏幕截图 |
| **其他** | | |
| Python 版本 | 3.12+ | 必需 |
| Node.js | 18+ | Playwright MCP 必需 |

---

## 📝 常见问题

### Q: 为什么识别不到唤醒词？
A:
- 检查麦克风是否正常工作
- 确认唤醒词是否在配置文件中（`config/keywords.txt`）
- 尝试提高音量，靠近麦克风说话
- 检查是否下载了 KWS 模型（3.3MB）

### Q: MCP 工具调用失败？
A:
- 确认已安装 Node.js（版本 18+）
- 检查 npx 命令是否可用：`npx --version`
- 手动测试 Playwright MCP：`npx @playwright/mcp@latest`
- 查看控制台错误信息

### Q: Pipecat 架构有什么优势？
A:
- ✅ 完全异步，无线程开销
- ✅ 符合 MCP Python SDK 官方最佳实践
- ✅ Pipeline 流式处理，更高效
- ✅ 代码简洁，易于维护
- ✅ 非阻塞执行，响应更快
- ✅ Vision 完全异步集成

---

## ⚠️ 注意事项

1. **API 费用**: 使用阿里云 API（LLM、Vision）会产生费用
   - 推荐使用 Piper TTS（免费本地）
   - Playwright 操作本地执行，无 API 费用

2. **隐私安全**:
   - API Key 不要提交到公开仓库
   - 建议使用环境变量管理敏感信息
   - 本地模型（Piper、Sherpa-ONNX）无隐私风险

3. **系统兼容**:
   - Playwright 支持跨平台
   - Pipecat 模式目前在 Windows 上测试

4. **网络需求**:
   - **无需网络**: KWS、ASR、Piper TTS（完全离线）
   - **需要网络**: LLM 决策、Vision 理解
   - **首次需要**: Playwright MCP 安装

---

## 🔥 最近更新

### v2.7.0 - Pipecat 官方架构 + Agent Skills（2026-04）

#### ✨ 核心特性
1. **Agent Skills 系统** - Claude Code 设计
   - LLM 自主判断使用哪个技能（无需关键词匹配）
   - 技能描述注入到 system prompt
   - 统一的 `skill_execute` 函数接口
   - 支持自定义技能扩展（browser、calendar、weather）
   - 零停用词、零关键词匹配、零干扰

2. **多 LLM 服务支持** - 工厂模式，灵活切换
   - **Qwen** - 阿里云 DashScope（中文优化）
   - **DeepSeek** - 强推理，低成本
   - **OpenAI** - GPT-4o, o1 等
   - **Anthropic Claude** - 支持 Thinking 模式

3. **WebSocket 实时语音合成** - 低延迟
   - DashScope Realtime TTS（~100-200ms 首包延迟）
   - 支持 server_commit 和 commit 两种模式
   - 符合 Pipecat Pipeline 标准

4. **RNNoise 音频降噪** - 深度学习降噪
   - RNNoise 深度学习降噪（<5ms 延迟）
   - soxr 高质量音频重采样
   - 可选降噪模式：rnnoise / noise_gate / pass_through

5. **音频设备配置** - 交互式设备选择
   - 启动时自动检测音频设备
   - 支持指定输入/输出设备
   - 设备索引持久化保存

#### 🔧 技术架构
```
Pipeline:
  Transport.input() → KWS → 降噪 → ASR → User Aggregator
  → Vision → LLM → TTS → Assistant Aggregator → Transport.output()
```

---

### v2.5.0 - MCP Server 配置文件化（2026-03）

#### ✨ 核心特性
- 新增 `config/mcp_servers.json` 配置文件
- 支持动态启用/禁用 Server
- 预配置 4 个常用 Server：playwright、windows、filesystem、github

---

### v2.4.0 - 多 LLM 服务支持（2026-03）

#### ✨ 核心特性
- 新增 `llm_services.py` 工厂模式
- 支持 Qwen/DeepSeek/OpenAI 服务切换
- 统一配置管理

---

## 🙏 致谢

- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) - 高性能语音识别框架
- [Playwright](https://playwright.dev/) - 强大的浏览器自动化工具
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP 官方 Python SDK
- [Pipecat AI](https://github.com/pipecat-ai/pipecat) - 实时音频处理框架
- [Piper TTS](https://github.com/rhasspy/piper) - 快速本地文本转语音引擎
- [阿里云 DashScope](https://dashscope.aliyun.com/) - 多模态 API 和 LLM 服务
- [Qwen](https://github.com/QwenLM/Qwen) - 强大的大语言模型和视觉模型

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

Made with ❤️

</div>
