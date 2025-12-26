# 智能语音助手

<div align="center">

**双阶段语音识别 + 多 LLM 服务 + Playwright 浏览器控制 + 实时音频处理**

基于 Sherpa-ONNX + Pipecat LLM Service + Playwright MCP + Piper + Pipecat 的中文语音助手

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.4.0-green.svg)](https://github.com/yourusername/chinese-voice-assistant)

</div>

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
  - 基于 Pipecat 官方框架（继承 OpenAILLMService）
  - **完全异步执行**，自动管理对话历史
  - **统一接口**，通过 `.env` 一键切换模型
  - 基于 MCP Python SDK 官方推荐模式

- **🎭 Playwright MCP**: 浏览器自动化操作
  - 网页导航、元素交互、截图、PDF生成等
  - 支持 Chrome/Firefox/Safari 浏览器控制
  - **完全异步**的工具调用（符合 MCP 官方最佳实践）

- **🔊 语音合成**: 多引擎支持
  - **Piper TTS** - 本地超低延迟（推荐）⚡
  - **RealtimeTTS** - 流式实时播放 🎵
  - **MeloTTS** - 中英文混合支持 🌐

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

### 🎨 技术亮点（混合架构）
- ⚡ **多 LLM 服务工厂** - 支持 Qwen/DeepSeek/OpenAI 灵活切换✨
- 🎯 **官方 LLM Service** - 继承 OpenAILLMService（官方框架）✨
- 🔄 **自动对话管理** - LLMContextAggregator（官方框架）✨
- 🛠️ **Function Calling** - MCP 工具无缝集成（官方机制）✨
- 🚀 **保留自定义优势** - KWS + ASR + Piper TTS（本地、免费）
- 🛡️ **完全异步架构** - 纯异步，无线程开销
- 👁️ **多模型 Vision 系统** - 本地/云端模型可配置切换
- 🧠 **Smart Turn v3** - 智能对话完成检测（支持 23 种语言）✨
  - ✅ 理解语言上下文（语法、语调、语义）
  - ✅ 避免句子中间被打断
  - ✅ 本地 CPU 推理（<100ms 延迟）
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

**改用官方（享受生态）**：
- ✨ LLM Service（多服务支持：Qwen/DeepSeek/OpenAI）
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
uv pip install -e .

# 安装 Smart Turn v3 依赖（智能对话检测）
uv pip install "pipecat-ai[local-smart-turn-v3]"

# 或使用 pip
pip install -e .
pip install "pipecat-ai[local-smart-turn-v3]"
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
# 指定使用哪个 LLM 服务：qwen | deepseek | openai
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
# 方式1：使用主入口
python main.py

# 方式2：直接运行 Pipecat 主程序（推荐）
uv run python -m src.voice_assistant.pipecat_main_v2
```

助手将自动启动 Pipecat 模式，开始持续监听唤醒词。

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
│   ├── __init__.py           # 模块导出 (40行)
│   ├── config.py             # 配置管理 (55行)
│   ├── wake_word.py          # 模型加载器 (95行)
│   ├── react_agent.py        # React 智能代理 (603行)
│   │                         # - 完全异步执行（Pipecat 模式）
│   ├── mcp_client.py         # MCP 客户端 (378行)
│   │                         # - MCPManager (异步，多Server管理)
│   ├── llm_services.py       # LLM 服务工厂 (280行)
│   │                         # - QwenLLMService, DeepSeekLLMService, OpenAILLMServiceWrapper
│   │                         # - LLMFactory (工厂模式)
│   ├── qwen_llm_service.py   # MCP 工具转换器 (180行)
│   │                         # - MCP Tools → OpenAI 格式
│   │                         # - Function Calling 注册
│   ├── pipecat_main_v2.py    # Pipecat 主程序 v2 (365行)
│   │                         # - 符合官方架构（BaseTransport + CancelFrame）
│   │                         # - 使用 LLM 工厂模式
│   ├── pyaudio_transport.py  # PyAudio Transport (201行)
│   │                         # - 标准 BaseTransport 实现
│   ├── vad_processor.py      # VAD Processor (63行)
│   │                         # - Silero VAD 集成（开发中）
│   ├── pipecat_adapters.py   # Pipecat Processors (676行)
│   │                         # - SherpaKWSProcessor (KWS)
│   │                         # - SherpaASRProcessor (ASR + 临时 RMS VAD)
│   │                         # - VisionProcessor (Vision)
│   │                         # - PiperTTSProcessor (TTS)
│   ├── vision_services.py    # Vision 服务工厂 (233行)
│   │                         # - MoondreamVisionService, QwenVLVisionService
│   │                         # - VisionFactory (工厂模式)
│   ├── tts.py                # TTS 语音合成 (372行)
│   └── vision.py             # 视觉理解 (136行)
│
├── scripts/                  # 工具脚本
│   ├── download_models.py    # 模型下载
│   └── pinyin_helper.py      # 拼音转换助手
│
├── tests/                    # 测试文件
│   ├── test_pipecat_v2.py    # Pipecat v2 架构测试
│   └── test_interruption.py  # 中断机制测试
│
├── docs/                     # 文档
│   ├── vad-optimization-summary.md    # VAD 优化总结
│   ├── pipecat-migration-guide.md    # v1.0 → v2.0 迁移指南
│   ├── pipecat-changes-comparison.md # 详细代码对比
│   └── interruption-analysis.md      # 中断机制分析
│
├── config/                   # 配置文件
│   └── keywords.txt          # 唤醒词配置
│
├── models/                   # 模型文件（需下载）
│   ├── piper/                # Piper TTS 模型
│   ├── sherpa-onnx-kws-*/    # KWS 模型 (3.3MB)
│   └── sherpa-onnx-paraformer-zh/ # ASR 模型 (120MB)
│
├── main.py                   # 主程序入口 (35行)
├── pyproject.toml            # 项目配置 (v2.4.0)
└── README.md                 # 项目文档
```

### 代码统计
| 模块 | 代码行数 | 主要功能 |
|-----|---------|---------|
| `pipecat_adapters.py` | 676 | Pipecat Processors（KWS/ASR+VAD/Vision/TTS） |
| `react_agent.py` | 603 | React 推理框架（完全异步） |
| `mcp_client.py` | 378 | MCP 客户端（异步多 Server） |
| `tts.py` | 372 | TTS 引擎管理（Piper/RealtimeTTS） |
| `pipecat_main_v2.py` | 365 | Pipecat Pipeline v2（LLM 工厂） |
| `llm_services.py` | 280 | LLM 服务工厂（Qwen/DeepSeek/OpenAI） |
| `vision_services.py` | 233 | Vision 服务工厂（多模型支持） |
| `pyaudio_transport.py` | 201 | 标准 PyAudio Transport |
| `qwen_llm_service.py` | 180 | MCP 工具转换器 + 函数注册 |
| `vision.py` | 136 | Qwen-VL-Max 视觉理解（异步） |
| `wake_word.py` | 95 | 模型加载器（KWS + ASR） |
| `vad_processor.py` | 63 | Silero VAD Processor（开发中） |
| `config.py` | 55 | 全局配置（LLM + Vision） |
| `__init__.py` | 40 | 模块导出 |
| `main.py` | 35 | Pipecat 单一入口 |
| **总计** | **~3,682** | **v2.4.0 完整实现** |

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

#### **Pipecat v2.2.1 架构**
```
Pipeline:
  PyAudioTransport.input() (音频输入 - 标准 BaseTransport + VAD + Turn Detection)
    ↓
  SherpaKWSProcessor (唤醒词检测 - 自定义)
    ↓
  SherpaASRProcessor (语音识别 - 自定义，响应 VAD 事件)
    ↓
  OpenAIUserContextAggregator (添加用户消息 - 官方 ✨)
    ↓
  VisionProcessor (视觉理解 - 自定义)
    ↓
  QwenLLMService (LLM + Function Calling - 官方 ✨)
    ↓
  PiperTTSProcessor (语音合成 - 自定义) ← ✨ v2.2.1 调整：提前到这里
    ↓
  OpenAIAssistantContextAggregator (保存助手响应 - 官方 ✨)
    ↓
  PyAudioTransport.output() (音频输出 - 标准 BaseTransport)
```

**关键特性**：
- ✅ Silero VAD（快速检测语音段，stop_secs=0.2）
- ✅ Smart Turn v3（智能判断对话完成，理解语言上下文）
- ✅ 避免句子中间被打断
- ✅ 支持 23 种语言
- ✅ **v2.2.1 修复**：TTS 在 aggregator 之前，确保接收 LLM 输出

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
| 流式 TTS | RealtimeTTS | 实时流式播放 |
| 混合 TTS | MeloTTS | 中英文支持 |
| **智能决策** | | |
| LLM 框架 | Pipecat LLM Service | 官方框架 + 自动历史管理 |
| LLM 模型 | Qwen-Plus | 意图理解+规划 |
| Function Calling | MCP Protocol | 工具无缝集成 |
| **浏览器控制** | | |
| MCP 框架 | Model Context Protocol | 官方 Python SDK v1.25.0 |
| 浏览器自动化 | Playwright MCP | 跨浏览器支持 |
| **音频处理** | | |
| 实时框架 | Pipecat AI v0.0.98 | Frame/Pipeline/Processor |
| 音频I/O | PyAudio | 录音播放 |
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

### v2.4.0 - 多 LLM 服务支持 + 工厂模式（2025-12-26）

#### ✨ 核心特性
1. **LLM 服务工厂架构** - 类似 Vision 的灵活设计
   - 新增 `llm_services.py` 模块（~280 行）
   - 支持 **Qwen**（阿里云 DashScope）
   - 支持 **DeepSeek**（强推理，低成本）
   - 支持 **OpenAI**（GPT-4o, o1 等）
   - 统一接口，工厂模式创建

2. **配置分组管理** - 清晰易读的配置
   - `LLM_SERVICE` 选择器（qwen/deepseek/openai）
   - 每个服务独立配置组
   - 向后兼容（保留 `DASHSCOPE_*` 变量名）
   - `.env.example` 清晰示例

3. **变量名语义化** - 不再硬编码 Qwen
   - `QWEN_API_KEY` 替代 `DASHSCOPE_API_KEY`
   - `DEEPSEEK_API_KEY`、`OPENAI_API_KEY` 等
   - 代码可读性提升

#### 🔧 技术实现
```python
# 工厂模式创建 LLM 服务
llm = create_llm_service(
    service="qwen",  # 或 deepseek, openai
    api_key="...",
    base_url="...",
    model="..."
)

# 统一上下文
context = create_llm_context(messages, tools=tools)
```

#### 📊 配置示例
```env
# 切换 LLM 服务只需修改一行
LLM_SERVICE=qwen     # 或 deepseek, openai

# 每个服务独立配置
QWEN_API_KEY=xxx
DEEPSEEK_API_KEY=xxx
OPENAI_API_KEY=xxx
```

#### 🎯 技术亮点
- ✅ 工厂模式 - 统一接口，灵活切换
- ✅ 配置分组 - 清晰易读
- ✅ 向后兼容 - 保留旧变量名映射
- ✅ 扩展友好 - 轻松添加新 LLM 服务
- ✅ 变量名语义化 - 不再硬编码特定厂商

#### 🐛 Bug 修复
- 修复 `QwenLLMService` 访问 `self.model` 报错
- 所有 LLM 服务添加 `_model_name` 属性
- 增强 `get_model_display_name()` fallback 逻辑

#### 📝 修改文件
- 新增：`src/voice_assistant/llm_services.py`
- 修改：`config.py`, `pipecat_main_v2.py`, `.env.example`

---

### v2.2.1 - 修复 TTS 无输出问题（2025-12-25）

#### 🐛 关键 Bug 修复
1. **修复 LLM 文本无法到达 TTS 处理器**
   - 问题：`assistant_aggregator` 消费了 `LLMTextFrame` 但不传递给下游
   - 原因：官方设计假设下游不需要 LLM 文本帧
   - 修复：调整 Pipeline 顺序，将 TTS 放在 `assistant_aggregator` **之前**
   - 结果：TTS 能正常接收并播放 LLM 响应 ✅

2. **修复 TTS 事件循环错误**
   - 问题：`There is no current event loop in thread 'asyncio_X'`
   - 原因：在线程池中执行 TTS 时尝试获取当前事件循环
   - 修复：在初始化时保存主事件循环引用，线程池中使用 `run_coroutine_threadsafe()`
   - 结果：TTS 正常在线程池中推送音频帧 ✅

#### ✨ 用户体验改进
- **实时 LLM 输出显示**：终端实时显示流式文本（`🤖 LLM: ...`）
- **简化 TTS 日志**：移除冗余的句子合成日志
- **Pipeline 架构优化**：新顺序 `LLM → TTS → assistant_aggregator`

#### 📊 Pipeline 结构调整
```
修复前（错误）：
llm → assistant_aggregator → tts  ❌ (TTS 收不到 LLMTextFrame)

修复后（正确）：
llm → tts → assistant_aggregator  ✅ (TTS 正常接收并播放)
```

#### 🎯 技术亮点
- ✅ 符合 Pipecat 帧传递机制（aggregator 消费但不传递文本帧）
- ✅ TTS 处理后传递帧给 aggregator（两者都正常工作）
- ✅ 降低 TTS 延迟（不需要等待 context 更新）
- ✅ 保持官方代码完整性（不修改 Pipecat 源码）

---

### v2.2.0 - Smart Turn v3 智能对话检测（2025-12-25）

#### ✨ 新增特性
1. **Smart Turn v3 集成** - 智能判断对话完成
   - ✅ 理解语言上下文（语法、语调、语速、语义）
   - ✅ 避免句子中间被打断
   - ✅ 支持 23 种语言
   - ✅ 本地 CPU 推理（<100ms 延迟）
   - ✅ 基于 Pipecat 官方 smart-turn 模型

2. **Silero VAD 完整集成** - 快速语音检测
   - ✅ 配置 `stop_secs=0.2`（快速检测停顿）
   - ✅ 配合 Turn Detection 智能判断
   - ✅ 正确启动 `_audio_task_handler`（修复 VAD 不工作问题）
   - ✅ 标准 `BaseInputTransport` 架构

3. **ASR 唤醒优化** - 零延迟响应
   - ✅ 唤醒后立即开始录音（不等 VAD 重新检测）
   - ✅ 只在唤醒后响应 VAD 事件（避免误触发）
   - ✅ 响应延迟从 2+ 秒 → <100ms

#### 🐛 Bug 修复
1. **修复 VAD 不工作问题**
   - 问题：`_audio_task_handler` 从未启动
   - 修复：在 `StartFrame` 处理时调用 `set_transport_ready()`
   - 结果：VAD 正常检测语音开始/停止

2. **修复唤醒后延迟问题**
   - 问题：用户说完唤醒词后需要等待 VAD 重新检测（2+ 秒延迟）
   - 修复：唤醒后立即开始录音（`recording = True`）
   - 结果：可以一口气说完"小智，打开百度"

3. **修复句子中间被打断问题**
   - 问题：只有 VAD（简单静音检测），用户停顿就触发识别
   - 修复：集成 Smart Turn v3（理解语言上下文）
   - 结果：可以正常说完整句话，中间停顿思考也不会被打断

#### 📝 新增依赖
```bash
pip install "pipecat-ai[local-smart-turn-v3]"
```
- transformers (11.4MB)
- tokenizers (2.6MB)
- huggingface-hub
- safetensors
- pyyaml

#### 🔧 技术改进
- ✅ 完全符合 Pipecat 官方 VAD + Turn Detection 架构
- ✅ 工作流程：VAD 快速检测（0.2s）→ Turn Detection 智能判断
- ✅ Turn Detection 分析最近 8 秒音频，判断完整/未完整
- ✅ 未完整时自动延长等待（最多 3 秒）
- ✅ 完整后立即识别，响应速度快

#### 📊 用户体验提升
- ✅ 唤醒后无需停顿，可以一口气说完指令
- ✅ 可以正常说完整句话，不会被中间打断
- ✅ 支持自然停顿思考（Smart Turn 智能判断）
- ✅ 响应速度提升 20 倍（2+ 秒 → <100ms）

---

### v2.1.1 - 架构优化与修复（2025-12-25）

#### 🐛 Bug 修复
1. **修复 Ctrl+C 挂起问题** - 程序优雅退出
   - ✅ 添加 `CancelFrame` 发送逻辑，正确停止 Pipeline
   - ✅ 音频输入循环响应 `CancelFrame` 退出
   - ✅ 所有清理操作添加超时保护（2-3 秒）
   - ✅ 退出时显示清理日志，提供清晰反馈

2. **符合 Pipecat v0.0.98 BaseTransport 标准**
   - ✅ 移除 `TransportParams` 传递（API 不支持）
   - ✅ 创建独立 `PyAudioTransport` 类
   - ✅ 实现标准 `input()` 和 `output()` 方法

#### ⚠️ 临时调整
1. **VAD 集成暂时禁用** - 等待 Pipecat API 兼容
   - SileroVADAnalyzer 缺少预期的公共 API（`start()` 方法）
   - ASR 暂时回退到简单 RMS VAD（音量检测）
   - 已完成 VAD 优化代码（待 Pipecat 更新后启用）

2. **ASR 暂用内置 VAD** - 临时方案
   - 使用 RMS 阈值检测（0.02）
   - 静音检测：20 帧（约 0.64 秒）
   - 超时保护：300 帧（约 10 秒）
   - TODO: 待 Pipecat VAD API 稳定后切换

#### 📝 新增文档
- `docs/vad-optimization-summary.md` - VAD 优化完整总结
- `pyaudio_transport.py` - 标准 Transport 实现
- `vad_processor.py` - Silero VAD Processor（待启用）

#### 🔧 技术改进
- ✅ Pipeline 清理机制更健壮
- ✅ 退出流程完全异步，无死锁风险
- ✅ 代码增加 11%（~2,833 → ~3,148 行，主要是新增 Transport 和 VAD）

---

### v2.2.1 - 环境变量配置 + LLM 修复（2025-12-25）

#### ✨ 新增特性
1. **环境变量配置管理** - 安全的 API 密钥管理
   - 使用 `.env` 文件管理所有 API 配置（DASHSCOPE_API_KEY、DASHSCOPE_API_URL、QWEN_MODEL、ALIYUN_APPKEY等）
   - `.env` 文件被 `.gitignore` 忽略，不会泄露密钥
   - 提供 `.env.example` 作为配置模板
   - 支持团队成员各自使用独立配置
   - 使用 `python-dotenv` 自动加载环境变量

2. **Qwen3 优化** - 提升响应速度和稳定性
   - 禁用 Qwen3 思考模式（`chat_template_kwargs: {enable_thinking: false}`）
   - 通过 `extra.body` 传递模型特殊参数
   - 保持与 OpenAI API 的完全兼容性

#### 🐛 Bug 修复
1. **LLM tool_calls 格式修复** - 修复 TTS 无输出问题
   - 修复 assistant message 缺少 `content` 字段导致 400 错误
   - 本地 Qwen 严格要求 tool_calls 时必须包含 `content`（即使为空）
   - 重写 `build_chat_completion_params()` 方法自动修正消息格式
   - 修复后 LLM 能正常输出文本回复，TTS 恢复工作
   - 错误示例：`{'role': 'assistant', 'tool_calls': [...]}` ❌
   - 正确格式：`{'role': 'assistant', 'content': '', 'tool_calls': [...]}` ✅

2. **TTS 音频输出修复** - 正确实现 Pipecat 音频输出接口
   - 实现 `write_audio_frame()` 方法（BaseOutputTransport 期望的接口）
   - 修复 OutputAudioRawFrame 的 destination 注册问题
   - 音频帧现在正确通过 MediaSender 流向输出设备
   - 移除手动 process_frame 处理，使用官方 MediaSender 机制

#### 🔧 技术改进
- ✅ 环境变量管理（python-dotenv）
- ✅ 敏感配置与代码分离
- ✅ LLM 消息格式自动修正
- ✅ Qwen3 特殊参数支持
- ✅ 符合 Pipecat BaseOutputTransport 标准接口
- ✅ 代码更安全、更规范、更易维护

#### 📝 修改文件
- `.env.example` - 配置模板（新增）
- `src/voice_assistant/config.py` - 使用环境变量
- `src/voice_assistant/qwen_llm_service.py` - LLM 格式修正 + Qwen3 参数
- `src/voice_assistant/pyaudio_transport.py` - 实现 write_audio_frame()
- `pyproject.toml` - 添加 python-dotenv 依赖

---

### v2.1.0 - Pipecat 官方 LLM Service 集成（2025-12）

#### ✨ 新增特性
1. **QwenLLMService 官方框架集成** - 享受 Pipecat 生态
   - 继承 `OpenAILLMService`，完全兼容 Qwen API
   - 自动管理对话历史（`LLMContextAggregator`）
   - 原生 Function Calling 支持（MCP 工具无缝集成）
   - 新增 `qwen_llm_service.py`（209 行）

2. **混合架构设计** - 自定义 + 官方最佳实践
   - **保留自定义**: KWS、ASR、Piper TTS、Qwen Vision（官方不支持）
   - **改用官方**: LLM Service、Context Aggregator、Function Calling
   - Pipeline 重构：`user_aggregator` → `llm` → `assistant_aggregator`
   - 代码增加 12%（~2480 → ~2833 行，主要是 LLM Service）

3. **MCP 工具转换器** - 自动集成到 LLM
   - `mcp_tools_to_function_schemas()` - 转换 MCP 工具为 FunctionSchema
   - `create_tools_schema_from_mcp()` - 创建 ToolsSchema
   - `register_mcp_functions()` - 统一注册 MCP 函数处理器
   - LLM 自动决定何时调用工具，无需手动 React 循环

4. **完整集成指南** - 开箱即用
   - 新增 `docs/qwen_llm_service_guide.md`（309 行）
   - 快速开始、MCP 集成、完整示例、API 参考
   - 架构对比、常见问题解答

#### 📊 性能提升
- 对话历史自动管理，无需手动维护
- Function Calling 原生支持，调用效率更高
- 代码简化，易于维护和扩展

#### 🔧 技术改进
- ✅ 符合 Pipecat 官方最佳实践（LLM Service + Context Aggregator）
- ✅ 完全兼容 OpenAI API 格式（Qwen DashScope）
- ✅ 保留所有自定义优势（本地 KWS/ASR/TTS）
- ✅ 生态兼容，可与其他官方 Processor 配合
- ✅ 混合架构，最佳平衡点

---

### v2.0.0 - Pipecat 单一模式迁移（2025-12）

#### ✨ 新增特性
1. **Pipecat 单一模式架构** - 统一为完全异步架构
   - 移除传统模式，统一到 Pipecat 框架
   - 完全异步 Pipeline 架构
   - KWS → ASR → Vision → Agent → TTS 流水线
   - 代码减少 28%（3457 → ~2480 行）

2. **Vision 异步集成** - 视觉理解完全异步化
   - 新增 `VisionProcessor`（智能路由）
   - 使用 httpx + aiofiles 异步 API 调用
   - 集成到 Pipecat Pipeline
   - 支持中断和流式处理

3. **Pipecat 官方中断机制** - 符合 Pipecat 最佳实践
   - 使用标准 `InterruptionFrame` 替代自定义帧
   - 配置 `allow_interruptions=True` 全局管理
   - 发出 `TTSStoppedFrame` 明确停止事件
   - 生态兼容：可与官方 TTS/LLM Processor 配合

4. **MCP 官方推荐模式重构**
   - 基于 MCP Python SDK v1.25.0 官方最佳实践
   - 移除所有线程和 `run_coroutine_threadsafe`
   - 纯异步调用 `await session.call_tool()`
   - 移除同步 MCP 包装器（MCPClientSync/MCPManagerSync）

5. **Piper TTS 集成** - 本地超低延迟语音合成
   - 延迟降低至 100-200ms
   - 完全离线运行
   - 中文音色自然

6. **Playwright MCP 集成** - 浏览器自动化控制
   - 网页导航、元素交互、截图
   - 支持 Chrome/Firefox/Safari
   - 跨平台支持

7. **React Agent 框架** - 智能推理决策
   - 多轮思考+行动循环
   - **完全异步执行**（移除同步模式）
   - 自动错误纠正
   - 长期记忆支持

#### 🐛 Bug 修复
- 修复 Pipecat 模式 Agent 阻塞 Pipeline 问题
- 修复 MCP 事件循环冲突导致超时
- 添加 TTS 中断机制
- 支持后台任务取消

#### 📊 性能提升
- TTS 延迟从 500ms → 100ms（Piper）
- MCP 调用完全异步，无线程开销
- Pipeline 流式处理，不阻塞
- 代码减少 28%（更易维护）

#### 🔧 技术改进
- ✅ 符合 MCP Python SDK 官方推荐模式
- ✅ 符合 Pipecat 官方中断机制（InterruptionFrame + TTSStoppedFrame）
- ✅ Vision 完全异步化（httpx + aiofiles）
- ✅ 移除所有同步代码和线程
- ✅ 完全异步架构，无自定义帧类型
- ✅ 生态兼容，可与官方 Processor 配合
- ✅ 代码简化，易于维护和扩展

---

## 🙏 致谢

- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) - 高性能语音识别框架
- [Playwright](https://playwright.dev/) - 强大的浏览器自动化工具
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP 官方 Python SDK
- [Pipecat AI](https://github.com/pipecat-ai/pipecat) - 实时音频处理框架
- [Piper TTS](https://github.com/rhasspy/piper) - 快速本地文本转语音引擎
- [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) - 流式语音合成库
- [阿里云 DashScope](https://dashscope.aliyun.com/) - 多模态 API 和 LLM 服务
- [Qwen](https://github.com/QwenLM/Qwen) - 强大的大语言模型和视觉模型

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

Made with ❤️

</div>
