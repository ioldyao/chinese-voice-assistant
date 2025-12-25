# 智能语音助手

<div align="center">

**双阶段语音识别 + Pipecat 官方 LLM + Playwright 浏览器控制 + 实时音频处理**

基于 Sherpa-ONNX + Qwen LLM Service + Playwright MCP + Piper + Pipecat 的中文语音助手

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.2.0-green.svg)](https://github.com/yourusername/chinese-voice-assistant)

</div>

## ✨ 特性

### 🚀 核心功能
- **🎤 语音唤醒**:
  - **阶段1 - KWS**: 轻量级关键词检测（3.3MB），持续监听，CPU占用低
  - **阶段2 - ASR**: 唤醒后启动完整语音识别（120MB），准确率高
  - 支持自定义唤醒词（默认：小智、你好助手、智能助手）

- **🧠 Qwen LLM Service**: 基于 Pipecat 官方框架
  - **完全异步执行**（继承 OpenAILLMService）
  - **自动管理对话历史**（LLMContextAggregator）
  - **Function Calling**（MCP 工具无缝集成）
  - 基于 MCP Python SDK 官方推荐模式

- **🎭 Playwright MCP**: 浏览器自动化操作
  - 网页导航、元素交互、截图、PDF生成等
  - 支持 Chrome/Firefox/Safari 浏览器控制
  - **完全异步**的工具调用（符合 MCP 官方最佳实践）

- **🔊 语音合成**: 多引擎支持
  - **Piper TTS** - 本地超低延迟（推荐）⚡
  - **RealtimeTTS** - 流式实时播放 🎵
  - **MeloTTS** - 中英文混合支持 🌐

- **👁️ 视觉理解**: Qwen-VL-Max 多模态理解
  - 屏幕内容分析
  - 支持窗口/全屏截图
  - **完全异步化**（httpx + aiofiles）

### 🎨 技术亮点（混合架构）
- ⚡ **官方 LLM Service** - QwenLLMService（继承 OpenAILLMService）✨
- 🎯 **自动对话管理** - LLMContextAggregator（官方框架）✨
- 🔄 **Function Calling** - MCP 工具无缝集成（官方机制）✨
- 🚀 **保留自定义优势** - KWS + ASR + Piper TTS（本地、免费）
- 🛡️ **完全异步架构** - 纯异步，无线程开销
- 👁️ **Vision 异步集成** - 视觉理解完全异步化
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
- ✅ Qwen Vision（保持现有 API）

**改用官方（享受生态）**：
- ✨ LLM Service（QwenLLMService）
- ✨ Context Aggregator（自动管理历史）
- ✨ Function Calling（MCP 工具集成）

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

### API Keys
在 `src/voice_assistant/config.py` 中配置 API Key：

```python
# 方式1: 直接修改配置文件
DASHSCOPE_API_KEY = "your-api-key-here"
ALIYUN_APPKEY = "your-app-key-here"

# 方式2: 使用环境变量（推荐）
export DASHSCOPE_API_KEY="your-api-key-here"
export ALIYUN_APPKEY="your-app-key-here"
```

获取 API Key：
- [阿里云 DashScope](https://dashscope.console.aliyun.com/)

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
│   ├── config.py             # 配置管理 (40行)
│   ├── wake_word.py          # 模型加载器 (95行)
│   ├── react_agent.py        # React 智能代理 (603行)
│   │                         # - 完全异步执行（Pipecat 模式）
│   ├── mcp_client.py         # MCP 客户端 (378行)
│   │                         # - MCPManager (异步，多Server管理)
│   ├── qwen_llm_service.py   # Qwen LLM Service (209行)
│   │                         # - QwenLLMService（官方框架）
│   │                         # - MCP Tools 转换器
│   │                         # - Function Calling 注册
│   ├── pipecat_main_v2.py    # Pipecat 主程序 v2 (365行)
│   │                         # - 符合官方架构（BaseTransport + CancelFrame）
│   │                         # - 修复 Ctrl+C 挂起问题
│   ├── pyaudio_transport.py  # PyAudio Transport (201行)
│   │                         # - 标准 BaseTransport 实现
│   ├── vad_processor.py      # VAD Processor (63行)
│   │                         # - Silero VAD 集成（开发中）
│   ├── pipecat_adapters.py   # Pipecat Processors (620行)
│   │                         # - SherpaKWSProcessor (KWS)
│   │                         # - SherpaASRProcessor (ASR + 临时 RMS VAD)
│   │                         # - VisionProcessor (Vision)
│   │                         # - PiperTTSProcessor (TTS)
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
├── main.py                   # 主程序入口 (26行)
├── pyproject.toml            # 项目配置 (v2.1.1)
└── README.md                 # 项目文档
```

### 代码统计
| 模块 | 代码行数 | 主要功能 |
|-----|---------|---------|
| `pipecat_adapters.py` | 620 | Pipecat Processors（KWS/ASR+VAD/Vision/TTS） |
| `react_agent.py` | 603 | React 推理框架（完全异步） |
| `mcp_client.py` | 378 | MCP 客户端（异步多 Server） |
| `tts.py` | 372 | TTS 引擎管理（Piper/RealtimeTTS） |
| `pipecat_main_v2.py` | 365 | Pipecat Pipeline v2（修复挂起） |
| `qwen_llm_service.py` | 209 | Qwen LLM Service（官方框架集成） |
| `pyaudio_transport.py` | 201 | 标准 PyAudio Transport |
| `vision.py` | 136 | Qwen-VL-Max 视觉理解（异步） |
| `wake_word.py` | 95 | 模型加载器（KWS + ASR） |
| `vad_processor.py` | 63 | Silero VAD Processor（开发中） |
| `config.py` | 40 | 全局配置 |
| `__init__.py` | 40 | 模块导出 |
| `main.py` | 26 | Pipecat 单一入口 |
| **总计** | **~3,148** | **v2.2.0 完整实现** |

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

#### **Pipecat v2.2 架构**
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
  OpenAIAssistantContextAggregator (保存助手响应 - 官方 ✨)
    ↓
  PiperTTSProcessor (语音合成 - 自定义)
    ↓
  PyAudioTransport.output() (音频输出 - 标准 BaseTransport)
```

**关键特性**：
- ✅ Silero VAD（快速检测语音段，stop_secs=0.2）
- ✅ Smart Turn v3（智能判断对话完成，理解语言上下文）
- ✅ 避免句子中间被打断
- ✅ 支持 23 种语言

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
