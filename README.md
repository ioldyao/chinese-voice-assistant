# Chinese Voice Assistant

<div align="center">

**Chinese Voice Assistant v2.7.0 - Pipecat Official Architecture + Agent Skills + WebSocket Realtime TTS**

Dual-stage Speech Recognition + Multi-LLM Support + MCP Tool Integration + Agent Skills + Smart Audio Processing

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.7.0-green.svg)](https://github.com/yourusername/chinese-voice-assistant)

</div>

## Quick Start

### 1. Install Dependencies

```bash
# Clone the project
git clone <repository-url>
cd chinese-voice-assistant

# Install dependencies
uv sync
```

### 2. Configure API Key

```bash
# Copy config file
cp .env.example .env

# Edit .env, configure at least one LLM API Key
```

**Recommended: Qwen** (free tier available):
```bash
LLM_SERVICE=qwen
QWEN_API_KEY=your-qwen-api-key-here
```

[Get API Key →](https://dashscope.console.aliyun.com/)

### 3. Start the Program

```bash
uv run python main.py
```

### 4. Start Talking

Say a wake word:
- "Xiaozhi" (小智)
- "Ni hao zhushou" (你好助手)
- "Zhineng zhushou" (智能助手)

---

## Documentation

| Document | Description |
|------|------|
| **[Chinese Documentation](README_zh.md)** | 中文文档 |
| **[Quick Start Guide](QUICKSTART.md)** | 5-minute quick start |
| **[Config Comparison](docs/CONFIG_COMPARISON.md)** | Choose the right config |
| **[Full Config Reference](.env.example)** | All config parameters |
| **[WebSocket TTS Guide](docs/QWEN_TTS_REALTIME.md)** | Realtime TTS details |
| **[TTS Migration Guide](docs/TTS_MIGRATION_GUIDE.md)** | Switch from HTTP to WebSocket |

---

## Features

### Core Functionality
- **Voice Wake-up**:
  - **Stage 1 - KWS**: Lightweight keyword detection (3.3MB), continuous listening, low CPU usage
  - **Stage 2 - ASR**: Full speech recognition after wake-up (120MB), high accuracy
  - Custom wake words supported (default: 小智, 你好助手, 智能助手)

- **Multi-LLM Support**: Factory pattern, flexible switching
  - **Qwen** - Alibaba Cloud DashScope (Chinese optimized, Function Calling)
  - **DeepSeek** - DeepSeek API (strong reasoning, low cost)
  - **OpenAI** - Official API (GPT-4o, o1, etc.)
  - **Anthropic Claude** - Claude Sonnet 4.5/Opus 4.6 (supports Thinking mode)
  - Based on Pipecat official framework (inherits OpenAILLMService)
  - **Fully async execution** with automatic conversation history management
  - **Unified interface**, switch models via `.env`

- **Playwright MCP**: Browser automation
  - Web navigation, element interaction, screenshots, PDF generation
  - Chrome/Firefox/Safari browser control
  - **Fully async** tool calls (MCP official best practice)

- **Agent Skills**: Claude Code-style skill system
  - LLM autonomously decides which skill to use (no keyword matching)
  - Skill descriptions injected into system prompt
  - Unified `skill_execute` function interface
  - Custom skill extensions supported
  - Zero stopwords, zero keyword matching, zero interference

- **Text-to-Speech**: Multi-engine support, low-latency streaming
  - **Piper TTS** - Local ultra-low latency (<100ms, recommended for dev)
  - **DashScope Realtime** - WebSocket realtime synthesis (~100-200ms, recommended for prod)
  - **DashScope HTTP** - HTTP streaming (~300ms, stable)
  - **Edge TTS** - Microsoft free API

- **Vision Understanding**: Multi-model support (configurable)
  - **Moondream (Local)** - Fully offline, privacy protection
    - Hardware acceleration (CUDA/MPS/CPU)
    - Auto image optimization
    - Chinese/English smart prompts
  - **Qwen-VL-Plus (API)** - High accuracy
  - **Qwen-VL-Max (API)** - Highest accuracy
  - Switch models via `.env`

- **Audio Noise Reduction**: RNNoise + soxr high-quality audio processing
  - **RNNoise** - Deep learning noise reduction (<5ms latency)
  - **soxr** - High-quality audio resampling
  - Selectable modes: rnnoise / noise_gate / pass_through
  - Only affects ASR, preserves original audio quality

- **Audio Device Configuration**: Interactive device selection
  - Auto-detect audio devices on startup
  - Support specifying input/output devices
  - Avoid stereo mix issues
  - Persistent device index saving

### Technical Highlights
- **Multi-LLM Factory** - Qwen/DeepSeek/OpenAI/Anthropic flexible switching
- **Official LLM Service** - Inherits OpenAILLMService (official framework)
- **Auto Conversation Management** - LLMContextAggregator (official framework)
- **Function Calling** - MCP tool seamless integration
- **Agent Skills System** - Claude Code design, LLM autonomous judgment
- **Smart Turn v3** - Intelligent conversation completion detection (23 languages)
- **RNNoise Denoising** - Deep learning audio processing (<5ms latency)
- **Audio Device Config** - Interactive device selection

---

## Installation

### Requirements
- Python 3.12+
- Windows 10/11
- Microphone device
- Node.js 18+ (for Playwright MCP)

### Install Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Download Models
```bash
# Download KWS + ASR + VAD models
python scripts/download_models.py

# Download Piper TTS Chinese model (recommended)
python download_piper_model.py
```

Model files (~250MB total):
- **KWS Model** (3.3MB) - Zipformer WenetSpeech (wake word detection)
- **ASR Model** (120MB) - Paraformer Chinese (speech recognition)
- **Piper TTS Model** (~50MB) - Chinese speech synthesis (local, ultra-low latency)
- **VAD Model** (1MB) - Silero VAD (silence detection)

---

## Configuration

### API Keys (using .env file)

1. Copy example config:
```bash
cp .env.example .env
```

2. Edit `.env` file:
```bash
# ==================== LLM Config ====================
# Select LLM service: qwen | deepseek | openai | anthropic
LLM_SERVICE=qwen

# Qwen (Alibaba DashScope) config
QWEN_API_KEY=your-qwen-api-key-here
QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
# Options: qwen-plus, qwen-max, qwen-turbo

# DeepSeek config
# DEEPSEEK_API_KEY=your-deepseek-api-key-here
# DEEPSEEK_MODEL=deepseek-chat

# OpenAI config
# OPENAI_API_KEY=your-openai-api-key-here
# OPENAI_MODEL=gpt-4o

# Anthropic Claude config
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
# ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
# ANTHROPIC_ENABLE_THINKING=false
# ANTHROPIC_THINKING_EFFORT=medium

# ==================== Vision Config ====================
VISION_SERVICE=moondream
```

### Multi-LLM Switching Examples

#### Switch to DeepSeek
```bash
LLM_SERVICE=deepseek
DEEPSEEK_API_KEY=sk-xxxxx
DEEPSEEK_MODEL=deepseek-chat
```

#### Switch to OpenAI
```bash
LLM_SERVICE=openai
OPENAI_API_KEY=sk-xxxxx
OPENAI_MODEL=gpt-4o
```

#### Switch to Anthropic Claude
```bash
LLM_SERVICE=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxx
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
```

### MCP Server Configuration

Edit `config/mcp_servers.json` to enable/disable servers:

```json
{
  "servers": [
    {
      "name": "playwright",
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "timeout": 120,
      "enabled": true
    }
  ]
}
```

---

## Usage

### Start Assistant
```bash
# Using uv (recommended)
uv run python main.py

# Or directly
python main.py
```

### Interaction Flow
1. **Wake up**: Say a wake word (e.g., "小智")
2. **Command**: Speak your command after the prompt
3. **Execute**: System automatically understands and executes

### Supported Commands

#### Browser Navigation
```
"Open Bilibili"
"Visit Baidu"
"Browser back"
"Refresh page"
```

#### Web Interaction
```
"Click search box"
"Click login button"
"Enter test text in input box"
```

#### Screenshots
```
"Take screenshot of current page"
"Save page as PDF"
```

#### Vision Understanding
```
"Look at what's displayed in the browser window"
"Analyze current screen content"
```

---

## Project Structure

```
chinese-voice-assistant/
├── src/voice_assistant/      # Core source code
│   ├── __init__.py           # Module exports (lazy import)
│   ├── config.py             # Configuration management
│   ├── wake_word.py          # Wake word system
│   ├── mcp_client.py         # MCP client (async, multi-server)
│   ├── llm_services.py       # LLM service factory
│   ├── qwen_llm_service.py   # MCP tool converter and function registration
│   ├── pipecat_main_v2.py    # Pipecat main program v3.1
│   ├── pyaudio_transport.py  # PyAudio Transport
│   ├── audio_processors.py   # Audio processors (RNNoise + soxr)
│   ├── audio_device_setup.py # Audio device configuration
│   ├── pipecat_adapters.py   # Pipecat Processors
│   ├── vision_services.py    # Vision service factory
│   ├── tts.py                # TTS engine management
│   ├── tts_realtime.py       # WebSocket realtime TTS
│   ├── tts_realtime_adapter.py # Realtime TTS adapter
│   ├── vision.py             # Vision understanding
│   └── skills/               # Agent Skills system
│       ├── base_skill.py     # Skill base class
│       ├── skill_loader.py   # Skill loader
│       ├── skill_manager.py  # Skill manager
│       └── skill_executor.py # Skill executor
│
├── skills/                   # Skill definitions
│   ├── browser/              # Browser skill
│   ├── calendar/             # Calendar skill
│   └── weather/              # Weather skill
│
├── config/                   # Configuration files
│   ├── keywords.txt          # Wake word config
│   └── mcp_servers.json      # MCP Server config
│
├── models/                   # Model files (download required)
│   ├── piper/                # Piper TTS models
│   ├── sherpa-onnx-kws-*/    # KWS model (3.3MB)
│   └── sherpa-onnx-paraformer-zh/ # ASR model (120MB)
│
├── main.py                   # Main entry point
├── pyproject.toml            # Project config (v2.7.0)
└── README.md                 # This file
```

### Code Statistics
| Module | Lines | Main Function |
|-----|---------|---------|
| `pipecat_main_v2.py` | 670 | Pipecat main program (Pipeline + LLM factory) |
| `pipecat_adapters.py` | ~600 | Pipecat Processors (KWS/ASR/Vision/TTS) |
| `llm_services.py` | ~400 | LLM service factory (Qwen/DeepSeek/OpenAI/Anthropic) |
| `qwen_llm_service.py` | ~500 | MCP tool conversion + function registration |
| `mcp_client.py` | ~400 | MCP client (async multi-server) |
| `tts.py` | ~350 | TTS engine management |
| `tts_realtime.py` | ~300 | WebSocket realtime TTS |
| `vision_services.py` | ~250 | Vision service factory |
| `pyaudio_transport.py` | ~330 | PyAudio Transport (VAD + Turn Detection) |
| `audio_processors.py` | ~300 | Audio processors (RNNoise + soxr) |
| `skills/` | ~500 | Agent Skills system |
| Others | ~800 | Config, adapters, utilities |
| **Total** | **~5,800** | **v2.7.0 complete implementation** |

---

## Tech Stack

| Component | Technology | Description |
|------|------|------|
| **Speech Recognition** | | |
| Wake Word Detection | Sherpa-ONNX (Zipformer) | 3.3MB lightweight KWS |
| Speech Recognition | Sherpa-ONNX (Paraformer) | 120MB Chinese ASR |
| Silence Detection | Silero VAD | 1MB voice activity detection |
| Conversation Detection | Smart Turn v3 | Intelligent turn completion (23 languages) |
| **Speech Synthesis** | | |
| Local TTS | Piper TTS | Ultra-low latency (recommended) |
| Cloud TTS | DashScope WebSocket | Realtime streaming |
| Free TTS | Edge TTS | Microsoft free API |
| **AI Decision** | | |
| LLM Framework | Pipecat LLM Service | Official framework + auto history |
| LLM Models | Qwen/DeepSeek/OpenAI/Claude | Multi-model, factory pattern |
| Function Calling | MCP Protocol | Seamless tool integration |
| Agent Skills | Claude Code design | LLM autonomous judgment |
| **Browser Control** | | |
| MCP Framework | Model Context Protocol | Official Python SDK |
| Browser Automation | Playwright MCP | Cross-browser support |
| **Audio Processing** | | |
| Realtime Framework | Pipecat AI | Frame/Pipeline/Processor |
| Audio I/O | PyAudio | Recording and playback |
| Audio Denoising | RNNoise + soxr | Deep learning + high-quality resampling |
| **Vision** | | |
| Multimodal Models | Qwen-VL / Moondream | Screen content analysis |
| Screenshot | PIL ImageGrab | Screen capture |
| **Other** | | |
| Python | 3.12+ | Required |
| Node.js | 18+ | Required for Playwright MCP |

---

## FAQ

### Q: Wake word not detected?
A:
- Check if microphone is working
- Verify wake word is in config file (`config/keywords.txt`)
- Try speaking louder and closer to the microphone
- Check if KWS model is downloaded (3.3MB)

### Q: MCP tool call failed?
A:
- Ensure Node.js is installed (version 18+)
- Check npx command: `npx --version`
- Test Playwright MCP manually: `npx @playwright/mcp@latest`
- Check console error messages

---

## Notes

1. **API Costs**: Using Alibaba Cloud API (LLM, Vision) incurs costs
   - Recommended: Piper TTS (free, local)
   - Playwright operations are local, no API cost

2. **Privacy & Security**:
   - Don't commit API Keys to public repos
   - Use environment variables for sensitive info
   - Local models (Piper, Sherpa-ONNX) have no privacy risks

3. **System Compatibility**:
   - Playwright supports cross-platform
   - Pipecat mode tested on Windows

4. **Network Requirements**:
   - **No network needed**: KWS, ASR, Piper TTS (fully offline)
   - **Network required**: LLM decisions, Vision understanding
   - **First time needed**: Playwright MCP installation

---

## Acknowledgments

- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) - High-performance speech recognition framework
- [Playwright](https://playwright.dev/) - Powerful browser automation tool
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP Official Python SDK
- [Pipecat AI](https://github.com/pipecat-ai/pipecat) - Realtime audio processing framework
- [Piper TTS](https://github.com/rhasspy/piper) - Fast local text-to-speech engine
- [Alibaba DashScope](https://dashscope.aliyun.com/) - Multimodal API and LLM services
- [Qwen](https://github.com/QwenLM/Qwen) - Powerful LLM and vision models

---

<div align="center">

**If this project helps you, please give a ⭐ Star!**

Made with ❤️

</div>
