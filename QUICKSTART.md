# 快速开始指南

## 🚀 5 分钟快速启动

### 1️⃣ 安装依赖

```bash
# 克隆项目（如果还没有）
git clone <repository-url>
cd chinese-voice-assistant

# 安装依赖
uv sync
```

### 2️⃣ 配置 API Key

```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env 文件，至少配置一个 LLM API Key
# 使用 Qwen（推荐，免费额度）：
# QWEN_API_KEY=your-qwen-api-key-here
```

**获取 API Key**：
- **Qwen**: https://dashscope.console.aliyun.com/ （推荐，有免费额度）
- **DeepSeek**: https://platform.deepseek.com/ （便宜）
- **OpenAI**: https://platform.openai.com/

### 3️⃣ 启动程序

```bash
# 方式 1：使用主程序（推荐）
uv run python main.py

# 方式 2：测试音频设备
uv run python list_audio_devices.py

# 方式 3：测试 TTS
uv run python test_realtime_tts.py
```

### 4️⃣ 开始对话

说出唤醒词：
- "小智"
- "你好助手"
- "智能助手"

然后说出你的指令！

---

## 📋 推荐配置方案

### 方案 1：开发环境（免费）

```bash
# .env 配置
LLM_SERVICE=qwen
QWEN_API_KEY=your-qwen-api-key-here
TTS_SERVICE=piper
VISION_SERVICE=moondream
```

**特点**：
- ✅ 完全免费（Qwen 有免费额度）
- ✅ 延迟最低（Piper 本地 TTS <100ms）
- ✅ 离线运行（Piper TTS + Moondream Vision）
- ⚠️ 音质一般（Piper TTS）

### 方案 2：生产环境（推荐）⭐

```bash
# .env 配置
LLM_SERVICE=qwen
QWEN_API_KEY=your-qwen-api-key-here
TTS_SERVICE=dashscope_realtime
DASHSCOPE_REALTIME_MODEL=qwen3-tts-flash-realtime
DASHSCOPE_REALTIME_VOICE=Cherry
DASHSCOPE_REALTIME_MODE=server_commit
VISION_SERVICE=qwen-vl-plus
```

**特点**：
- ✅ 低延迟（WebSocket ~100-200ms）
- ✅ 高音质（Qwen TTS）
- ✅ 稳定可靠
- 💰 费用低（按量计费）

### 方案 3：最低延迟

```bash
# .env 配置
LLM_SERVICE=qwen
QWEN_API_KEY=your-qwen-api-key-here
TTS_SERVICE=dashscope_realtime
DASHSCOPE_REALTIME_MODE=commit
VISION_SERVICE=qwen-vl-plus
```

**特点**：
- ⚡ 最低延迟（~100ms）
- ⚠️ 需要手动控制合成时机
- 💰 费用低

---

## 🔧 常见问题

### Q1: 没有声音？

检查音频设备：
```bash
uv run python list_audio_devices.py
```

然后在 `.env` 中配置：
```bash
AUDIO_INPUT_DEVICE_INDEX=1
AUDIO_OUTPUT_DEVICE_INDEX=5
```

**注意**：避免选择"立体声混音"设备！

### Q2: 唤醒词检测不到？

1. 确保麦克风正常工作
2. 靠近麦克风说话（30cm 内）
3. 清晰地说出唤醒词："小智"

### Q3: 如何切换 LLM 服务？

修改 `.env` 配置：

```bash
# 切换到 DeepSeek
LLM_SERVICE=deepseek
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# 切换到 OpenAI
LLM_SERVICE=openai
OPENAI_API_KEY=your-openai-api-key-here
```

### Q4: 如何切换 TTS 引擎？

修改 `.env` 配置：

```bash
# 切换到 WebSocket Realtime（推荐）
TTS_SERVICE=dashscope_realtime

# 切换到 HTTP 流式
TTS_SERVICE=dashscope

# 切换到本地 Piper
TTS_SERVICE=piper
```

### Q5: WebSocket 连接失败？

1. 检查 API Key 是否正确
2. 检查网络是否可以访问 DashScope
3. 检查防火墙是否允许 WebSocket 连接

临时解决方案：切换回 HTTP 流式
```bash
TTS_SERVICE=dashscope
```

---

## 📚 下一步

- 📖 [完整配置文档](.env.example)
- 📖 [WebSocket TTS 详细指南](docs/QWEN_TTS_REALTIME.md)
- 📖 [从 HTTP 迁移到 WebSocket](docs/TTS_MIGRATION_GUIDE.md)
- 💻 [示例代码](Example/)

---

## ⚡ 性能优化建议

### 降低延迟

1. **使用 WebSocket Realtime TTS**
   ```bash
   TTS_SERVICE=dashscope_realtime
   ```

2. **使用 Piper 本地 TTS**
   ```bash
   TTS_SERVICE=piper
   ```

3. **选择更快的 LLM**
   ```bash
   QWEN_MODEL=qwen-turbo
   ```

### 提升音质

1. **使用 WebSocket Realtime TTS**
   ```bash
   TTS_SERVICE=dashscope_realtime
   DASHSCOPE_REALTIME_VOICE=Cherry
   ```

2. **使用 Azure TTS（付费）**
   ```bash
   TTS_SERVICE=azure
   AZURE_TTS_VOICE=zh-CN-XiaoxiaoNeural
   ```

### 降低成本

1. **使用 Piper 本地 TTS**（免费）
2. **使用 Qwen Turbo**（更便宜）
3. **使用 DeepSeek**（性价比高）

---

## 🎯 核心功能

### ✅ 已实现

- 🎤 **语音唤醒**：KWS + ASR 双阶段检测
- 🧠 **多 LLM 支持**：Qwen/DeepSeek/OpenAI/Anthropic
- 🔊 **多 TTS 引擎**：Piper/DashScope Realtime/Edge/Azure
- 👁️ **视觉理解**：Moondream/Qwen-VL
- 🔧 **MCP 工具**：Playwright 浏览器控制
- 🛠️ **Agent Skills**：LLM 自主判断技能使用
- 🔇 **音频降噪**：RNNoise + soxr

### 📊 架构特点

- 基于 Pipecat 官方框架
- 完全异步架构（asyncio）
- 模块化设计，易于扩展
- 配置驱动，.env 一键切换

---

## 💡 使用技巧

### 1. 技能使用

LLM 会自动判断使用技能，无需关键词：

```
你："查一下北京天气"
LLM：自动调用 weather 技能

你："打开百度"
LLM：自动调用 browser 技能
```

### 2. 浏览器控制

```
你："打开百度搜索"
你："点击搜索框"
你："输入 Claude Code"
```

### 3. 视觉理解

```
你："查看当前屏幕"
LLM：截图并分析内容
```

---

## 🆘 需要帮助？

- 📖 [完整文档](README.md)
- 🐛 [报告问题](https://github.com/yourusername/chinese-voice-assistant/issues)
- 💬 [讨论区](https://github.com/yourusername/chinese-voice-assistant/discussions)

---

**祝使用愉快！🎉**
