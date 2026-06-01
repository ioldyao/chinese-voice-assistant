# Qwen-TTS-Realtime WebSocket 使用指南

## 概述

Qwen-TTS-Realtime 是基于 WebSocket 的实时语音合成服务，相比 HTTP 流式有更低延迟（~100-200ms）。

## 核心优势

| 特性 | HTTP 流式 | WebSocket Realtime |
|------|-----------|-------------------|
| **首包延迟** | ~300ms | ~100-200ms ⚡ |
| **连接方式** | 短连接 | 持久连接 🔌 |
| **交互模式** | 单次请求 | 事件驱动 🎯 |
| **适用场景** | 通用场景 | 实时对话 💬 |
| **复杂度** | 简单 | 中等 |

## 快速开始

### 1. 安装依赖

```bash
# 确保 dashscope 版本 >= 1.25.11
uv add dashscope --upgrade
```

### 2. 配置环境变量

创建或编辑 `.env` 文件：

```bash
# TTS 服务选择
TTS_SERVICE=dashscope_realtime

# DashScope Realtime 配置
DASHSCOPE_API_KEY=your-api-key-here
DASHSCOPE_REALTIME_MODEL=qwen3-tts-flash-realtime
DASHSCOPE_REALTIME_VOICE=Cherry
DASHSCOPE_REALTIME_MODE=server_commit
```

### 3. 运行程序

```bash
# 使用主程序
uv run python main.py

# 或运行测试脚本
uv run python test_realtime_tts.py
```

## 配置说明

### 模型选择

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| **qwen3-tts-flash-realtime** | 快速、高质量、推荐 | 实时对话（首选） |
| **qwen3-tts-instruct-flash-realtime** | 支持指令控制 | 需要精细控制 |
| **qwen-tts-realtime** | 经典模型 | 兼容性需求 |

### 音色选择

#### 系统音色（免费）
- **Cherry**: 女声，温柔（推荐）
- **Ethan**: 男声，稳重
- **Sunny**: 女声，活泼
- **Dylan**: 男声，年轻

#### 专属音色（付费）
- 声音复刻（Voice Cloning）
- 声音设计（Voice Design）

### 交互模式

#### server_commit（推荐）
```python
# 服务端自动判断合成时机
# 简单易用，适合大多数场景
mode="server_commit"

# 使用方式
await tts.speak("你好")
await tts.speak("世界")
# 系统自动合并和分段
```

#### commit（最低延迟）
```python
# 客户端手动控制合成时机
# 延迟最低，但需要代码支持
mode="commit"

# 使用方式
await tts.append_text("你好")
await tts.append_text("世界")
await tts.commit()  # 手动触发合成
```

## 高级功能

### 指令控制（仅 qwen3-tts-instruct-flash-realtime）

```python
config = RealtimeTTSConfig(
    model="qwen3-tts-instruct-flash-realtime",
    instructions="语速较快，带有明显的上扬语调，适合介绍时尚产品。",
    optimize_instructions=True  # 自动优化指令
)
```

### 音频参数控制

```python
config = RealtimeTTSConfig(
    speech_rate=1.2,   # 语速 [0.5, 2.0]
    volume=70,         # 音量 [0, 100]
    pitch_rate=1.1,    # 语调 [0.5, 2.0]
    response_format="mp3",  # 音频格式
    sample_rate=24000      # 采样率
)
```

## 代码示例

### 基础使用

```python
from voice_assistant.tts_realtime import create_realtime_tts

# 创建 TTS 实例
tts = create_realtime_tts(
    model="qwen3-tts-flash-realtime",
    voice="Cherry",
    mode="server_commit"
)

# 初始化
await tts.initialize()

# 合成并播放
await tts.speak("你好，世界")

# 关闭连接
await tts.close()
```

### Pipecat 集成

```python
from voice_assistant.tts_realtime_adapter import create_realtime_tts_processor

# 创建 TTS Processor
tts_proc = create_realtime_tts_processor(
    model="qwen3-tts-flash-realtime",
    voice="Cherry"
)

# 添加到 Pipeline
pipeline = Pipeline([
    transport.input(),
    kws_proc,
    asr_proc,
    user_aggregator,
    llm,
    tts_proc,  # WebSocket TTS
    assistant_aggregator,
    transport.output()
])
```

## 性能优化

### 1. 首包延迟优化

首包延迟是指从发送文本到收到第一个音频数据的时间。

**优化建议：**
- 使用 WebSocket（~100-200ms）而非 HTTP（~300ms）
- 使用 `server_commit` 模式（自动优化）
- 减少文本长度（短文本响应更快）

### 2. 连接复用

```python
# ✅ 推荐：复用连接
tts = create_realtime_tts()
await tts.initialize()

await tts.speak("第一句话")
await tts.speak("第二句话")
await tts.speak("第三句话")

await tts.close()

# ❌ 不推荐：每次创建新连接
for text in texts:
    tts = create_realtime_tts()
    await tts.initialize()
    await tts.speak(text)
    await tts.close()
```

### 3. 流式播放

```python
# ✅ 推荐：边生成边播放
await tts.speak("这是一段很长的文本...")
# 自动流式播放，延迟低

# ❌ 不推荐：等待完整音频
audio_data = await tts.synthesize("文本...")
play(audio_data)  # 需要等待完整生成
```

## 常见问题

### Q1: 如何切换回 HTTP 流式？

```bash
# .env 文件
TTS_SERVICE=dashscope  # 改回 dashscope
```

### Q2: WebSocket 连接失败怎么办？

检查：
1. API Key 是否正确
2. 网络是否可以访问 `wss://dashscope.aliyuncs.com`
3. 防火墙是否允许 WebSocket 连接

### Q3: 如何测试首包延迟？

运行测试脚本，查看控制台输出：
```
✓ 首包延迟: 150ms
```

### Q4: commit 模式如何使用？

```python
# 创建 commit 模式实例
tts = create_realtime_tts(mode="commit")

# 添加文本
await tts.append_text("你好")
await tts.append_text("世界")

# 手动触发合成
await tts.commit()
```

### Q5: 支持哪些音频格式？

- **pcm**（默认，推荐）
- **wav**
- **mp3**
- **opus**

## 技术对比

### HTTP 流式 vs WebSocket Realtime

```python
# HTTP 流式（当前实现）
response = dashscope.MultiModalConversation.call(
    model="qwen3-tts-flash",
    stream=True
)
for chunk in response:
    audio = base64.b64decode(chunk.output.audio.data)
    play(audio)

# WebSocket Realtime（新实现）
tts = QwenRealtimeTTS(config)
await tts.initialize()
await tts.speak("文本")
# 自动建立持久连接，低延迟流式播放
```

## 相关资源

- [官方文档](https://help.aliyun.com/zh/model-studio/qwen-tts-realtime)
- [GitHub 示例](https://github.com/aliyun/alibabacloud-bailian-speech-demo)
- [API 参考](https://help.aliyun.com/zh/model-studio/qwen-tts-realtime)
- [配置示例](.env.realtime.example)

## 更新日志

### v2.9.0 (2025-04-01)
- ✅ 新增 Qwen-TTS-Realtime WebSocket 支持
- ✅ 首包延迟优化至 ~100-200ms
- ✅ 支持 server_commit 和 commit 两种模式
- ✅ 支持指令控制（语速、语调等）
- ✅ 完整的 Pipecat Pipeline 集成

## 总结

**推荐配置：**

- **开发环境**：Piper TTS（本地，<100ms）
- **实时对话**：DashScope Realtime（WebSocket，~100-200ms）⭐
- **生产环境**：DashScope HTTP（稳定，~300ms）

**切换方式：**

```bash
# .env 文件
TTS_SERVICE=dashscope_realtime  # WebSocket
# 或
TTS_SERVICE=dashscope           # HTTP
# 或
TTS_SERVICE=piper               # 本地
```
