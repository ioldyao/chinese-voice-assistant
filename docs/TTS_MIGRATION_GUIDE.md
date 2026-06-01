# 从 HTTP 流式切换到 WebSocket Realtime 指南

## 概述

本指南帮助你从 DashScope HTTP 流式 TTS 切换到 WebSocket Realtime TTS，获得更低的延迟（~100-200ms vs ~300ms）。

## 快速切换（3 步）

### 1. 更新依赖

```bash
# 确保 dashscope >= 1.25.11
uv add dashscope --upgrade
```

### 2. 修改配置

编辑 `.env` 文件：

```bash
# 从这个
TTS_SERVICE=dashscope
DASHSCOPE_TTS_MODEL=qwen3-tts-flash
DASHSCOPE_TTS_VOICE=Cherry

# 改为这个
TTS_SERVICE=dashscope_realtime
DASHSCOPE_REALTIME_MODEL=qwen3-tts-flash-realtime
DASHSCOPE_REALTIME_VOICE=Cherry
DASHSCOPE_REALTIME_MODE=server_commit
```

### 3. 重启程序

```bash
uv run python main.py
```

## 详细对比

### HTTP 流式（原有实现）

```python
# 原有代码（pipecat_adapters.py）
response = self.tts.dashscope.MultiModalConversation.call(
    model="qwen3-tts-flash",
    text=text,
    voice="Cherry",
    stream=True
)

for chunk in response:
    audio_b64 = chunk.output.audio.data
    audio_data = base64.b64decode(audio_b64)
    # 播放音频
```

**特点**：
- ✅ 简单易用
- ✅ 稳定可靠
- ❌ 首包延迟 ~300ms
- ❌ 每次请求建立新连接

### WebSocket Realtime（新实现）

```python
# 新代码（tts_realtime.py）
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime

tts = QwenTtsRealtime(
    model="qwen3-tts-flash-realtime",
    callback=MyCallback(),
    url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
)

tts.connect()
tts.update_session(voice="Cherry", mode="server_commit")
tts.append_text("你好，世界")
tts.finish()
```

**特点**：
- ✅ 首包延迟 ~100-200ms
- ✅ 持久连接，复用性高
- ✅ 事件驱动架构
- ❌ 相对复杂

## 配置参数对照

### HTTP 流式参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `DASHSCOPE_TTS_MODEL` | 模型名称 | `qwen3-tts-flash` |
| `DASHSCOPE_TTS_VOICE` | 音色 | `Cherry` |

### WebSocket Realtime 参数

| 参数 | 说明 | 示例 | 必需 |
|------|------|------|------|
| `DASHSCOPE_REALTIME_MODEL` | 模型名称 | `qwen3-tts-flash-realtime` | ✅ |
| `DASHSCOPE_REALTIME_VOICE` | 音色 | `Cherry` | ✅ |
| `DASHSCOPE_REALTIME_MODE` | 交互模式 | `server_commit` | ✅ |
| `DASHSCOPE_REALTIME_URL` | WebSocket URL | `wss://...` | ❌ |
| `DASHSCOPE_REALTIME_LANGUAGE_TYPE` | 语种 | `Chinese` | ❌ |
| `DASHSCOPE_REALTIME_SPEECH_RATE` | 语速 | `1.0` | ❌ |
| `DASHSCOPE_REALTIME_VOLUME` | 音量 | `50` | ❌ |
| `DASHSCOPE_REALTIME_PITCH_RATE` | 语调 | `1.0` | ❌ |

## 模型名称对照

| HTTP 流式 | WebSocket Realtime | 说明 |
|-----------|-------------------|------|
| `qwen3-tts-flash` | `qwen3-tts-flash-realtime` | 快速高质量 |
| `qwen3-tts-instruct-flash` | `qwen3-tts-instruct-flash-realtime` | 支持指令控制 |
| `qwen-audio-turbo` | - | 仅 HTTP 支持 |
| `sambert-zhichu-v1` | `qwen-tts-realtime` | 经典模型 |

## 交互模式说明

### server_commit（推荐）

**适用场景**：大多数实时对话场景

**特点**：
- 服务端自动判断合成时机
- 平衡延迟与质量
- 简单易用

**使用方式**：
```python
tts = create_realtime_tts(mode="server_commit")
await tts.speak("你好")
await tts.speak("世界")
# 系统自动合并和分段
```

### commit（高级）

**适用场景**：需要精细控制的场景

**特点**：
- 客户端手动控制合成时机
- 延迟最低
- 需要额外代码支持

**使用方式**：
```python
tts = create_realtime_tts(mode="commit")
await tts.append_text("你好")
await tts.append_text("世界")
await tts.commit()  # 手动触发合成
```

## 代码迁移示例

### 场景 1：基础 TTS

**HTTP 流式**：
```python
from voice_assistant.tts import TTSManager

tts = TTSManager(engine_type="dashscope")
tts.speak("你好，世界")
```

**WebSocket Realtime**：
```python
from voice_assistant.tts_realtime import create_realtime_tts

tts = create_realtime_tts(
    model="qwen3-tts-flash-realtime",
    voice="Cherry",
    mode="server_commit"
)
await tts.initialize()
await tts.speak("你好，世界")
await tts.close()
```

### 场景 2：Pipecat Pipeline 集成

**HTTP 流式**（自动）：
```python
# .env 配置
TTS_SERVICE=dashscope

# Pipeline 自动使用
pipeline = Pipeline([
    llm,
    tts_proc,  # 自动使用 DashScope HTTP
    transport.output()
])
```

**WebSocket Realtime**（自动）：
```python
# .env 配置
TTS_SERVICE=dashscope_realtime

# Pipeline 自动使用
pipeline = Pipeline([
    llm,
    tts_proc,  # 自动使用 WebSocket Realtime
    transport.output()
])
```

## 性能测试

### 测试首包延迟

```bash
# 运行测试脚本
uv run python test_realtime_tts.py

# 查看输出
✓ 首包延迟: 150ms
```

### 预期性能对比

| 场景 | HTTP 流式 | WebSocket Realtime |
|------|-----------|-------------------|
| **短文本（<20字）** | ~300ms | ~100ms ⚡ |
| **长文本（>100字）** | ~400ms | ~200ms ⚡ |
| **多轮对话** | 每次 ~300ms | 首次 ~100ms，后续 ~50ms ⚡ |

## 常见问题

### Q1: 切换后无法启动？

检查：
1. dashscope 版本是否 >= 1.25.11
2. API Key 是否正确
3. 网络是否可以访问 WebSocket

```bash
# 检查版本
uv run python -c "import dashscope; print(dashscope.__version__)"

# 检查网络
curl -I https://dashscope.aliyuncs.com
```

### Q2: 音质有变化吗？

答：没有变化。WebSocket 和 HTTP 使用相同的模型和音色，音质完全一致。

### Q3: 可以回退到 HTTP 流式吗？

答：可以。只需修改 `.env` 配置：

```bash
TTS_SERVICE=dashscope  # 改回 HTTP
```

### Q4: commit 模式如何使用？

答：commit 模式需要手动控制合成时机，当前简化版本使用 server_commit。

```python
# 当前实现
tts = create_realtime_tts(mode="server_commit")
await tts.speak("文本")

# commit 模式需要额外实现
# 参考：test_realtime_tts.py 中的 test_commit_mode()
```

### Q5: 支持哪些高级功能？

**WebSocket Realtime 独有**：
- ✅ 指令控制（语速、语调、音量）
- ✅ 持久连接复用
- ✅ 更低的首包延迟

**两者都支持**：
- ✅ 流式播放
- ✅ 多种音色
- ✅ 多种音频格式

## 推荐配置

### 开发环境

```bash
# 使用本地 Piper TTS（最快，免费）
TTS_SERVICE=piper
```

### 生产环境

```bash
# 使用 WebSocket Realtime（低延迟，高质量）
TTS_SERVICE=dashscope_realtime
DASHSCOPE_REALTIME_MODEL=qwen3-tts-flash-realtime
DASHSCOPE_REALTIME_VOICE=Cherry
DASHSCOPE_REALTIME_MODE=server_commit
```

### 备用方案

```bash
# 使用 HTTP 流式（稳定可靠）
TTS_SERVICE=dashscope
DASHSCOPE_TTS_MODEL=qwen3-tts-flash
DASHSCOPE_TTS_VOICE=Cherry
```

## 总结

**切换建议**：

1. ✅ **新项目**：直接使用 WebSocket Realtime
2. ✅ **实时对话**：推荐 WebSocket Realtime
3. ⚠️ **批量合成**：保持 HTTP 流式
4. ⚠️ **离线环境**：使用 Piper TTS

**核心优势**：

- 🚀 **更低延迟**：~100-200ms vs ~300ms
- 🔌 **持久连接**：复用连接，减少开销
- 🎯 **事件驱动**：更灵活的控制
- 🎭 **指令控制**：语速、语调、音量等精细控制

**相关资源**：

- [完整使用指南](QWEN_TTS_REALTIME.md)
- [配置示例](../.env.realtime.example)
- [测试代码](../test_realtime_tts.py)
- [官方文档](https://help.aliyun.com/zh/model-studio/qwen-tts-realtime)
