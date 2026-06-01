# 错误修复说明

## 问题：NameError: name 'TTS_SERVICE' is not defined

### ✅ 已修复

我已经修复了导入错误，问题出在 `pipecat_main_v2.py` 和 `react_agent.py` 中使用了 `TTS_SERVICE` 变量但没有导入。

### 🔧 修复内容

#### 1. `pipecat_main_v2.py` - 添加缺失的导入

```python
# 添加了这些配置变量的导入
from .config import (
    ...
    TTS_SERVICE,
    DASHSCOPE_REALTIME_MODEL,
    DASHSCOPE_REALTIME_VOICE,
    DASHSCOPE_REALTIME_MODE,
)
```

#### 2. `pipecat_main_v2.py` - 添加错误处理

```python
# WebSocket Realtime TTS 初始化失败时降级使用 Piper TTS
if TTS_SERVICE == "dashscope_realtime":
    try:
        tts_proc = QwenRealtimeTTSProcessor(...)
        print("✓ DashScope Realtime TTS 已初始化")
    except Exception as e:
        print(f"⚠️  DashScope Realtime TTS 初始化失败: {e}")
        print("   降级使用 Piper TTS")
        tts_proc = PiperTTSProcessor(wake_system.agent.tts)
```

#### 3. `react_agent.py` - 添加 dashscope_realtime 支持

```python
# ReactAgent 暂不支持 WebSocket Realtime
# 自动降级使用 DashScope HTTP 流式
tts_config = {
    ...
    "dashscope_realtime": {
        # 降级使用 DashScope HTTP 流式
        "engine_type": "dashscope",
        ...
    },
    ...
}
```

---

## 🚀 现在可以使用 WebSocket Realtime TTS 了！

### 方式 1：快速测试（使用 Piper TTS，默认）

```bash
# .env 配置（默认）
TTS_SERVICE=piper

# 运行程序
uv run python main.py
```

### 方式 2：使用 WebSocket Realtime TTS（推荐）

```bash
# 1. 修改 .env 配置
TTS_SERVICE=dashscope_realtime
DASHSCOPE_REALTIME_MODEL=qwen3-tts-flash-realtime
DASHSCOPE_REALTIME_VOICE=Cherry
DASHSCOPE_REALTIME_MODE=server_commit

# 2. 确保 QWEN_API_KEY 已配置
QWEN_API_KEY=your-qwen-api-key-here

# 3. 运行程序
uv run python main.py
```

### 方式 3：使用 DashScope HTTP 流式（稳定）

```bash
# .env 配置
TTS_SERVICE=dashscope
DASHSCOPE_TTS_MODEL=qwen3-tts-flash
DASHSCOPE_TTS_VOICE=Cherry

# 运行程序
uv run python main.py
```

---

## 📊 TTS 引擎对比

| 引擎 | 首包延迟 | 状态 | 推荐场景 |
|------|---------|------|---------|
| **Piper** | <100ms | ✅ 完全支持 | 开发环境 ⭐ |
| **DashScope Realtime** | ~100-200ms | ✅ Pipeline 支持 | 生产环境 🆕 |
| **DashScope HTTP** | ~300ms | ✅ 完全支持 | 稳定可靠 |
| **Edge TTS** | ~500ms | ✅ 完全支持 | 备用方案 |
| **Azure TTS** | ~400ms | ✅ 完全支持 | 高质量需求 |

**注意**：ReactAgent（用于浏览器控制）暂不支持 WebSocket Realtime，会自动降级使用 DashScope HTTP。

---

## 🔍 验证修复

运行以下命令验证配置导入是否正常：

```bash
uv run python -c "from src.voice_assistant.config import TTS_SERVICE; print('OK:', TTS_SERVICE)"
```

预期输出：
```
OK: piper
```

或

```bash
uv run python main.py
```

预期输出：
```
正在初始化智能语音助手...
📋 加载关键词: ...
✓ 使用 Piper TTS（本地，超快）...
✓ KWS模型已加载
✓ ASR模型已加载
✓ React Agent 已创建（MCP 将稍后异步启动）
✓ KWS/ASR 模型已加载
...
```

---

## ❓ 常见问题

### Q1: 为什么 ReactAgent 不支持 WebSocket Realtime？

A: ReactAgent 用于浏览器控制等任务，它会异步调用 MCP 工具。WebSocket Realtime TTS 是为 Pipeline 设计的，两者架构不同。ReactAgent 会自动降级使用 DashScope HTTP 流式。

### Q2: 我看到警告 "ReactAgent 暂不支持 WebSocket Realtime TTS"？

A: 这是正常的。ReactAgent 会自动降级使用 DashScope HTTP 流式，不影响功能使用。

### Q3: Pipeline 中的 TTS 使用的是哪个引擎？

A: 取决于 `.env` 中的 `TTS_SERVICE` 配置：
- `piper` → Piper TTS（本地）
- `dashscope_realtime` → WebSocket Realtime TTS 🆕
- `dashscope` → DashScope HTTP 流式
- `edge` → Edge TTS
- `azure` → Azure TTS

### Q4: 如何切换回 Piper TTS？

A: 修改 `.env` 配置：
```bash
TTS_SERVICE=piper
```

---

## ✅ 修复完成

现在程序应该可以正常运行了！推荐配置：

```bash
# 开发环境（免费）
TTS_SERVICE=piper

# 生产环境（推荐）⭐
TTS_SERVICE=dashscope_realtime
```

详细文档：
- [快速开始指南](QUICKSTART.md)
- [配置方案对比](docs/CONFIG_COMPARISON.md)
- [WebSocket TTS 详细说明](docs/QWEN_TTS_REALTIME.md)
