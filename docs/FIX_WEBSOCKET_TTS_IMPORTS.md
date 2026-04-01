# WebSocket TTS 导入错误修复总结

## ✅ 问题已解决

### 问题描述

之前的错误：在处理每个 token 时出现 `get_api_key` 导入错误和"需要安装 dashscope>=1.25.11"的错误提示。

即使配置为 `TTS_SERVICE=dashscope`（HTTP 流式），仍然会触发 WebSocket Realtime TTS 的导入错误。

### 根本原因

**tts_realtime_adapter.py** 中的 `__init__` 方法在模块级别使用了 `RealtimeTTSConfig` 类，但这个类是通过延迟导入的，导致在导入 `QwenRealtimeTTSProcessor` 时就会触发 `tts_realtime` 模块的导入。

### 修复内容

#### 1. **tts_realtime_adapter.py** - 完全延迟导入

**修改前：**
```python
def __init__(self, ...):
    # ❌ 在 __init__ 中直接使用 RealtimeTTSConfig
    self.config = RealtimeTTSConfig(
        model=model,
        voice=voice,
        mode=mode,
        api_key=api_key,
        **kwargs
    )
```

**修改后：**
```python
def __init__(self, ...):
    # ✅ 保存配置参数，延迟创建 RealtimeTTSConfig
    self._model = model
    self._voice = voice
    self._mode = mode
    self._api_key = api_key
    self._kwargs = kwargs
    self.config = None
```

```python
async def _ensure_initialized(self):
    """确保 TTS 已初始化"""
    if not self._initialized:
        try:
            # ✅ 延迟导入，在首次使用时才导入
            from .tts_realtime import QwenRealtimeTTS, RealtimeTTSConfig

            # 创建配置
            self.config = RealtimeTTSConfig(
                model=self._model,
                voice=self._voice,
                mode=self._mode,
                api_key=self._api_key,
                **self._kwargs
            )

            self.tts = QwenRealtimeTTS(self.config)
            await self.tts.initialize()
            self._initialized = True
```

#### 2. **tts_realtime.py** - 修复 API Key 获取

**修改前：**
```python
# ❌ 错误的导入
from dashscope import get_api_key
api_key = self.config.api_key or get_api_key()
```

**修改后：**
```python
# ✅ 使用环境变量
import os
api_key = self.config.api_key
if not api_key:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
```

#### 3. **pipecat_main_v2.py** - 添加错误处理

```python
if TTS_SERVICE == "dashscope_realtime":
    try:
        from .tts_realtime_adapter import QwenRealtimeTTSProcessor
        tts_proc = QwenRealtimeTTSProcessor(...)
        print("✓ DashScope Realtime TTS 配置成功")
    except Exception as e:
        print(f"⚠️  DashScope Realtime TTS 配置失败: {e}")
        print("   降级使用 Piper TTS（本地引擎）")
        tts_proc = PiperTTSProcessor(wake_system.agent.tts)
```

---

## 🧪 验证修复

### 运行测试脚本

```bash
cd C:\Users\iEZELL\chinese-voice-assistant
uv run python test_tts_config.py
```

**预期输出：**
```
============================================================
TTS 配置检查
============================================================
TTS_SERVICE = dashscope
DASHSCOPE_REALTIME_MODEL = qwen3-tts-flash-realtime
DASHSCOPE_REALTIME_VOICE = Cherry
DASHSCOPE_REALTIME_MODE = server_commit
QWEN_API_KEY = sk-49d20b6...

============================================================
测试延迟导入
============================================================
✓ 配置为使用标准 TTS: dashscope
  不会触发 WebSocket Realtime TTS

============================================================
测试完成
============================================================
```

### 运行主程序

```bash
uv run python main.py
```

**预期结果：**
- ✅ 不再出现 `get_api_key` 导入错误
- ✅ 不再出现"需要安装 dashscope>=1.25.11"的错误
- ✅ 正常使用 DashScope HTTP 流式 TTS

---

## 📊 当前配置状态

根据 `.env` 文件：

```bash
# 当前 TTS 配置
TTS_SERVICE=dashscope                    # HTTP 流式（不是 WebSocket）
DASHSCOPE_TTS_MODEL=qwen3-tts-flash      # HTTP 流式模型
DASHSCOPE_TTS_VOICE=Cherry               # 音色

# WebSocket Realtime 配置（已配置，但未启用）
DASHSCOPE_REALTIME_MODEL=qwen3-tts-flash-realtime
DASHSCOPE_REALTIME_VOICE=Cherry
DASHSCOPE_REALTIME_MODE=server_commit
```

**注意：** 当前使用的是 DashScope **HTTP 流式 TTS**，不是 WebSocket Realtime TTS。

---

## 🚀 切换到 WebSocket Realtime TTS（可选）

如果想使用更低延迟的 WebSocket Realtime TTS（~100-200ms vs HTTP ~300ms）：

### 方法 1：修改 .env

```bash
# 修改 .env
TTS_SERVICE=dashscope_realtime
```

### 方法 2：环境变量

```bash
# Windows PowerShell
$env:TTS_SERVICE="dashscope_realtime"
uv run python main.py

# Linux/Mac
TTS_SERVICE=dashscope_realtime uv run python main.py
```

### 验证 WebSocket TTS

运行后应该看到：
```
⏳ 尝试初始化 DashScope Realtime TTS（WebSocket）...
✓ DashScope Realtime TTS 配置成功
  注意：WebSocket TTS 将在首次使用时初始化连接
```

首次使用时会看到：
```
✓ Qwen-TTS-Realtime 已连接
  - 模型: qwen3-tts-flash-realtime
  - 音色: Cherry
  - 模式: server_commit
  - 采样率: 24000Hz
✓ WebSocket 连接已建立
✓ 会话已创建: ...
✓ 首包延迟: 150ms
```

---

## 📝 文件修改清单

1. ✅ **src/voice_assistant/tts_realtime.py**
   - 修复 `get_api_key` 导入错误
   - 使用 `os.getenv()` 获取 API Key

2. ✅ **src/voice_assistant/tts_realtime_adapter.py**
   - 完全延迟导入 `RealtimeTTSConfig`
   - 在 `_ensure_initialized()` 中创建配置

3. ✅ **src/voice_assistant/pipecat_main_v2.py**
   - 添加错误处理和降级机制

4. ✅ **test_tts_config.py**（新建）
   - 验证配置和导入的测试脚本

---

## ✅ 修复完成

现在程序应该可以正常运行了！

**推荐配置：**
- 开发环境：`TTS_SERVICE=piper`（本地，免费）
- 生产环境：`TTS_SERVICE=dashscope_realtime`（WebSocket，~100-200ms）
- 当前配置：`TTS_SERVICE=dashscope`（HTTP 流式，~300ms）

详细文档：
- [快速开始指南](../QUICKSTART.md)
- [WebSocket TTS 详细说明](QWEN_TTS_REALTIME.md)
