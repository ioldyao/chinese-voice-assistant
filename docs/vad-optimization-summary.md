# VAD 优化总结 - 集成 Pipecat SileroVADAnalyzer

## 优化概述

将自定义 VAD 逻辑替换为 Pipecat 官方的 SileroVADAnalyzer，简化代码并提升准确性。

---

## 改动清单

### 1. ✅ PyAudioTransport - 支持 TransportParams

**文件**: `src/voice_assistant/pyaudio_transport.py`

**改动**：
```python
# 导入 TransportParams
from pipecat.transports.base_transport import BaseTransport, TransportParams

class PyAudioTransport(BaseTransport):
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,
        params: TransportParams = None,  # ✅ 新增参数
    ):
        # ✅ 传递 params 给 BaseTransport
        super().__init__(params or TransportParams())
```

**作用**: 允许传入 VAD 配置到 Transport

---

### 2. ✅ pipecat_main_v2.py - 配置 SileroVADAnalyzer

**文件**: `src/voice_assistant/pipecat_main_v2.py`

**改动**：
```python
# 导入 VAD 模块
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.base_transport import TransportParams

# 创建 VAD Analyzer
vad_analyzer = SileroVADAnalyzer(
    params=VADParams(
        confidence=0.7,      # VAD 置信度阈值
        start_secs=0.2,      # 确认开始说话的时间（快速响应）
        stop_secs=0.8,       # 确认停止说话的时间
        min_volume=0.6,      # 最小音量阈值
    )
)

# 配置 Transport 使用 VAD
transport = PyAudioTransport(
    sample_rate=16000,
    params=TransportParams(
        vad_analyzer=vad_analyzer,  # ✅ 启用 VAD
        audio_in_enabled=True,
        audio_out_enabled=True,
    )
)
```

**作用**:
- 启用 Pipecat 官方 Silero VAD 模型
- 自动检测用户开始/停止说话
- 自动发送 `UserStartedSpeakingFrame` 和 `UserStoppedSpeakingFrame`

---

### 3. ✅ SherpaASRProcessor - 简化（移除自定义 VAD）

**文件**: `src/voice_assistant/pipecat_adapters.py`

**改动前（复杂）**：
```python
class SherpaASRProcessor:
    def __init__(self, asr_model):
        # ❌ 自定义 VAD 参数
        self.silence_threshold = 0.02
        self.max_silence_frames = 20
        self.silence_count = 0
        self.has_speech = False

    async def process_frame(self, frame, direction):
        # ❌ 响应 InterruptionFrame 开始录音
        if isinstance(frame, InterruptionFrame):
            self.recording = True

        if self.recording:
            # ❌ 自己计算音量
            volume = np.sqrt(np.mean(audio_data**2))

            # ❌ 自己判断静音
            if volume >= self.silence_threshold:
                self.has_speech = True
                self.silence_count = 0
            else:
                self.silence_count += 1

            # ❌ 自己判断停止
            if self.silence_count > self.max_silence_frames:
                # 识别...
```

**改动后（简化）**：
```python
class SherpaASRProcessor:
    def __init__(self, asr_model):
        # ✅ 只保留录音状态
        self.recording = False
        self.buffer = []

        # ✅ 添加超时保护
        self.max_record_frames = 300  # 约 10 秒

    async def process_frame(self, frame, direction):
        # ✅ 响应 Pipecat VAD 的开始说话信号
        if isinstance(frame, UserStartedSpeakingFrame):
            print("📝 VAD 检测到开始说话，开始录音...")
            self.recording = True
            self.buffer = []

        # ✅ 响应 Pipecat VAD 的停止说话信号
        if isinstance(frame, UserStoppedSpeakingFrame):
            print("✓ VAD 检测到停止说话，开始识别...")
            # 拼接音频并识别
            full_audio = np.concatenate(self.buffer)
            text = await self._recognize_async(full_audio)
            # 输出 TranscriptionFrame

        # ✅ 录音过程（简化 - 只缓冲音频）
        if self.recording and isinstance(frame, AudioRawFrame):
            self.buffer.append(audio_data)

            # ✅ 超时保护
            if self.frame_count > self.max_record_frames:
                print("⚠️ 录音超时（10秒），强制停止")
                # 强制触发识别...
```

**改进**：
- ❌ 移除 50+ 行自定义 VAD 逻辑
- ✅ 响应 Pipecat VAD 帧
- ✅ 代码从 ~80 行简化到 ~50 行
- ✅ 添加 10 秒超时保护

---

### 4. ✅ SherpaKWSProcessor - 优化唤醒协作流程

**文件**: `src/voice_assistant/pipecat_adapters.py`

**改动前**：
```python
if result:  # 检测到唤醒词
    # ❌ 发送两个帧
    await self.push_frame(UserStartedSpeakingFrame(), direction)
    await self.push_frame(InterruptionFrame(), direction)
```

**改动后**：
```python
if result:  # 检测到唤醒词
    # ✅ 只发送中断帧（中断 TTS）
    # 让 VAD 自动检测用户后续说话
    await self.push_frame(InterruptionFrame(), direction)
```

**改进**：
- 职责明确：KWS 只负责唤醒 + 中断 TTS
- VAD 负责检测用户后续说话
- 避免冲突和重复

---

## 新的工作流程

### 场景：用户说"小智，明天天气怎么样"

```
时间轴：
T0: 持续接收 AudioRawFrame（PyAudioInputProcessor）
    ↓
T1: 用户说"小智"
    ↓
T2: KWS 检测到唤醒词 "小智"
    ├─→ 打印：🔔 检测到唤醒词: 小智
    └─→ 发送：InterruptionFrame（中断 TTS）
    ↓
T3: TTS 收到 InterruptionFrame
    └─→ 停止播放（如果正在播放）
    ↓
T4: 用户继续说"明天天气怎么样"
    ↓
T5: ✅ Pipecat VAD 检测到语音开始
    ├─→ 发送：UserStartedSpeakingFrame
    └─→ ASR 收到 → 开始录音
    ↓
T6: ASR 缓冲音频帧
    ↓
T7: 用户说完，停止
    ↓
T8: ✅ Pipecat VAD 检测到语音停止（静音 > 0.8秒）
    ├─→ 发送：UserStoppedSpeakingFrame
    └─→ ASR 收到 → 停止录音，开始识别
    ↓
T9: ASR 识别完成
    ├─→ 打印：✓ 识别结果: 明天天气怎么样
    └─→ 发送：TranscriptionFrame(text="明天天气怎么样")
    ↓
T10: LLM 处理并生成响应
```

---

## 性能对比

| 方面 | 优化前 | 优化后 |
|------|--------|--------|
| **VAD 实现** | 自定义 RMS 检测 | Pipecat Silero VAD |
| **VAD 准确性** | ⚠️ 简单阈值检测 | ✅ 专业 VAD 模型 |
| **代码复杂度** | ❌ ASR 内部 80+ 行 | ✅ ASR 简化到 50 行 |
| **参数配置** | ❌ 硬编码在 ASR 内部 | ✅ VADParams 统一配置 |
| **环境适应** | ❌ 固定阈值（0.02） | ✅ Silero 自适应 |
| **超时保护** | ❌ 无 | ✅ 10 秒超时 |
| **未来扩展** | ❌ 无法集成 Turn Detection | ✅ 可轻松集成 smart-turn |

---

## VAD 参数说明

```python
VADParams(
    confidence=0.7,      # 置信度阈值（0.0-1.0）
                         # 建议：0.5-0.8，默认 0.7

    start_secs=0.2,      # 确认开始说话的时间（秒）
                         # 建议：0.1-0.3，默认 0.2
                         # 越低越灵敏，但可能误触发

    stop_secs=0.8,       # 确认停止说话的时间（秒）
                         # 建议：0.5-1.0，默认 0.8
                         # 越低响应越快，但可能截断

    min_volume=0.6,      # 最小音量阈值（0.0-1.0）
                         # 建议：0.5-0.7，默认 0.6
                         # 根据环境噪音调整
)
```

---

## 测试建议

### 1. 基础功能测试

运行主程序：
```bash
uv run python -m src.voice_assistant.pipecat_main_v2
```

**测试场景**：
- [ ] 说"小智" → 观察是否检测到唤醒词
- [ ] 继续说"明天天气怎么样" → 观察 VAD 是否检测到语音
- [ ] 观察是否正确识别并响应

**预期日志**：
```
🔔 检测到唤醒词: 小智
📝 VAD 检测到开始说话，开始录音...
✓ VAD 检测到停止说话，开始识别...
✓ 识别结果: 明天天气怎么样
```

### 2. VAD 参数调优测试

如果发现问题：

**问题1: VAD 太灵敏（经常误触发）**
```python
# 提高阈值
VADParams(
    confidence=0.8,      # 提高置信度
    start_secs=0.3,      # 延长确认时间
    min_volume=0.7,      # 提高音量阈值
)
```

**问题2: VAD 不够灵敏（说话未检测到）**
```python
# 降低阈值
VADParams(
    confidence=0.6,      # 降低置信度
    start_secs=0.15,     # 缩短确认时间
    min_volume=0.5,      # 降低音量阈值
)
```

**问题3: 语音被截断（说话被提前截断）**
```python
# 延长停止检测时间
VADParams(
    stop_secs=1.0,       # 增加到 1.0 秒
)
```

### 3. 超时保护测试

**测试场景**：
- [ ] 说"小智"后一直不停说话（超过 10 秒）
- [ ] 观察是否自动停止并识别

**预期日志**：
```
📝 VAD 检测到开始说话，开始录音...
⚠️ 录音超时（10.0秒），强制停止
✓ 识别结果（超时）: [识别到的文本]
```

---

## 潜在问题与解决方案

### 问题1: 唤醒词"小智"被录入识别结果

**现象**：识别结果包含"小智"，如："小智明天天气怎么样"

**原因**：
- 用户说"小智"时，VAD 同时检测到语音
- 在 KWS 检测完成前，VAD 已经开始录音

**解决方案A（推荐）**：在 LLM prompt 中过滤唤醒词
```python
# system prompt 添加：
"""
用户的输入可能包含唤醒词"小智"、"你好助手"等，请忽略这些唤醒词，只处理实际的指令。

示例：
- 用户输入："小智明天天气怎么样" → 理解为："明天天气怎么样"
- 用户输入："你好助手打开浏览器" → 理解为："打开浏览器"
"""
```

**解决方案B（复杂）**：在 ASR 中延迟录音
```python
class SherpaASRProcessor:
    def __init__(self, ...):
        self.wake_word_detected_time = None
        self.ignore_duration = 0.5  # 唤醒后忽略 0.5 秒

    async def process_frame(self, frame, direction):
        # 检测到唤醒词后的短时间内忽略 VAD 信号
        if isinstance(frame, UserStartedSpeakingFrame):
            if self.wake_word_detected_time:
                elapsed = time.time() - self.wake_word_detected_time
                if elapsed < self.ignore_duration:
                    return  # 忽略
            self.recording = True
```

### 问题2: VAD 误触发（环境噪音）

**解决方案**：调整 VAD 参数
```python
VADParams(
    confidence=0.8,      # 提高置信度
    min_volume=0.7,      # 提高音量阈值
)
```

---

## 总结

### ✅ 优化成果

1. **代码简化**：ASR 代码从 80+ 行减少到 50 行
2. **职责明确**：
   - KWS：检测唤醒词 + 中断 TTS
   - VAD：检测语音开始/停止
   - ASR：录音 + 识别
3. **更高准确性**：使用专业 Silero VAD 模型
4. **更好的可维护性**：参数统一配置，易于调优
5. **安全保护**：添加 10 秒超时保护
6. **未来可扩展**：可轻松集成 Turn Detection（smart-turn）

### 📝 下一步

- [ ] 测试基础功能
- [ ] 根据实际环境调优 VAD 参数
- [ ] 如有需要，添加唤醒词过滤逻辑
- [ ] 可选：集成 Turn Detection（smart-turn）以获得更自然的对话体验

---

**优化完成时间**: 2025-12-25
**架构版本**: Pipecat v2.1（集成官方 VAD）
