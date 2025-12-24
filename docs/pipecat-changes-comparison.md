# Pipecat 架构修复 - 关键改动对比

## 1. Transport 层

### 修改前 ❌
```python
# 手动创建音频循环（pipecat_main.py:354-395）
class SimplePyAudioTransport:
    def __init__(self, sample_rate=16000):
        self.p = pyaudio.PyAudio()
        self.input_stream = self.p.open(...)
        self.output_stream = self.p.open(...)

    async def read_audio_frames(self):
        while self.running:
            audio_bytes = await asyncio.to_thread(
                self.input_stream.read, self.chunk_size
            )
            yield AudioRawFrame(audio=audio_bytes, ...)

# 手动推送音频帧（pipecat_main.py:379-382）
async def audio_input_loop():
    async for audio_frame in transport.read_audio_frames():
        await task.queue_frames([audio_frame])  # 外部循环
```

### 修改后 ✅
```python
# 标准 Transport 接口（pyaudio_transport.py）
class PyAudioTransport(BaseTransport):
    def input(self) -> FrameProcessor:
        return PyAudioInputProcessor(self)

    def output(self) -> FrameProcessor:
        return PyAudioOutputProcessor(self)

# Pipeline 自动处理（pipecat_main_v2.py:139）
pipeline = Pipeline([
    transport.input(),    # ✅ 自动推送 AudioRawFrame
    # ...
    transport.output(),   # ✅ 自动播放 OutputAudioRawFrame
])
```

**差异**：
- ✅ 符合 `BaseTransport` 接口
- ✅ Pipeline 自动管理音频流
- ✅ 易于切换到其他 Transport（Daily.co、WebSocket）

---

## 2. TTS 处理

### 修改前 ❌
```python
# 直接播放到音频流（pipecat_adapters.py:414-416）
def _synthesize_and_play_sync(self, text):
    audio_generator = self.tts.piper_voice.synthesize(text)
    for chunk in audio_generator:
        audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)

        # ❌ 直接写入音频流（绕过 Pipeline）
        if self.transport and self.transport.output_stream:
            self.transport.output_stream.write(audio_int16.tobytes())
```

### 修改后 ✅
```python
# 生成标准 Frame（pipecat_adapters.py:421-438）
def _synthesize_and_push_sync(self, text):
    audio_generator = self.tts.piper_voice.synthesize(text)
    for chunk in audio_generator:
        audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)

        # ✅ 生成标准 OutputAudioRawFrame
        audio_frame = OutputAudioRawFrame(
            audio=audio_int16.tobytes(),
            sample_rate=sample_rate,
            num_channels=1
        )

        # ✅ 推送到 Pipeline（让 transport.output() 播放）
        loop = asyncio.get_event_loop()
        future = asyncio.run_coroutine_threadsafe(
            self.push_frame(audio_frame, FrameDirection.DOWNSTREAM),
            loop
        )
        future.result(timeout=1.0)
```

**差异**：
- ✅ 生成标准 `OutputAudioRawFrame`
- ✅ 流经 Pipeline（可监控、可中断）
- ✅ 让 `transport.output()` 负责实际播放

---

## 3. Vision 处理

### 修改前 ❌
```python
# 推送新的 TranscriptionFrame（pipecat_adapters.py:623-632）
class QwenVisionProcessor(FrameProcessor):
    async def process_frame(self, frame, direction):
        if isinstance(frame, UserImageRawFrame):
            result = await self._call_vision_api(...)

            # ❌ 推送新 Frame（不符合标准）
            await self.push_frame(
                TranscriptionFrame(
                    text=f"[视觉观察] {result}",
                    user_id="system",
                    timestamp=self._get_timestamp()
                ),
                direction
            )
```

### 修改后 ✅
```python
# 直接修改 context（pipecat_adapters.py:580-617）
class VisionProcessor(FrameProcessor):
    def __init__(self, api_url, api_key, context):
        self.context = context  # ✅ 接收 LLMContext

    async def process_frame(self, frame, direction):
        # ✅ 检查 context 中的最后一条用户消息
        if self.context.messages:
            last_message = self.context.messages[-1]
            if last_message.get("role") == "user":
                text_content = last_message.get("content", "")

                if self._needs_vision(text_content):
                    result = await self._call_vision_api(...)

                    # ✅ 直接修改 context（LLM 自动看到）
                    self.context.messages.append({
                        "role": "system",
                        "content": f"[视觉观察] {result}"
                    })

        # ✅ 传递所有帧（不推送新 Frame）
        await self.push_frame(frame, direction)
```

**差异**：
- ✅ 直接修改 `context.messages`
- ✅ 无需推送额外 Frame
- ✅ 符合 Pipecat 标准模式

---

## 4. Pipeline 顺序

### 修改前 ❌
```python
# pipecat_main.py:320-329
pipeline = Pipeline([
    kws_proc,                       # 1. KWS
    asr_proc,                       # 2. ASR
    screenshot_proc,                # 3. Screenshot（判断 vision）
    qwen_vision_proc,               # 4. Vision API
    user_aggregator,                # 5. ❌ 在 vision 之后添加用户消息
    llm,                            # 6. LLM
    tts_proc,                       # 7. TTS（直接播放）
    assistant_aggregator,           # 8. ❌ 在 TTS 之后保存助手响应
])
```

### 修改后 ✅
```python
# pipecat_main_v2.py:139-149
pipeline = Pipeline([
    transport.input(),              # 1. ✅ 官方音频输入
    kws_proc,                       # 2. KWS
    asr_proc,                       # 3. ASR
    user_aggregator,                # 4. ✅ 紧跟 ASR（先添加用户消息）
    vision_proc,                    # 5. ✅ 读取 context 判断 vision
    llm,                            # 6. LLM
    assistant_aggregator,           # 7. ✅ 紧跟 LLM（保存助手响应）
    tts_proc,                       # 8. TTS（生成 Frame）
    transport.output(),             # 9. ✅ 官方音频输出
])
```

**官方推荐顺序**：
```
transport.input() → stt → context_aggregator.user() → llm → tts → transport.output() → context_aggregator.assistant()
```

**差异**：
- ✅ `user_aggregator` 在 vision 之前（context 先包含用户消息）
- ✅ `assistant_aggregator` 紧跟 LLM（正确保存工具调用历史）
- ✅ TTS 在 aggregator 之后（语义清晰）

---

## 5. 主程序结构

### 修改前 ❌
```python
# pipecat_main.py:354-414
async def run_pipeline_with_audio(pipeline, transport):
    task = PipelineTask(pipeline, ...)
    await task.queue_frames([StartFrame()])

    # ❌ 外部音频输入循环
    async def audio_input_loop():
        async for audio_frame in transport.read_audio_frames():
            await task.queue_frames([audio_frame])

    runner = PipelineRunner()
    audio_task = asyncio.create_task(audio_input_loop())

    try:
        await runner.run(task)
    finally:
        audio_task.cancel()
```

### 修改后 ✅
```python
# pipecat_main_v2.py:185-207
async def main():
    # ✅ 创建 Pipeline
    pipeline, transport, wake_system, mcp = await create_pipecat_pipeline()

    # ✅ 创建 PipelineTask
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
        )
    )

    await task.queue_frames([StartFrame()])

    # ✅ 运行 Pipeline（官方方式，无需外部循环）
    runner = PipelineRunner()
    await runner.run(task)
```

**差异**：
- ✅ 无需外部音频循环（`transport.input()` 自动处理）
- ✅ 更简洁、更符合官方示例
- ✅ 易于维护和调试

---

## 总结

| 组件 | 修改前 | 修改后 | 符合标准 |
|------|--------|--------|---------|
| **Transport** | 手动循环 | 标准接口 | ✅ |
| **TTS** | 直接播放 | 生成 Frame | ✅ |
| **Vision** | 推送 Frame | 修改 Context | ✅ |
| **Pipeline 顺序** | 不标准 | 官方推荐 | ✅ |
| **主程序** | 复杂 | 简洁 | ✅ |

**核心改进**：
1. ✅ 所有数据流经 Pipeline（无绕过）
2. ✅ 使用标准 Frame 类型
3. ✅ 符合官方架构模式
4. ✅ 易于切换服务和 Transport
5. ✅ 更好的调试和监控
