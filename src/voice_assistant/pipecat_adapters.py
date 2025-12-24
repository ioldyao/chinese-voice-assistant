"""Pipecat 适配器 - 封装现有组件为 Pipecat Processors"""
import asyncio
import numpy as np
import tempfile
import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from PIL import ImageGrab

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TextFrame,
    TTSAudioRawFrame,
    OutputAudioRawFrame,  # ✅ TTS 音频输出帧
    InterruptionFrame,  # ✅ 官方中断帧
    TTSStoppedFrame,    # ✅ 官方 TTS 停止帧
    TranscriptionFrame,  # ✅ 用于 LLMUserContextAggregator
    UserStartedSpeakingFrame,  # ✅ 用户开始说话
    UserStoppedSpeakingFrame,  # ✅ 触发 LLM 处理
    EndFrame,
)


# ==================== Sherpa-ONNX KWS Processor ====================

class SherpaKWSProcessor(FrameProcessor):
    """
    Sherpa-ONNX KWS 适配器（简化版 - 配合 Pipecat VAD）

    改进：
    - ✅ 检测到唤醒词时发送 InterruptionFrame（中断 TTS）
    - ✅ 依赖 Pipecat VAD 自动检测用户后续说话
    - ✅ 职责明确，逻辑简化
    """

    def __init__(self, kws_model):
        super().__init__()
        self.kws_model = kws_model
        self.kws_stream = kws_model.create_stream()
        self.sample_rate = 16000
        self.is_awake = False  # 用于并行 Pipeline 的条件判断
        self.last_keyword = None  # 保存最后检测到的唤醒词

    async def process_frame(self, frame, direction):
        """处理音频帧，检测唤醒词"""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            # 提取音频数据
            audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0

            # 喂入 KWS 模型
            self.kws_stream.accept_waveform(self.sample_rate, audio_data)

            # 检测关键词
            while self.kws_model.is_ready(self.kws_stream):
                self.kws_model.decode_stream(self.kws_stream)

            result = self.kws_model.get_result(self.kws_stream)

            if result:
                print(f"🔔 检测到唤醒词: {result}")
                self.is_awake = True
                self.last_keyword = result

                # ✅ 只发送中断帧（中断 TTS，如果正在播放）
                # 注意：不发送 UserStartedSpeakingFrame，让 VAD 自动检测用户后续说话
                await self.push_frame(
                    InterruptionFrame(),
                    direction
                )

                # 重置 KWS 流
                self.kws_stream = self.kws_model.create_stream()

            # 继续传递音频帧（供后续处理器使用）
            await self.push_frame(frame, direction)
        else:
            # 其他帧直接传递
            await self.push_frame(frame, direction)


# ==================== Sherpa-ONNX ASR Processor ====================

class SherpaASRProcessor(FrameProcessor):
    """
    Sherpa-ONNX ASR 适配器（临时版本 - 内置简单 VAD）

    注意：
    - ⚠️ 当前使用简单的 RMS VAD（因为 Pipecat VAD 暂时禁用）
    - ✅ 响应 InterruptionFrame 开始录音（来自 KWS）
    - ✅ 使用音量检测判断停止说话
    - ✅ 添加超时保护

    TODO: 集成 Pipecat SileroVADAnalyzer 后移除此逻辑
    """

    def __init__(self, asr_model, sample_rate=16000):
        super().__init__()
        self.asr_model = asr_model
        self.sample_rate = sample_rate

        # 录音状态
        self.recording = False
        self.buffer = []
        self.frame_count = 0

        # ⚠️ 临时使用简单 VAD（待替换为 Pipecat VAD）
        self.silence_threshold = 0.02  # RMS 阈值
        self.max_silence_frames = 20   # 约 0.64 秒静音
        self.silence_count = 0
        self.has_speech = False

        # ✅ 超时保护（防止无限录音）
        self.max_record_frames = 300  # 约 10 秒（300帧 × 32ms）
        self.max_record_duration = 10.0  # 秒

    async def process_frame(self, frame, direction):
        """处理音频帧，识别语音"""
        await super().process_frame(frame, direction)

        # ✅ 响应 KWS 的中断信号（开始录音）
        if isinstance(frame, InterruptionFrame):
            print("📝 检测到唤醒词，开始录音...")
            self.recording = True
            self.buffer = []
            self.frame_count = 0
            self.silence_count = 0
            self.has_speech = False
            await self.push_frame(frame, direction)
            return

        # ✅ 录音过程（使用简单 RMS VAD）
        if self.recording and isinstance(frame, AudioRawFrame):
            # 提取音频数据
            audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            self.buffer.append(audio_data)
            self.frame_count += 1

            # ⚠️ 简单 VAD：计算音量
            volume = np.sqrt(np.mean(audio_data**2))

            if volume >= self.silence_threshold:
                self.has_speech = True
                self.silence_count = 0
            else:
                if self.has_speech:  # 只有在检测到语音后才计算静音
                    self.silence_count += 1

            # 检查是否应该停止录音（静音判断）
            if self.has_speech and self.silence_count > self.max_silence_frames:
                print(f"✓ 检测到静音（{self.silence_count} 帧），开始识别...")

                # 拼接音频
                full_audio = np.concatenate(self.buffer)

                # ASR 识别
                text = await self._recognize_async(full_audio)

                if text:
                    print(f"✓ 识别结果: {text}")
                    # ✅ 使用 TranscriptionFrame（LLMUserContextAggregator 期望的类型）
                    await self.push_frame(
                        TranscriptionFrame(text=text, user_id="user", timestamp=self._get_timestamp()),
                        direction
                    )

                # 重置状态
                self.recording = False
                self.buffer = []
                self.frame_count = 0
                self.silence_count = 0
                self.has_speech = False
                return

            # ✅ 超时保护
            if self.frame_count > self.max_record_frames:
                print(f"⚠️ 录音超时（{self.max_record_duration}秒），强制停止")
                # 强制触发识别
                if self.buffer:
                    full_audio = np.concatenate(self.buffer)
                    text = await self._recognize_async(full_audio)
                    if text:
                        print(f"✓ 识别结果（超时）: {text}")
                        await self.push_frame(
                            TranscriptionFrame(text=text, user_id="user", timestamp=self._get_timestamp()),
                            direction
                        )

                # 重置状态
                self.recording = False
                self.buffer = []
                self.frame_count = 0
                self.silence_count = 0
                self.has_speech = False
                return

            # 继续传递音频帧
            await self.push_frame(frame, direction)
        else:
            # 其他帧直接传递
            await self.push_frame(frame, direction)

    async def _recognize_async(self, audio_data):
        """异步 ASR 识别（在线程池中执行）"""
        def _recognize_sync():
            # 创建 ASR 流
            asr_stream = self.asr_model.create_stream()
            asr_stream.accept_waveform(self.sample_rate, audio_data)
            self.asr_model.decode_stream(asr_stream)
            return asr_stream.result.text.strip()

        # 在线程池中执行（避免阻塞事件循环）
        return await asyncio.to_thread(_recognize_sync)

    def _get_timestamp(self):
        """获取当前时间戳（ISO 8601 格式）"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


# ==================== React Agent Processor ====================

class ReactAgentProcessor(FrameProcessor):
    """
    React Agent Processor - 基于 MCP 官方推荐模式

    使用完全异步的实现：
    - 直接调用 agent.execute_command_async()
    - 在主事件循环中执行 MCP 调用
    - 后台任务执行，不阻塞 Pipeline
    - 响应官方 InterruptionFrame 中断信号
    """

    def __init__(self, react_agent):
        """
        初始化 React Agent Processor

        Args:
            react_agent: ReactAgent 实例（已初始化 MCP）
        """
        super().__init__()
        self.agent = react_agent
        self.current_task = None
        self.cancel_flag = False

    async def process_frame(self, frame, direction):
        """处理帧"""
        await super().process_frame(frame, direction)

        # ✅ 响应官方中断帧，取消当前任务
        if isinstance(frame, InterruptionFrame):
            if self.current_task and not self.current_task.done():
                print("⏸️  检测到中断信号，取消当前 Agent 任务")
                self.cancel_flag = True
                self.current_task.cancel()
            await self.push_frame(frame, direction)
            return

        # 处理文本命令
        if isinstance(frame, TextFrame):
            print(f"🤖 React Agent 处理: {frame.text}")
            self.cancel_flag = False

            # 创建后台任务（不等待完成）
            self.current_task = asyncio.create_task(
                self._execute_and_push_result(frame.text, direction)
            )

            # 立即传递帧，不阻塞 Pipeline
            await self.push_frame(frame, direction)
        else:
            # 其他帧直接传递
            await self.push_frame(frame, direction)

    async def _execute_and_push_result(self, command: str, direction):
        """
        执行命令并推送结果（后台任务）

        基于 MCP 官方推荐模式：直接异步调用，无需线程
        """
        try:
            print(f"✨ 开始执行命令: {command}")

            # 使用官方推荐的异步方式直接调用
            result = await self.agent.execute_command_async(command, enable_voice=False)

            print(f"✅ 执行完成: success={result.get('success')}")

            # 检查是否被取消
            if self.cancel_flag:
                print("⏸️  任务已取消，不推送结果")
                return

            # 推送结果文本（如果成功）
            if result.get("success") and result.get("message"):
                result_frame = TextFrame(result["message"])
                await self.push_frame(result_frame, direction)

        except asyncio.CancelledError:
            print("⏸️  任务被取消")
            raise
        except Exception as e:
            print(f"❌ 执行异常: {e}")
            import traceback
            traceback.print_exc()



# ==================== Piper TTS Processor ====================

class PiperTTSProcessor(FrameProcessor):
    """
    Piper TTS 适配器 - 句子级流式播放（符合 Pipecat 标准）

    改进：
    - ✅ 生成标准 OutputAudioRawFrame（不直接播放）
    - ✅ 让 transport.output() 负责实际播放
    - ✅ 支持中断和句子级缓冲
    """

    def __init__(self, tts_manager):
        super().__init__()
        self.tts = tts_manager
        self.interrupt_flag = False  # 中断标志
        self.is_speaking = False  # TTS 播放状态

        # 句子缓冲区（按句子流式播放）
        self.sentence_buffer = ""
        self.sentence_delimiters = ["。", "！", "？", ".", "!", "?", "\n"]

    def interrupt(self):
        """中断当前TTS播放"""
        self.interrupt_flag = True

    async def process_frame(self, frame, direction):
        """处理文本帧，生成 TTS 音频"""
        await super().process_frame(frame, direction)

        # 响应官方中断帧，设置中断
        if isinstance(frame, InterruptionFrame):
            if self.is_speaking:
                print("⏸️  检测到中断信号，停止TTS播放")
                self.interrupt()
            # 清空缓冲区
            self.sentence_buffer = ""
            # 传递中断帧
            await self.push_frame(frame, direction)
            return

        # 检测 LLM 响应结束（播放剩余缓冲）
        from pipecat.frames.frames import LLMFullResponseEndFrame
        if isinstance(frame, LLMFullResponseEndFrame):
            # 播放剩余的不完整句子
            if self.sentence_buffer.strip():
                print(f"🔊 TTS 合成（剩余）: {self.sentence_buffer}")
                await self._synthesize_and_push(self.sentence_buffer)
                self.sentence_buffer = ""

            # 发送 turn 结束信号
            if not self.is_speaking:
                await self.push_frame(UserStoppedSpeakingFrame(), direction)

            await self.push_frame(frame, direction)
            return

        # 流式处理文本帧（句子级缓冲）
        if isinstance(frame, TextFrame):
            self.sentence_buffer += frame.text

            # 检查是否有完整句子
            for delimiter in self.sentence_delimiters:
                if delimiter in self.sentence_buffer:
                    # 分割句子
                    parts = self.sentence_buffer.split(delimiter, 1)
                    sentence = parts[0] + delimiter  # 包含标点符号
                    self.sentence_buffer = parts[1] if len(parts) > 1 else ""

                    # 立即合成完整句子
                    print(f"🔊 TTS 合成: {sentence.strip()}")
                    await self._synthesize_and_push(sentence.strip())
                    break

            # 传递原始文本帧
            await self.push_frame(frame, direction)
        else:
            # 其他帧直接传递
            await self.push_frame(frame, direction)

    async def _synthesize_and_push(self, text: str):
        """
        异步合成并推送音频帧（符合 Pipecat 标准）

        改进：生成 OutputAudioRawFrame 而不是直接播放
        """
        if not text:
            return

        self.interrupt_flag = False
        self.is_speaking = True

        # 在线程池中执行合成
        was_interrupted = await asyncio.to_thread(
            self._synthesize_and_push_sync,
            text
        )

        self.is_speaking = False
        if was_interrupted:
            print("⏸️  TTS 已被中断")
            await self.push_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

    def _synthesize_and_push_sync(self, text: str) -> bool:
        """
        同步 TTS 合成（在线程池中执行），支持中断

        改进：生成 OutputAudioRawFrame 而不是直接播放

        Returns:
            bool: 是否被中断
        """
        if self.tts.engine_type == "piper":
            try:
                # 使用 Piper TTS 生成音频
                audio_generator = self.tts.piper_voice.synthesize(text)

                for chunk in audio_generator:
                    # 检查中断标志
                    if self.interrupt_flag:
                        print("⏸️  TTS 合成已中断")
                        return True  # 返回中断标志

                    # 提取音频数据
                    audio_float = chunk.audio_float_array
                    sample_rate = chunk.sample_rate

                    # 转换为 int16
                    audio_int16 = (audio_float * 32767).astype(np.int16)

                    # ✅ 生成标准 OutputAudioRawFrame（而不是直接播放）
                    audio_frame = OutputAudioRawFrame(
                        audio=audio_int16.tobytes(),
                        sample_rate=sample_rate,
                        num_channels=1
                    )

                    # ✅ 推送到 Pipeline（让 transport.output() 播放）
                    # 注意：这里需要使用同步方式推送（在线程池中）
                    # 使用 asyncio.run_coroutine_threadsafe 在主事件循环中推送
                    import asyncio
                    loop = asyncio.get_event_loop()
                    future = asyncio.run_coroutine_threadsafe(
                        self.push_frame(audio_frame, FrameDirection.DOWNSTREAM),
                        loop
                    )
                    # 等待推送完成
                    future.result(timeout=1.0)

                return False  # 正常完成，未中断

            except Exception as e:
                print(f"❌ TTS 生成失败: {e}")
                return False

        return False


# ==================== Vision Processor (符合 Pipecat 标准) ====================

class VisionProcessor(FrameProcessor):
    """
    Vision Processor - 符合 Pipecat 官方推荐模式

    架构改进：
    - ✅ 接收 LLMContext，直接修改 context（而不是推送新 Frame）
    - ✅ 在 user_aggregator 之后运行（context 已包含用户消息）
    - ✅ Vision 结果添加到 context，LLM 自动看到
    - ✅ 无需推送额外 Frame，符合官方架构
    """

    def __init__(self, api_url: str, api_key: str, context):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.context = context  # LLMContext 实例

        # Vision 关键词
        self.vision_keywords = [
            "看", "查看", "讲解", "描述", "显示什么", "显示的",
            "分析", "识别", "内容是", "画面", "截图", "图片",
            "界面", "当前", "屏幕"
        ]
        self.operation_keywords = [
            "点击", "输入", "打开", "关闭", "启动", "切换",
            "滚动", "搜索", "执行", "运行", "按"
        ]

    def _needs_vision(self, text: str) -> bool:
        """判断是否需要视觉理解"""
        # 操作关键词优先（React 模式）
        if any(kw in text for kw in self.operation_keywords):
            return False
        # 视觉关键词
        return any(kw in text for kw in self.vision_keywords)

    async def _capture_screenshot_async(self) -> tuple[bytes, tuple[int, int]]:
        """异步截图，返回 (图片字节, 尺寸)"""
        def capture():
            from PIL import ImageGrab
            import io

            # 尝试窗口截图
            try:
                import ctypes
                from ctypes import wintypes

                # 设置 DPI 感知
                try:
                    ctypes.windll.shcore.SetProcessDpiAwareness(2)
                except:
                    pass

                # 获取前台窗口
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                if hwnd:
                    rect = wintypes.RECT()
                    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))

                    # 修正边框
                    padding = 8
                    bbox = (
                        rect.left + padding,
                        rect.top,
                        rect.right - padding,
                        rect.bottom - padding
                    )
                    screenshot = ImageGrab.grab(bbox=bbox)
                else:
                    screenshot = ImageGrab.grab()
            except Exception:
                # 降级到全屏
                screenshot = ImageGrab.grab()

            # 转换为字节（内存操作）
            img_byte_arr = io.BytesIO()
            screenshot.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue(), screenshot.size

        return await asyncio.to_thread(capture)

    async def _call_vision_api(self, img_bytes: bytes, question: str) -> str:
        """调用 Qwen-VL-Max API（异步）"""
        import httpx
        import base64

        img_base64 = base64.b64encode(img_bytes).decode()

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.api_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "qwen-vl-max",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ]
                        }],
                        "max_tokens": 2000,
                        "temperature": 0.7
                    }
                )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"API错误 {response.status_code}: {response.text}"

        except Exception as e:
            return f"Vision API 调用失败: {str(e)}"

    async def process_frame(self, frame, direction):
        """处理帧"""
        await super().process_frame(frame, direction)

        # 响应中断
        if isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            return

        # ✅ 检查 context 中的最后一条用户消息（由 user_aggregator 添加）
        if self.context.messages:
            last_message = self.context.messages[-1]

            # 只处理用户消息
            if last_message.get("role") == "user":
                text_content = last_message.get("content", "")

                if self._needs_vision(text_content):
                    print(f"🔍 Vision 模式: {text_content}")

                    try:
                        # 异步截图
                        img_bytes, size = await self._capture_screenshot_async()

                        # 调用 Vision API
                        print(f"📸 调用 Vision API...")
                        result = await self._call_vision_api(img_bytes, text_content)

                        print(f"📊 Vision 结果: {result}")

                        # ✅ 直接修改 context（添加 Vision 观察结果）
                        # 方式1：作为 system 消息
                        self.context.messages.append({
                            "role": "system",
                            "content": f"[视觉观察] {result}"
                        })

                        # 方式2：修改用户消息（包含 Vision 结果）
                        # last_message["content"] = f"{text_content}\n\n[视觉观察] {result}"

                    except Exception as e:
                        print(f"❌ Vision 处理失败: {e}")
                        import traceback
                        traceback.print_exc()

        # ✅ 传递所有帧（不推送新 Frame）
        await self.push_frame(frame, direction)
