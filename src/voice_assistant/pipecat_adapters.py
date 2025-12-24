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

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TextFrame,
    TTSAudioRawFrame,
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
    Sherpa-ONNX KWS 适配器

    将现有的 Sherpa-ONNX KWS 模型封装为 Pipecat Processor
    处理音频帧，检测唤醒词，输出官方 InterruptionFrame
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

                # ✅ 发送用户开始说话的信号
                await self.push_frame(
                    UserStartedSpeakingFrame(),
                    direction
                )

                # ✅ 发出官方中断帧（触发 ASR 开始录音）
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
    Sherpa-ONNX ASR 适配器

    将现有的 Sherpa-ONNX ASR 模型封装为 Pipecat Processor
    检测到唤醒词后开始录音，使用静音检测自动停止，输出识别文本
    """

    def __init__(self, asr_model, sample_rate=16000):
        super().__init__()
        self.asr_model = asr_model
        self.sample_rate = sample_rate

        # 录音状态
        self.recording = False
        self.buffer = []

        # 静音检测参数（与原有逻辑一致）
        self.silence_threshold = 0.02
        self.max_silence_frames = 20  # 约 1.3 秒
        self.min_record_frames = 15   # 最小录音保护

        self.silence_count = 0
        self.has_speech = False
        self.frame_count = 0

    async def process_frame(self, frame, direction):
        """处理音频帧，识别语音"""
        await super().process_frame(frame, direction)

        # ✅ 检测官方中断帧，开始录音
        if isinstance(frame, InterruptionFrame):
            print("📝 开始录音识别...")
            self.recording = True
            self.buffer = []
            self.silence_count = 0
            self.has_speech = False
            self.frame_count = 0

            # 传递中断帧
            await self.push_frame(frame, direction)
            return

        # 录音过程
        if self.recording and isinstance(frame, AudioRawFrame):
            # 提取音频数据
            audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            self.buffer.append(audio_data)
            self.frame_count += 1

            # 计算音量
            volume = np.sqrt(np.mean(audio_data**2))

            # 静音检测
            if volume >= self.silence_threshold:
                self.has_speech = True
                self.silence_count = 0
            else:
                self.silence_count += 1

            # 停止条件：有语音 + 连续静音 + 超过最小保护帧
            if (self.has_speech and
                self.silence_count > self.max_silence_frames and
                self.frame_count > self.min_record_frames):

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
                    # 发送 UserStoppedSpeakingFrame 触发 LLM 处理
                    await self.push_frame(
                        UserStoppedSpeakingFrame(),
                        direction
                    )

                # 重置状态
                self.recording = False
                self.buffer = []

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
    Piper TTS 适配器 - 支持文本聚合

    将现有的 TTSManagerStreaming 封装为 Pipecat Processor
    接收文本帧，生成音频帧并直接播放
    响应官方 InterruptionFrame，发出 TTSStoppedFrame
    聚合流式文本，完整播放
    """

    def __init__(self, tts_manager, transport=None):
        super().__init__()
        self.tts = tts_manager
        self.transport = transport  # 用于直接播放音频
        self.interrupt_flag = False  # 中断标志
        self.is_speaking = False  # TTS 播放状态

        # ✅ 文本聚合缓冲区
        self.text_buffer = []
        self.is_buffering = False

    def interrupt(self):
        """中断当前TTS播放"""
        self.interrupt_flag = True

    async def process_frame(self, frame, direction):
        """处理文本帧，生成 TTS 音频"""
        await super().process_frame(frame, direction)

        # ✅ 响应官方中断帧，设置中断
        if isinstance(frame, InterruptionFrame):
            if self.is_speaking:
                print("⏸️  检测到中断信号，停止TTS播放")
                self.interrupt()
            # 清空缓冲区
            self.text_buffer = []
            self.is_buffering = False
            # 传递中断帧
            await self.push_frame(frame, direction)
            return

        # ✅ 检测 LLM 响应开始（开始缓冲）
        from pipecat.frames.frames import LLMFullResponseStartFrame
        if isinstance(frame, LLMFullResponseStartFrame):
            self.text_buffer = []
            self.is_buffering = True
            await self.push_frame(frame, direction)
            return

        # ✅ 检测 LLM 响应结束（播放缓冲内容）
        from pipecat.frames.frames import LLMFullResponseEndFrame
        if isinstance(frame, LLMFullResponseEndFrame):
            self.is_buffering = False
            if self.text_buffer:
                # 聚合所有文本
                full_text = "".join(self.text_buffer)
                print(f"🔊 TTS 合成（完整）: {full_text[:50]}...")

                # 播放完整文本
                self.interrupt_flag = False
                self.is_speaking = True

                was_interrupted = await asyncio.to_thread(
                    self._synthesize_and_play_sync,
                    full_text
                )

                self.is_speaking = False
                if was_interrupted:
                    print("⏸️  TTS 已被中断")
                    await self.push_frame(TTSStoppedFrame(), direction)
                else:
                    print("✓ TTS 播放完成")
                    # ✅ 发送 turn 结束信号
                    await self.push_frame(UserStoppedSpeakingFrame(), direction)

                self.text_buffer = []

            await self.push_frame(frame, direction)
            return

        # ✅ 缓冲文本帧（流式输出）
        if isinstance(frame, TextFrame):
            if self.is_buffering:
                # 缓冲模式：收集文本，不播放
                self.text_buffer.append(frame.text)
            else:
                # 非缓冲模式（兼容非流式输出）：直接播放
                print(f"🔊 TTS 合成: {frame.text}")
                self.interrupt_flag = False
                self.is_speaking = True

                was_interrupted = await asyncio.to_thread(
                    self._synthesize_and_play_sync,
                    frame.text
                )

                self.is_speaking = False
                if was_interrupted:
                    print("⏸️  TTS 已被中断")
                    await self.push_frame(TTSStoppedFrame(), direction)
                else:
                    print("✓ TTS 播放完成")
                    await self.push_frame(UserStoppedSpeakingFrame(), direction)

            # 传递原始文本帧
            await self.push_frame(frame, direction)
        else:
            # 其他帧直接传递
            await self.push_frame(frame, direction)

    def _synthesize_and_play_sync(self, text) -> bool:
        """
        同步 TTS 合成并播放（在线程池中执行），支持中断

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
                        print("⏸️  TTS 播放已中断")
                        return True  # 返回中断标志

                    # 提取音频数据
                    audio_float = chunk.audio_float_array
                    sample_rate = chunk.sample_rate

                    # 转换为 int16
                    audio_int16 = (audio_float * 32767).astype(np.int16)

                    # 直接播放到 transport
                    if self.transport and self.transport.output_stream:
                        self.transport.output_stream.write(audio_int16.tobytes())

                return False  # 正常完成，未中断

            except Exception as e:
                print(f"❌ TTS 生成失败: {e}")
                return False


# ==================== Vision Processors (Pipecat 官方模式) ====================

class ScreenshotProcessor(FrameProcessor):
    """
    截图 Processor - 判断并生成 UserImageRawFrame

    采用 Pipecat 官方推荐模式：
    - 判断是否需要视觉理解
    - 截图并生成 UserImageRawFrame（内存传递，无文件 I/O）
    - 传递给 QwenVisionProcessor 处理
    """

    def __init__(self):
        super().__init__()
        self.vision_keywords = [
            "看", "查看", "讲解", "描述", "显示什么", "显示的",
            "分析", "识别", "内容是", "画面", "截图", "图片"
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
        """
        异步截图，返回 (图片字节, 尺寸)

        使用 Pipecat 官方模式：内存传递，无文件 I/O
        """
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

    async def process_frame(self, frame, direction):
        """处理帧"""
        await super().process_frame(frame, direction)

        # 响应中断
        if isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            return

        # ✅ 处理文本命令（支持 TextFrame 和 TranscriptionFrame）
        text_content = None
        if isinstance(frame, TextFrame):
            text_content = frame.text
        elif isinstance(frame, TranscriptionFrame):
            text_content = frame.text

        if text_content:
            if self._needs_vision(text_content):
                print(f"🔍 Vision 模式: {text_content}")

                try:
                    # 异步截图（内存操作）
                    img_bytes, size = await self._capture_screenshot_async()

                    # 创建 UserImageRawFrame（Pipecat 官方 Frame）
                    from pipecat.frames.frames import UserImageRawFrame

                    vision_frame = UserImageRawFrame(
                        image=img_bytes,
                        size=size,
                        format="PNG"
                    )

                    # 附加用户问题（用于 Vision API）
                    vision_frame.user_question = text_content

                    # 推送到 QwenVisionProcessor
                    await self.push_frame(vision_frame, direction)
                    return

                except Exception as e:
                    print(f"❌ 截图失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 失败时传递原始帧给 Agent
                    await self.push_frame(frame, direction)
                    return

        # 非 Vision 任务，传递给下游
        await self.push_frame(frame, direction)


class QwenVisionProcessor(FrameProcessor):
    """
    Qwen Vision API Processor - 处理 UserImageRawFrame

    采用 Pipecat 官方推荐模式：
    - 接收 UserImageRawFrame（官方 Frame 类型）
    - 调用 Qwen-VL-Max API（完全异步）
    - 返回 TextFrame（结果）
    """

    def __init__(self, api_url: str, api_key: str):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key

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
                return f"API错误 {response.status_code}: {response.text[:100]}"

        except Exception as e:
            return f"Vision API 调用失败: {str(e)}"

    async def process_frame(self, frame, direction):
        """处理帧"""
        await super().process_frame(frame, direction)

        # 响应中断
        if isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            return

        # 处理 UserImageRawFrame（Pipecat 官方 Frame）
        from pipecat.frames.frames import UserImageRawFrame

        if isinstance(frame, UserImageRawFrame):
            # 提取问题
            question = getattr(frame, 'user_question', "屏幕上有什么内容？")

            print(f"📸 调用 Vision API...")

            # 异步调用 API
            result = await self._call_vision_api(frame.image, question)

            # 输出结果
            print(f"📊 Vision 结果: {result[:100]}...")

            # 推送结果（TextFrame）
            await self.push_frame(TextFrame(result), direction)
            return

        # 其他帧类型，直接传递
        await self.push_frame(frame, direction)
