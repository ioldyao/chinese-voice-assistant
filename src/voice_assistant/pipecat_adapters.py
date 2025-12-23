"""Pipecat 适配器 - 封装现有组件为 Pipecat Processors"""
import asyncio
import numpy as np
from typing import Optional
from dataclasses import dataclass

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TextFrame,
    TTSAudioRawFrame,
    StartInterruptionFrame,
    EndFrame,
)


# ==================== 自定义 Frame 类型 ====================

@dataclass
class WakeWordDetectedFrame(Frame):
    """唤醒词检测帧"""
    keyword: str
    confidence: float = 1.0


@dataclass
class ReActStepFrame(Frame):
    """React 推理步骤帧"""
    thought: str
    action: str
    action_input: dict
    observation: str
    success: bool


# ==================== Sherpa-ONNX KWS Processor ====================

class SherpaKWSProcessor(FrameProcessor):
    """
    Sherpa-ONNX KWS 适配器

    将现有的 Sherpa-ONNX KWS 模型封装为 Pipecat Processor
    处理音频帧，检测唤醒词，输出 WakeWordDetectedFrame
    """

    def __init__(self, kws_model):
        super().__init__()
        self.kws_model = kws_model
        self.kws_stream = kws_model.create_stream()
        self.sample_rate = 16000
        self.is_awake = False  # 用于并行 Pipeline 的条件判断

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

                # 发出唤醒事件
                await self.push_frame(
                    WakeWordDetectedFrame(keyword=result),
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

        # 检测唤醒词，开始录音
        if isinstance(frame, WakeWordDetectedFrame):
            print("📝 开始录音识别...")
            self.recording = True
            self.buffer = []
            self.silence_count = 0
            self.has_speech = False
            self.frame_count = 0

            # 传递唤醒帧
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
                    await self.push_frame(
                        TextFrame(text=text),
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


# ==================== React Agent Processor ====================

class ReactAgentProcessor(FrameProcessor):
    """
    React Agent Processor - 基于 MCP 官方推荐模式

    使用完全异步的实现：
    - 直接调用 agent.execute_command_async()
    - 在主事件循环中执行 MCP 调用
    - 后台任务执行，不阻塞 Pipeline
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

        # 检查唤醒词，取消当前任务
        if isinstance(frame, WakeWordDetectedFrame):
            if self.current_task and not self.current_task.done():
                print("⏸️  检测到新唤醒词，取消当前 Agent 任务")
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
    Piper TTS 适配器

    将现有的 TTSManagerStreaming 封装为 Pipecat Processor
    接收文本帧，生成音频帧并直接播放，支持中断
    """

    def __init__(self, tts_manager, transport=None):
        super().__init__()
        self.tts = tts_manager
        self.transport = transport  # 用于直接播放音频
        self.interrupt_flag = False  # 中断标志

    def interrupt(self):
        """中断当前TTS播放"""
        self.interrupt_flag = True

    async def process_frame(self, frame, direction):
        """处理文本帧，生成 TTS 音频"""
        await super().process_frame(frame, direction)

        # 检测唤醒词，设置中断
        if isinstance(frame, WakeWordDetectedFrame):
            print("⏸️  检测到新唤醒词，中断TTS播放")
            self.interrupt()
            # 传递唤醒帧
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextFrame):
            print(f"🔊 TTS 合成: {frame.text}")

            # 重置中断标志
            self.interrupt_flag = False

            # 在线程池中生成和播放音频
            await asyncio.to_thread(
                self._synthesize_and_play_sync,
                frame.text
            )

            # 传递原始文本帧
            await self.push_frame(frame, direction)
        else:
            # 其他帧直接传递
            await self.push_frame(frame, direction)

    def _synthesize_and_play_sync(self, text):
        """同步 TTS 合成并播放（在线程池中执行），支持中断"""
        if self.tts.engine_type == "piper":
            try:
                # 使用 Piper TTS 生成音频
                audio_generator = self.tts.piper_voice.synthesize(text)

                for chunk in audio_generator:
                    # 检查中断标志
                    if self.interrupt_flag:
                        print("⏸️  TTS 播放已中断")
                        break

                    # 提取音频数据
                    audio_float = chunk.audio_float_array
                    sample_rate = chunk.sample_rate

                    # 转换为 int16
                    audio_int16 = (audio_float * 32767).astype(np.int16)

                    # 直接播放到 transport
                    if self.transport and self.transport.output_stream:
                        self.transport.output_stream.write(audio_int16.tobytes())

            except Exception as e:
                print(f"❌ TTS 生成失败: {e}")
