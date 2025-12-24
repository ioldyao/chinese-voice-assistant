"""PyAudio Transport - 标准 Pipecat Transport 实现"""
import asyncio
import numpy as np
import pyaudio
from typing import Optional

from pipecat.transports.base_transport import BaseTransport
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    CancelFrame,
)


class PyAudioInputProcessor(FrameProcessor):
    """
    PyAudio 音频输入处理器

    符合 Pipecat 标准：
    - 从麦克风读取音频
    - 生成 AudioRawFrame
    - 推送到 Pipeline
    """

    def __init__(self, transport: "PyAudioTransport"):
        super().__init__()
        self.transport = transport
        self._running = False
        self._input_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """处理帧"""
        await super().process_frame(frame, direction)

        # 启动音频输入
        if isinstance(frame, StartFrame):
            self._running = True
            self._input_task = asyncio.create_task(self._audio_input_loop())
            await self.push_frame(frame, direction)

        # 停止音频输入
        elif isinstance(frame, CancelFrame):
            self._running = False
            if self._input_task:
                self._input_task.cancel()
                try:
                    await self._input_task
                except asyncio.CancelledError:
                    pass
            await self.push_frame(frame, direction)

        else:
            # 其他帧直接传递
            await self.push_frame(frame, direction)

    async def _audio_input_loop(self):
        """持续读取音频并推送到 Pipeline"""
        try:
            while self._running:
                # 从麦克风读取音频
                audio_bytes = await asyncio.to_thread(
                    self.transport.input_stream.read,
                    self.transport.chunk_size,
                    exception_on_overflow=False
                )

                # 创建标准 AudioRawFrame
                frame = AudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self.transport.sample_rate,
                    num_channels=self.transport.channels
                )

                # 推送到 Pipeline（DOWNSTREAM）
                await self.push_frame(frame, FrameDirection.DOWNSTREAM)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"❌ 音频输入错误: {e}")


class PyAudioOutputProcessor(FrameProcessor):
    """
    PyAudio 音频输出处理器

    符合 Pipecat 标准：
    - 接收 OutputAudioRawFrame
    - 播放到扬声器
    - 不生成新 Frame（Pipeline 终点）
    """

    def __init__(self, transport: "PyAudioTransport"):
        super().__init__()
        self.transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """处理帧"""
        await super().process_frame(frame, direction)

        # 播放音频帧
        if isinstance(frame, OutputAudioRawFrame):
            try:
                # 异步播放（不阻塞 Pipeline）
                await asyncio.to_thread(
                    self.transport.output_stream.write,
                    frame.audio
                )
            except Exception as e:
                print(f"❌ 音频播放错误: {e}")

        # 传递所有帧（用于后续处理器，如 context_aggregator）
        await self.push_frame(frame, direction)


class PyAudioTransport(BaseTransport):
    """
    PyAudio Transport - 符合 Pipecat 标准接口

    用法：
        transport = PyAudioTransport(sample_rate=16000)

        pipeline = Pipeline([
            transport.input(),    # 音频输入
            # ... 处理器 ...
            transport.output(),   # 音频输出
        ])
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,
    ):
        # ✅ BaseTransport 只接受 name, input_name, output_name 参数
        super().__init__()

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

        # PyAudio 实例
        self.p = pyaudio.PyAudio()
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None

        # 处理器（延迟创建）
        self._input_processor: Optional[PyAudioInputProcessor] = None
        self._output_processor: Optional[PyAudioOutputProcessor] = None

    def input(self) -> FrameProcessor:
        """返回音频输入处理器（符合 Pipecat 标准）"""
        if not self._input_processor:
            self._input_processor = PyAudioInputProcessor(self)
        return self._input_processor

    def output(self) -> FrameProcessor:
        """返回音频输出处理器（符合 Pipecat 标准）"""
        if not self._output_processor:
            self._output_processor = PyAudioOutputProcessor(self)
        return self._output_processor

    async def start(self):
        """启动音频流"""
        # 打开输入流
        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        # 打开输出流
        self.output_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )

        print(f"✓ PyAudio Transport 已启动 ({self.sample_rate}Hz, {self.channels}ch)")

    async def stop(self):
        """停止音频流"""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()

        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()

        self.p.terminate()
        print("✓ PyAudio Transport 已停止")
