"""PyAudio Transport - 标准 Pipecat Transport 实现（符合官方架构 v2）"""
import asyncio
import numpy as np
import pyaudio
from typing import Optional

from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    CancelFrame,
)


class PyAudioInputTransport(BaseInputTransport):
    """
    PyAudio 音频输入处理器（符合 Pipecat 官方架构）

    继承 BaseInputTransport，自动获得：
    - VAD 支持（通过 TransportParams.vad_analyzer）
    - Turn Detection 支持
    - 标准音频流处理
    """

    def __init__(self, transport: "PyAudioTransport", params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self.transport = transport
        self._running = False
        self._input_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """处理帧"""
        await super().process_frame(frame, direction)

        # ✅ 关键修复：在 StartFrame 后调用 set_transport_ready()
        # 这会创建 _audio_task，启动 VAD 处理循环
        if isinstance(frame, StartFrame):
            # BaseInputTransport 已经处理了 StartFrame（推送帧 + 调用 start()）
            # 现在我们需要启动音频输入循环
            self._running = True

            # ✅ 调用父类方法创建 _audio_task（VAD 处理循环）
            await self.set_transport_ready(frame)

            # 启动我们的音频读取循环
            self._input_task = asyncio.create_task(self._audio_input_loop())

        # 停止音频输入
        elif isinstance(frame, CancelFrame):
            self._running = False
            if self._input_task:
                self._input_task.cancel()
                try:
                    await self._input_task
                except asyncio.CancelledError:
                    pass
            # BaseInputTransport 已经处理了 CancelFrame

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

                # ✅ 创建标准 InputAudioRawFrame
                frame = InputAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self.transport.sample_rate,
                    num_channels=self.transport.channels
                )

                # ✅ 关键修复：使用 BaseInputTransport 的 push_audio_frame
                # 这会把音频放入队列，_audio_task_handler 会处理 VAD
                await self.push_audio_frame(frame)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"❌ 音频输入错误: {e}")



class PyAudioOutputTransport(BaseOutputTransport):
    """
    PyAudio 音频输出处理器（符合 Pipecat 官方架构）

    继承 BaseOutputTransport：
    - 接收 OutputAudioRawFrame
    - 播放到扬声器
    - 不生成新 Frame（Pipeline 终点）
    """

    def __init__(self, transport: "PyAudioTransport", params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self.transport = transport
        self._first_audio = True  # 首次播放标志

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        """
        写入音频帧到输出设备（被 MediaSender 调用）

        这是 BaseOutputTransport 期望的接口方法
        """
        try:
            # 首次播放时提示
            if self._first_audio:
                print(f"🔊 开始播放 TTS 音频 ({frame.sample_rate}Hz)")
                self._first_audio = False

            # 异步播放（不阻塞 Pipeline）
            await asyncio.to_thread(
                self.transport.output_stream.write,
                frame.audio
            )
        except Exception as e:
            print(f"❌ 音频播放错误: {e}")
            # 重置标志，以便下次播放时重新提示
            self._first_audio = True



class PyAudioTransport(BaseTransport):
    """
    PyAudio Transport - 符合 Pipecat 官方架构 v2

    改进：
    - ✅ 接受 TransportParams（支持 VAD、Turn Detection）
    - ✅ Input/Output 继承 BaseInputTransport/BaseOutputTransport
    - ✅ 自动处理 VAD（通过 BaseInputTransport）

    用法：
        from pipecat.transports.base_transport import TransportParams
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams

        transport = PyAudioTransport(
            sample_rate=16000,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(stop_secs=0.8)
                )
            )
        )

        pipeline = Pipeline([
            transport.input(),    # 自动处理 VAD
            # ... 处理器 ...
            transport.output(),   # 音频输出
        ])
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,
        params: TransportParams = None,
    ):
        # ✅ BaseTransport 只接受 name 参数
        super().__init__(name="PyAudioTransport")

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

        # ✅ 保存 params（或使用默认值）
        self._params = params or TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )

        # PyAudio 实例
        self.p = pyaudio.PyAudio()
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None

        # ✅ 创建 input/output processors 并传递 params
        self._input_processor: Optional[PyAudioInputTransport] = None
        self._output_processor: Optional[PyAudioOutputTransport] = None

    def input(self) -> FrameProcessor:
        """返回音频输入处理器（符合 Pipecat 标准）"""
        if not self._input_processor:
            # ✅ 传递 params 给 BaseInputTransport
            self._input_processor = PyAudioInputTransport(self, self._params)
        return self._input_processor

    def output(self) -> FrameProcessor:
        """返回音频输出处理器（符合 Pipecat 标准）"""
        if not self._output_processor:
            # ✅ 传递 params 给 BaseOutputTransport
            self._output_processor = PyAudioOutputTransport(self, self._params)
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

        # ✅ 如果配置了 VAD，显示信息
        if self._params.vad_analyzer:
            print(f"✓ VAD 已启用: {type(self._params.vad_analyzer).__name__}")

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

