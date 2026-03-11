"""
Sherpa-ONNX STT Service - 符合 Pipecat 官方接口

将 Sherpa-ONNX ASR 包装成 Pipecat STTService，配合 WakeCheckFilter 使用。

架构：
transport.input() → SherpaSTTService → WakeCheckFilter → LLM → ...

优点：
- ✅ 符合 Pipecat 官方架构
- ✅ 使用 WakeCheckFilter 处理唤醒词
- ✅ 自动处理 VAD 事件
- ✅ 生成标准的 TranscriptionFrame
"""
import asyncio
import numpy as np
from typing import AsyncGenerator, Optional

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language


class SherpaSTTService(SegmentedSTTService):
    """
    Sherpa-ONNX STT Service（符合 Pipecat 官方接口）

    使用 SegmentedSTTService 作为基类，自动处理 VAD 事件和音频缓冲。

    功能：
    - 响应 UserStartedSpeakingFrame 开始缓冲音频
    - 响应 UserStoppedSpeakingFrame 停止并识别
    - 生成标准的 TranscriptionFrame

    使用示例：
    ```python
    from pipecat.processors.filters import WakeCheckFilter

    stt = SherpaSTTService(asr_model=asr_model)

    wake_filter = WakeCheckFilter(
        wake_phrases=["小智", "你好助手", "智能助手"],
        keepalive_timeout=5.0
    )

    pipeline = Pipeline([
        transport.input(),
        stt,          # ← SherpaSTTService
        wake_filter,  # ← 检查唤醒词
        llm,
        tts,
        transport.output()
    ])
    ```

    注意：
    - 需要 VAD 启用（在 TransportParams 中配置）
    - 识别结果会作为 TranscriptionFrame 传递
    - WakeCheckFilter 检查文本是否包含唤醒词
    """

    def __init__(self, asr_model, *, sample_rate: int = 16000, **kwargs):
        """
        初始化 SherpaSTTService

        Args:
            asr_model: Sherpa-ONNX ASR 模型实例
            sample_rate: 音频采样率（默认 16000）
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.asr_model = asr_model

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        运行语音识别

        此方法由 SegmentedSTTService 自动调用：
        - 当 UserStoppedSpeakingFrame 时
        - 传入累积的音频数据

        Args:
            audio: 累积的音频数据（WAV 格式字节）

        Yields:
            TranscriptionFrame: 识别结果
        """
        try:
            # 转换音频数据
            # audio 是 WAV 格式（由 SegmentedSTTService 自动封装）
            # 需要提取 PCM 数据

            import wave
            import io

            # 解析 WAV 文件
            wav_file = io.BytesIO(audio)
            with wave.open(wav_file, 'rb') as wav:
                # 读取音频数据
                frames = wav.getnframes()
                audio_data = wav.readframes(frames)

            # 转换为 numpy 数组（int16 → float32）
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # 调用 Sherpa-ONNX ASR 识别
            text = await self._recognize_async(audio_array)

            if text:
                # 生成 TranscriptionFrame
                yield TranscriptionFrame(
                    text=text,
                    user_id="",  # 本地音频，无用户 ID
                    timestamp=self._get_timestamp(),
                    language=Language.ZH,
                )

        except Exception as e:
            # 识别错误，记录但不中断
            from loguru import logger
            logger.error(f"SherpaSTTService 识别错误: {e}")

    async def _recognize_async(self, audio_array: np.ndarray) -> str:
        """
        异步语音识别

        Args:
            audio_array: 音频数组（float32，范围 -1.0 到 1.0）

        Returns:
            识别的文本
        """
        # Sherpa-ONNX 是同步的，需要在线程池中运行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._sync_recognize,
            audio_array
        )

    def _sync_recognize(self, audio_array: np.ndarray) -> str:
        """
        同步语音识别（在线程池中运行）

        Args:
            audio_array: 音频数组（float32）

        Returns:
            识别的文本
        """
        # 创建识别流
        stream = self.asr_model.create_stream()

        # 喂入音频数据
        stream.accept_waveform(16000, audio_array)

        # 解码并获取结果
        self.asr_model.decode_stream(stream)
        result = stream.result.text.strip()

        return result if result else ""

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        import time
        return f"{time.time():.3f}"
