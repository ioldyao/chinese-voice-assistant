"""Silero VAD Processor - 集成 Pipecat VAD"""
import asyncio
import numpy as np
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    StartFrame,
    CancelFrame,
)


class SileroVADProcessor(FrameProcessor):
    """
    Silero VAD Processor - 检测语音开始/停止

    功能：
    - 接收 AudioRawFrame
    - 使用 Silero VAD 检测语音活动
    - 发送 UserStartedSpeakingFrame / UserStoppedSpeakingFrame
    """

    def __init__(self, vad_analyzer, sample_rate: int = 16000):
        super().__init__()
        self.vad_analyzer = vad_analyzer
        self.sample_rate = sample_rate
        self._is_speaking = False
        self._started = False

    async def process_frame(self, frame, direction):
        """处理音频帧，进行 VAD 检测"""
        await super().process_frame(frame, direction)

        # ✅ 启动 VAD analyzer（在收到 StartFrame 时）
        if isinstance(frame, StartFrame) and not self._started:
            await self.vad_analyzer.start(self.sample_rate)
            self._started = True
            print("✓ Silero VAD 已启动")
            await self.push_frame(frame, direction)
            return

        # ✅ 停止 VAD analyzer
        if isinstance(frame, CancelFrame):
            if self._started:
                await self.vad_analyzer.stop()
                self._started = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, AudioRawFrame) and self._started:
            # ✅ 使用 await 调用异步方法
            vad_frames = await self.vad_analyzer.analyze_audio(frame.audio)

            # 推送 VAD 生成的帧
            for vad_frame in vad_frames:
                await self.push_frame(vad_frame, direction)

        # 继续传递原始帧
        await self.push_frame(frame, direction)

