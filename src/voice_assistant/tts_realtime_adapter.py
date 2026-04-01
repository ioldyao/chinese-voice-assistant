"""
Qwen-TTS-Realtime WebSocket Pipecat 适配器

将 Qwen-TTS-Realtime WebSocket 实现实例适配为 Pipecat 标准 Processor。

特点：
- 符合 Pipecat Pipeline 标准
- 支持句子级流式播放
- 支持中断机制
- 首包延迟优化（~100-200ms）

使用示例：
```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor

tts_processor = QwenRealtimeTTSProcessor()

pipeline = Pipeline([
    llm,
    tts_processor,
    transport.output()
])
```
"""

import asyncio
import numpy as np
from typing import Optional
from pathlib import Path

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    OutputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
)

# 延迟导入，避免模块顶层导入错误
# from .tts_realtime import QwenRealtimeTTS, RealtimeTTSConfig


class QwenRealtimeTTSProcessor(FrameProcessor):
    """
    Qwen-TTS-Realtime WebSocket TTS 适配器

    符合 Pipecat 标准，支持流式播放和中断。

    特点：
    - WebSocket 持久连接，低延迟
    - 增量音频流式返回
    - 支持句子级缓冲和中断
    - 首包延迟优化（~100-200ms）

    使用示例：
    ```python
    tts = QwenRealtimeTTSProcessor(
        model="qwen3-tts-flash-realtime",
        voice="Cherry",
        mode="server_commit"
    )

    pipeline = Pipeline([
        llm,
        tts,
        transport.output()
    ])
    ```
    """

    def __init__(
        self,
        model: str = "qwen3-tts-flash-realtime",
        voice: str = "Cherry",
        mode: str = "server_commit",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 TTS Processor

        Args:
            model: 模型名称（qwen3-tts-flash-realtime | qwen3-tts-instruct-flash-realtime）
            voice: 音色（Cherry | Ethan | Sunny | Dylan）
            mode: 交互模式（server_commit | commit）
            api_key: DashScope API Key
            **kwargs: 其他配置参数
        """
        super().__init__()

        # 保存配置参数，延迟创建 RealtimeTTSConfig
        self._model = model
        self._voice = voice
        self._mode = mode
        self._api_key = api_key
        self._kwargs = kwargs
        self.config = None

        # TTS 实例
        self.tts: Optional[QwenRealtimeTTS] = None
        self._initialized = False

        # 播放状态
        self.interrupt_flag = False
        self.is_speaking = False

        # 句子缓冲区（按句子流式播放）
        self.sentence_buffer = ""
        self.sentence_delimiters = ["。", "！", "？", ".", "!", "?", "\n", "，", ",", " "]

        # 事件循环引用（用于线程池中推送帧）
        self._loop = None

        # LLM 输出显示
        self._llm_started = False

    async def _ensure_initialized(self):
        """确保 TTS 已初始化"""
        if not self._initialized:
            try:
                # 延迟导入，避免模块顶层导入错误
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
            except Exception as e:
                print(f"❌ WebSocket TTS 初始化失败: {e}")
                print("💡 提示：请确保安装了 dashscope>=1.25.11")
                print("   运行: uv add dashscope --upgrade")
                print("   或在 .env 中设置: TTS_SERVICE=piper")
                # 重新抛出异常，让上层处理
                raise

    def interrupt(self):
        """中断当前 TTS 播放"""
        self.interrupt_flag = True
        if self.tts:
            self.tts.stop()
        self.is_speaking = False

    async def process_frame(self, frame, direction):
        """处理帧"""
        await super().process_frame(frame, direction)

        # 保存事件循环引用
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        # 实时显示 LLM 输出
        if isinstance(frame, TextFrame):
            if not self._llm_started:
                print("\n🤖 LLM: ", end="", flush=True)
                self._llm_started = True
            print(frame.text, end="", flush=True)

        # 确保已初始化
        if isinstance(frame, TextFrame):
            await self._ensure_initialized()

        # 响应中断信号
        if isinstance(frame, InterruptionFrame):
            if self.is_speaking:
                print("⏸️  检测到中断信号，停止 TTS 播放")
                self.interrupt()

            # 清空缓冲区
            self.sentence_buffer = ""
            self._llm_started = False

            # 传递中断帧
            await self.push_frame(frame, direction)
            return

        # 检测 LLM 响应结束
        if isinstance(frame, LLMFullResponseEndFrame):
            # 结束时换行
            if self._llm_started:
                print("\n")
                self._llm_started = False

            # 播放剩余缓冲
            if self.sentence_buffer.strip():
                await self._synthesize_and_push(self.sentence_buffer)
                self.sentence_buffer = ""

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
                    await self._synthesize_and_push(sentence.strip())
                    break

            # 传递原始文本帧
            await self.push_frame(frame, direction)
        else:
            # 其他帧直接传递
            await self.push_frame(frame, direction)

    async def _synthesize_and_push(self, text: str):
        """
        异步合成并推送音频帧

        Args:
            text: 要合成的文本
        """
        if not text or not self.tts:
            return

        self.interrupt_flag = False
        self.is_speaking = True

        try:
            # 调用 WebSocket TTS
            await self.tts.speak(text)

            # 注意：WebSocket TTS 会自动播放音频
            # 所以这里不需要手动推送音频帧
            # 如果需要集成到 Pipeline，可以修改为：

            # # 创建音频生成器
            # async for audio_chunk in self.tts.synthesize_stream(text):
            #     if self.interrupt_flag:
            #         break
            #
            #     # 重采样到 16kHz
            #     audio_16k = self._resample_audio(audio_chunk)
            #
            #     # 生成标准音频帧
            #     audio_frame = OutputAudioRawFrame(
            #         audio=audio_16k.tobytes(),
            #         sample_rate=16000,
            #         num_channels=1
            #     )
            #
            #     # 推送到 Pipeline
            #     await self.push_frame(audio_frame, FrameDirection.DOWNSTREAM)

        except Exception as e:
            print(f"❌ TTS 合成失败: {e}")
        finally:
            self.is_speaking = False

    def _resample_audio(self, audio_data: np.ndarray, from_rate: int = 24000, to_rate: int = 16000) -> np.ndarray:
        """
        重采样音频数据

        Args:
            audio_data: 原始音频数据
            from_rate: 原始采样率
            to_rate: 目标采样率

        Returns:
            重采样后的音频数据
        """
        if from_rate == to_rate:
            return audio_data

        # 使用 numpy 插值重采样
        samples_ratio = len(audio_data) * to_rate // from_rate
        resampled = np.interp(
            np.linspace(0, len(audio_data) - 1, samples_ratio),
            np.arange(len(audio_data)),
            audio_data.astype(np.float32)
        ).astype(np.int16)

        return resampled

    async def cleanup(self):
        """清理资源"""
        if self.tts:
            await self.tts.close()
            self._initialized = False


# ==================== 工厂函数 ====================

def create_realtime_tts_processor(
    model: str = "qwen3-tts-flash-realtime",
    voice: str = "Cherry",
    mode: str = "server_commit",
    api_key: Optional[str] = None,
    **kwargs
):
    """
    创建 Qwen-TTS-Realtime Processor（便捷函数）

    Args:
        model: 模型名称
        voice: 音色
        mode: 交互模式
        api_key: API 密钥
        **kwargs: 其他配置参数

    Returns:
        QwenRealtimeTTSProcessor 实例

    使用示例：
    ```python
    tts = create_realtime_tts_processor(
        model="qwen3-tts-flash-realtime",
        voice="Cherry"
    )

    pipeline = Pipeline([
        llm,
        tts,
        transport.output()
    ])
    ```
    """
    return QwenRealtimeTTSProcessor(
        model=model,
        voice=voice,
        mode=mode,
        api_key=api_key,
        **kwargs
    )
