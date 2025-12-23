"""Pipecat 主程序 - 基于 Pipeline 架构的语音助手"""
import asyncio
import signal
import sys
from pathlib import Path

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.base_transport import TransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

# 导入适配器
from .pipecat_adapters import (
    SherpaKWSProcessor,
    SherpaASRProcessor,
    ReactAgentProcessor,
    PiperTTSProcessor,
)

# 导入现有组件
from .wake_word import SmartWakeWordSystem
from .config import MODELS_DIR


class SimplePyAudioTransport:
    """
    简化的 PyAudio Transport

    在 Phase 1 中，我们使用简化的音频传输实现
    直接使用 PyAudio 进行音频 I/O
    """

    def __init__(self, sample_rate=16000, channels=1):
        import pyaudio
        import numpy as np

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 512

        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

        self.running = False
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()

    async def start(self):
        """启动音频传输"""
        self.running = True

        # 启动输入流
        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=None
        )

        # 启动输出流
        self.output_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )

        print("✓ 音频传输已启动")

    async def stop(self):
        """停止音频传输"""
        self.running = False

        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()

        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()

        self.p.terminate()
        print("✓ 音频传输已停止")

    async def read_audio_frames(self):
        """读取音频帧（生成器）"""
        from pipecat.frames.frames import AudioRawFrame

        while self.running:
            try:
                # 从麦克风读取音频
                audio_bytes = await asyncio.to_thread(
                    self.input_stream.read,
                    self.chunk_size,
                    exception_on_overflow=False
                )

                # 创建音频帧
                frame = AudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self.sample_rate,
                    num_channels=self.channels
                )

                yield frame

            except Exception as e:
                print(f"❌ 音频读取错误: {e}")
                break

    async def write_audio_frame(self, frame):
        """写入音频帧到扬声器"""
        from pipecat.frames.frames import TTSAudioRawFrame

        if isinstance(frame, TTSAudioRawFrame) and self.output_stream:
            try:
                await asyncio.to_thread(
                    self.output_stream.write,
                    frame.audio
                )
            except Exception as e:
                print(f"❌ 音频播放错误: {e}")


async def create_pipecat_pipeline():
    """
    创建 Pipecat Pipeline

    Phase 1: 线性 Pipeline（不优化并行）
    麦克风 → KWS → ASR → React Agent → TTS → 扬声器
    """
    print("\n" + "="*60)
    print("🚀 Pipecat 模式 - 初始化中...")
    print("="*60)

    # 1. 初始化现有组件
    print("\n⏳ 正在加载模型...")

    # 创建 wake_word 系统（仅获取模型，不启动循环）
    wake_system = SmartWakeWordSystem(enable_voice=False)

    # 2. 创建 Pipecat Processors
    print("\n⏳ 正在创建 Pipecat Processors...")

    kws_proc = SherpaKWSProcessor(wake_system.kws_model)
    asr_proc = SherpaASRProcessor(wake_system.asr_model)
    agent_proc = ReactAgentProcessor(wake_system.agent)
    tts_proc = PiperTTSProcessor(wake_system.agent.tts)

    print("✓ KWS Processor 已创建")
    print("✓ ASR Processor 已创建")
    print("✓ React Agent Processor 已创建")
    print("✓ TTS Processor 已创建")

    # 3. 创建音频传输
    print("\n⏳ 正在创建音频传输...")
    transport = SimplePyAudioTransport(sample_rate=16000)
    await transport.start()

    # 4. 构建 Pipeline（线性结构）
    print("\n⏳ 正在构建 Pipeline...")

    pipeline = Pipeline([
        kws_proc,
        asr_proc,
        agent_proc,
        tts_proc,
    ])

    print("✓ Pipeline 已构建")
    print("\n" + "="*60)
    print("✓ Pipecat 模式启动完成！")
    print("="*60)
    print("\n💬 说出唤醒词开始对话...")
    print("   默认唤醒词: 小智、你好助手、智能助手")
    print("   按 Ctrl+C 退出\n")

    return pipeline, transport


async def run_pipeline_with_audio(pipeline, transport):
    """
    运行 Pipeline 并处理音频 I/O
    """
    try:
        # 创建两个任务：
        # 1. 音频输入 → Pipeline
        # 2. Pipeline → 音频输出

        async def audio_input_task():
            """音频输入任务"""
            async for audio_frame in transport.read_audio_frames():
                # 推送音频帧到 Pipeline
                await pipeline.process_frame(audio_frame, pipeline.FrameDirection.DOWNSTREAM)

        async def audio_output_task():
            """音频输出任务"""
            # 获取 Pipeline 的输出帧
            async for frame in pipeline.get_output_frames():
                # 写入音频帧到扬声器
                await transport.write_audio_frame(frame)

        # 并行运行两个任务
        await asyncio.gather(
            audio_input_task(),
            audio_output_task()
        )

    except asyncio.CancelledError:
        print("\n⏹️  Pipeline 已停止")
    except Exception as e:
        print(f"\n❌ Pipeline 运行错误: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Pipecat 主程序"""
    pipeline = None
    transport = None

    try:
        # 创建 Pipeline
        pipeline, transport = await create_pipecat_pipeline()

        # 设置信号处理（Ctrl+C 优雅退出）
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()

        def signal_handler(sig, frame):
            print("\n⏹️  收到退出信号...")
            stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)

        # 运行 Pipeline
        pipeline_task = asyncio.create_task(
            run_pipeline_with_audio(pipeline, transport)
        )

        # 等待退出信号
        await stop_event.wait()

        # 取消 Pipeline 任务
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass

    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # 清理资源
        if transport:
            await transport.stop()

        print("\n👋 再见！")


if __name__ == "__main__":
    asyncio.run(main())
