"""音频预处理器 - RNNoise 降噪 + soxr 高质量重采样"""
import numpy as np
from pathlib import Path
from typing import Optional

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, AudioRawFrame


class RNNoiseProcessor(FrameProcessor):
    """
    RNNoise 降噪处理器（使用 soxr 高质量重采样）

    特性：
    - ✅ 使用 RNNoise（Xiph）进行高质量降噪
    - ✅ 使用 soxr 进行 16kHz ↔ 48kHz 高质量重采样
    - ✅ 直接处理每个音频帧（无缓冲延迟）
    - ✅ 低延迟（~10ms）
    - ✅ 深度降噪，适合语音识别

    Pipeline 位置：
        transport.input() → RNNoiseProcessor → KWS → ASR → ...
    """

    def __init__(self, enable_debug=False):
        """
        初始化 RNNoise 降噪器

        Args:
            enable_debug: 启用调试日志
        """
        super().__init__()
        self.enable_debug = enable_debug

        # 音频参数
        self.input_sample_rate = 16000  # 系统采样率
        self.rnnoise_sample_rate = 48000  # RNNoise 原生采样率
        self.frame_size = 480  # RNNoise 帧大小（10ms @ 48kHz）

        # RNNoise 实例（延迟初始化）
        self.denoiser = None
        self.denoiser_initialized = False

        # 统计信息
        self.processed_samples = 0

        # 尝试导入依赖
        try:
            import soxr
            self.soxr = soxr
            print("✓ soxr 已导入")
        except ImportError:
            raise ImportError(
                "soxr 未安装。请运行: uv add soxr\n"
                "或: pip install soxr"
            )

        try:
            from pyrnnoise import RNNoise
            self.RNNoise = RNNoise
            print("✓ pyrnnoise 已导入")
        except ImportError:
            raise ImportError(
                "pyrnnoise 未安装。请运行: uv add pyrnnoise\n"
                "或: pip install pyrnnoise"
            )

    def _init_denoiser(self):
        """延迟初始化 RNNoise（避免在 import 时初始化）"""
        if not self.denoiser_initialized:
            self.denoiser = self.RNNoise(sample_rate=self.rnnoise_sample_rate)
            self.denoiser_initialized = True
            print(f"✓ RNNoise 已初始化（{self.rnnoise_sample_rate}kHz）")

    async def process_frame(self, frame, direction):
        """处理音频帧，进行降噪"""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            # 确保初始化
            self._init_denoiser()

            # 提取音频数据（int16 → float32）
            audio_int16 = np.frombuffer(frame.audio, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # 🔧 使用 soxr 上采样：16kHz → 48kHz
            # 强制输出长度为输入长度的 3 倍（保持时间对齐）
            target_length_48khz = int(len(audio_float32) * self.rnnoise_sample_rate / self.input_sample_rate)
            audio_48khz = self.soxr.resample(
                audio_float32,
                self.rnnoise_sample_rate,  # 48000
                self.input_sample_rate,    # 16000
                quality="HQ"  # 高质量模式
            )

            # 确保长度正确
            if len(audio_48khz) != target_length_48khz:
                if len(audio_48khz) > target_length_48khz:
                    audio_48khz = audio_48khz[:target_length_48khz]
                else:
                    audio_48khz = np.pad(audio_48khz, (0, target_length_48khz - len(audio_48khz)))

            # RNNoise 降噪（逐帧处理）
            denoised_48khz = []

            for i in range(0, len(audio_48khz), self.frame_size):
                frame_data = audio_48khz[i:i + self.frame_size]

                # 填充最后一帧
                if len(frame_data) < self.frame_size:
                    frame_data = np.pad(frame_data, (0, self.frame_size - len(frame_data)))

                # 转换为 int16
                frame_int16 = (frame_data * 32767).astype(np.int16)

                try:
                    speech_prob, denoised_int16 = self.denoiser.denoise_frame(frame_int16)
                    denoised_float = denoised_int16.astype(np.float32) / 32767.0

                    # 截断填充部分
                    denoised_float = denoised_float[:min(len(frame_data), self.frame_size)]
                    denoised_48khz.append(denoised_float)
                except Exception as e:
                    if self.enable_debug:
                        print(f"⚠️ RNNoise 降噪失败: {e}，使用原始音频")
                    denoised_48khz.append(frame_data)

            # 拼接降噪后的音频
            denoised_48khz = np.concatenate(denoised_48khz)

            # 🔧 使用 soxr 下采样：48kHz → 16kHz
            # 强制输出长度与原始音频一致
            target_length_16khz = len(audio_float32)
            denoised_16khz = self.soxr.resample(
                denoised_48khz,
                self.input_sample_rate,    # 16000
                self.rnnoise_sample_rate,  # 48000
                quality="HQ"  # 高质量模式
            )

            # 确保长度与原始音频一致
            if len(denoised_16khz) != target_length_16khz:
                if len(denoised_16khz) > target_length_16khz:
                    denoised_16khz = denoised_16khz[:target_length_16khz]
                else:
                    denoised_16khz = np.pad(denoised_16khz, (0, target_length_16khz - len(denoised_16khz)))

            # 统计
            self.processed_samples += len(denoised_16khz)
            if self.enable_debug and self.processed_samples % 16000 == 0:  # 每秒
                print(f"📊 RNNoise: 已处理 {self.processed_samples / 16000:.1f} 秒")

            # 转换回 int16
            denoised_int16 = (denoised_16khz * 32768.0).astype(np.int16)

            # 生成降噪后的音频帧
            denoised_frame = AudioRawFrame(
                audio=denoised_int16.tobytes(),
                sample_rate=self.input_sample_rate,
                num_channels=1
            )

            await self.push_frame(denoised_frame, direction)
        else:
            # 其他帧直接传递
            await self.push_frame(frame, direction)


class SimpleNoiseGateProcessor(FrameProcessor):
    """
    简单噪声门处理器（备用方案）

    如果 RNNoise 不可用，可以使用这个简单的噪声门：
    - 检测低音量信号（可能是噪声）
    - 衰减或静音低于阈值的信号
    - 无延迟，轻量级

    Args:
        threshold_db: 噪声门阈值（分贝），默认 -40dB
        attack_ms: 启动时间（毫秒）
        release_ms: 释放时间（毫秒）
    """

    def __init__(self, threshold_db=-40, attack_ms=5, release_ms=50):
        super().__init__()
        self.threshold = 10 ** (threshold_db / 20)  # 转换为线性幅度
        self.attack_coeff = np.exp(-1 / (attack_ms * 0.001 * 16000))
        self.release_coeff = np.exp(-1 / (release_ms * 0.001 * 16000))
        self.gain = 0.0

    async def process_frame(self, frame, direction):
        """处理音频帧，应用噪声门"""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            # 提取音频数据
            audio = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0

            # 计算信号幅度（RMS）
            rms = np.sqrt(np.mean(audio ** 2)) + 1e-10

            # 目标增益
            target_gain = 1.0 if rms > self.threshold else 0.0

            # 平滑增益变化
            if target_gain > self.gain:
                self.gain = self.attack_coeff * self.gain + (1 - self.attack_coeff) * target_gain
            else:
                self.gain = self.release_coeff * self.gain + (1 - self.release_coeff) * target_gain

            # 应用增益
            processed = audio * self.gain

            # 转换回 int16
            processed_int16 = (processed * 32768.0).astype(np.int16)

            # 生成处理后的帧
            processed_frame = AudioRawFrame(
                audio=processed_int16.tobytes(),
                sample_rate=16000,
                num_channels=1
            )

            await self.push_frame(processed_frame, direction)
        else:
            await self.push_frame(frame, direction)


class PassThroughProcessor(FrameProcessor):
    """
    直通处理器（用于测试）

    不做任何处理，直接传递音频帧
    """

    async def process_frame(self, frame, direction):
        """直接传递帧"""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def create_noise_reduction_processor(method="rnnoise", **kwargs):
    """
    工厂函数：创建降噪处理器

    Args:
        method: 降噪方法（"rnnoise"、"noise_gate" 或 "pass_through"）
        **kwargs: 传递给处理器的参数

    Returns:
        FrameProcessor: 降噪处理器实例

    Examples:
        # 使用 RNNoise（推荐）
        processor = create_noise_reduction_processor("rnnoise", enable_debug=True)

        # 使用简单噪声门（备用）
        processor = create_noise_reduction_processor("noise_gate", threshold_db=-40)

        # 直通模式（测试）
        processor = create_noise_reduction_processor("pass_through")
    """
    if method == "rnnoise":
        return RNNoiseProcessor(**kwargs)
    elif method == "noise_gate":
        return SimpleNoiseGateProcessor(**kwargs)
    elif method == "pass_through":
        return PassThroughProcessor()
    else:
        raise ValueError(f"未知的降噪方法: {method}。支持: 'rnnoise', 'noise_gate', 'pass_through'")
