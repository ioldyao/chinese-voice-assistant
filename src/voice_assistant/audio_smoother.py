"""
音频平滑处理器

解决流式 TTS 播放时的电音和卡顿问题：
- 音频块之间的电平突变导致爆音
- 缺少淡入淡出导致听起来一段一段的

解决方案：
1. 淡入淡出（Fade In/Out）：音频块开始和结束时平滑过渡
2. 交叉淡化（Cross-fade）：音频块之间重叠混合
3. 音频缓冲：确保连续播放

使用示例：
```python
smoother = AudioSmoother(sample_rate=16000)

# 处理每个音频块
smoothed_audio = smoother.smooth(audio_chunk)
```
"""

import numpy as np
from typing import Optional


class AudioSmoother:
    """
    音频平滑处理器

    特点：
    - 淡入淡出（Fade In/Out）
    - 交叉淡化（Cross-fade）
    - 防止电平突变

    参数：
        sample_rate: 采样率（Hz）
        fade_duration: 淡入淡出时长（秒），默认 0.01（10ms）
        cross_fade_duration: 交叉淡化时长（秒），默认 0.005（5ms）
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        fade_duration: float = 0.01,  # 10ms 淡入淡出
        cross_fade_duration: float = 0.005  # 5ms 交叉淡化
    ):
        self.sample_rate = sample_rate
        self.fade_duration = fade_duration
        self.cross_fade_duration = cross_fade_duration

        # 计算样本数
        self.fade_samples = int(fade_duration * sample_rate)
        self.cross_fade_samples = int(cross_fade_duration * sample_rate)

        # 保存上一个音频块的尾部（用于交叉淡化）
        self.prev_tail: Optional[np.ndarray] = None

        # 预计算淡入淡出曲线
        self._fade_in_curve = self._create_fade_in_curve()
        self._fade_out_curve = self._create_fade_out_curve()

    def _create_fade_in_curve(self) -> np.ndarray:
        """
        创建淡入曲线（0 → 1）

        使用余弦曲线实现平滑淡入
        """
        if self.fade_samples == 0:
            return np.array([1.0])

        x = np.linspace(0, np.pi, self.fade_samples)
        curve = (1 - np.cos(x)) / 2  # 余弦曲线：0 → 1
        return curve.astype(np.float32)

    def _create_fade_out_curve(self) -> np.ndarray:
        """
        创建淡出曲线（1 → 0）

        使用余弦曲线实现平滑淡出
        """
        if self.fade_samples == 0:
            return np.array([1.0])

        x = np.linspace(0, np.pi, self.fade_samples)
        curve = (1 + np.cos(x)) / 2  # 余弦曲线：1 → 0
        return curve.astype(np.float32)

    def smooth(self, audio: np.ndarray) -> np.ndarray:
        """
        平滑处理音频块

        Args:
            audio: 输入音频（int16 或 float32）

        Returns:
            平滑后的音频（与输入相同类型）
        """
        if len(audio) == 0:
            return audio

        # 转换为 float32 进行处理
        if audio.dtype == np.int16:
            is_int16 = True
            audio_float = audio.astype(np.float32) / 32767.0
        else:
            is_int16 = False
            audio_float = audio.astype(np.float32)

        # 应用淡入淡出
        audio_float = self._apply_fade(audio_float)

        # 应用交叉淡化
        audio_float = self._apply_cross_fade(audio_float)

        # 转换回原始类型
        if is_int16:
            result = (audio_float * 32767.0).astype(np.int16)
        else:
            result = audio_float.astype(np.float32)

        return result

    def _apply_fade(self, audio: np.ndarray) -> np.ndarray:
        """
        应用淡入淡出

        Args:
            audio: float32 音频，范围 [-1.0, 1.0]

        Returns:
            淡入淡出后的音频
        """
        result = audio.copy()

        # 淡入（开头）
        if len(result) > self.fade_samples * 2:
            fade_in_len = min(self.fade_samples, len(result) // 2)
            result[:fade_in_len] *= self._fade_in_curve[:fade_in_len]

        # 淡出（结尾）
        if len(result) > self.fade_samples * 2:
            fade_out_len = min(self.fade_samples, len(result) // 2)
            result[-fade_out_len:] *= self._fade_out_curve[-fade_out_len:]

        return result

    def _apply_cross_fade(self, audio: np.ndarray) -> np.ndarray:
        """
        应用交叉淡化

        将上一个音频块的尾部与当前音频块的开头混合

        Args:
            audio: float32 音频，范围 [-1.0, 1.0]

        Returns:
            交叉淡化后的音频
        """
        if self.prev_tail is None:
            # 第一个音频块，只保存尾部
            if len(audio) > self.cross_fade_samples:
                self.prev_tail = audio[-self.cross_fade_samples:].copy()
            return audio

        # 交叉淡化
        if len(audio) < self.cross_fade_samples:
            # 音频块太短，直接返回
            self.prev_tail = None
            return audio

        # 创建交叉淡化曲线
        cross_fade_curve = np.linspace(0, 1, self.cross_fade_samples).astype(np.float32)

        # 混合上一个尾部和当前开头
        prev_part = self.prev_tail * (1 - cross_fade_curve)  # 淡出
        curr_part = audio[:self.cross_fade_samples] * cross_fade_curve  # 淡入
        cross_faded = prev_part + curr_part

        # 组合音频
        result = np.concatenate([
            audio[:-self.cross_fade_samples],  # 移除原始开头
            cross_faded,  # 交叉淡化部分
            audio[self.cross_fade_samples:]  # 剩余部分
        ])

        # 更新尾部
        self.prev_tail = result[-self.cross_fade_samples:].copy()

        return result

    def reset(self):
        """重置状态（在切换对话时调用）"""
        self.prev_tail = None


# ==================== 便捷函数 ====================

def create_audio_smoother(
    sample_rate: int = 16000,
    fade_duration: float = 0.01,
    cross_fade_duration: float = 0.005
) -> AudioSmoother:
    """
    创建音频平滑器（便捷函数）

    Args:
        sample_rate: 采样率
        fade_duration: 淡入淡出时长（秒）
        cross_fade_duration: 交叉淡化时长（秒）

    Returns:
        AudioSmoother 实例

    使用示例：
    ```python
    smoother = create_audio_smoother(sample_rate=16000)

    # 处理音频
    smoothed = smoother.smooth(audio_chunk)
    ```
    """
    return AudioSmoother(
        sample_rate=sample_rate,
        fade_duration=fade_duration,
        cross_fade_duration=cross_fade_duration
    )


def smooth_audio_chunk(
    audio: np.ndarray,
    sample_rate: int = 16000,
    smoother: Optional[AudioSmoother] = None
) -> np.ndarray:
    """
    平滑处理音频块（便捷函数）

    Args:
        audio: 输入音频（int16 或 float32）
        sample_rate: 采样率
        smoother: 可选的 AudioSmoother 实例（如果为 None，会创建新的）

    Returns:
        平滑后的音频

    使用示例：
    ```python
    # 简单使用（每次创建新的 smoother）
    smoothed = smooth_audio_chunk(audio_chunk, sample_rate=16000)

    # 高效使用（复用 smoother）
    smoother = AudioSmoother(sample_rate=16000)
    for chunk in audio_chunks:
        smoothed = smoother.smooth(chunk)
    ```
    """
    if smoother is None:
        smoother = AudioSmoother(sample_rate=sample_rate)

    return smoother.smooth(audio)
