"""
Qwen-TTS-Realtime WebSocket 流式语音合成

基于 DashScope WebSocket API 的实时语音合成实现。
相比 HTTP 流式，WebSocket 提供更低延迟和更好的实时性。

官方文档：
- https://help.aliyun.com/zh/model-studio/qwen-tts-realtime
- https://github.com/aliyun/alibabacloud-bailian-speech-demo

核心特性：
- WebSocket 持久连接 + 事件驱动
- 两种交互模式：server_commit（自动） / commit（手动）
- 增量音频流式返回（response.audio.delta）
- 首包延迟优化（~100-200ms）

使用示例：
```python
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback

class MyCallback(QwenTtsRealtimeCallback):
    def on_event(self, response):
        if response['type'] == 'response.audio.delta':
            audio_data = base64.b64decode(response['delta'])
            # 播放音频

tts = QwenTtsRealtime(
    model='qwen3-tts-flash-realtime',
    callback=MyCallback(),
    url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
)
tts.connect()
tts.update_session(voice='Cherry', mode='server_commit')
tts.append_text('你好，世界')
tts.finish()
```
"""

import asyncio
import base64
import json
import queue  # ✅ 线程安全的 queue
import threading
import time
from typing import Optional, Callable
from pathlib import Path
from dataclasses import dataclass

import pyaudio
import numpy as np

# DashScope WebSocket Realtime TTS
try:
    from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback
except ImportError:
    # 如果未安装 dashscope，延迟导入
    QwenTtsRealtime = None
    QwenTtsRealtimeCallback = None


@dataclass
class RealtimeTTSConfig:
    """Qwen-TTS-Realtime 配置"""
    # 模型配置
    model: str = "qwen3-tts-flash-realtime"  # 或 qwen3-tts-instruct-flash-realtime
    voice: str = "Cherry"
    language_type: str = "Chinese"

    # 交互模式
    mode: str = "server_commit"  # server_commit（自动） | commit（手动）

    # 音频格式
    response_format: str = "pcm"  # pcm | wav | mp3 | opus
    sample_rate: int = 24000  # 8000 | 16000 | 24000 | 48000

    # 高级参数（仅部分模型支持）
    speech_rate: float = 1.0  # [0.5, 2.0]
    volume: int = 50  # [0, 100]
    pitch_rate: float = 1.0  # [0.5, 2.0]

    # 指令控制（仅 qwen3-tts-instruct-flash-realtime）
    instructions: Optional[str] = None
    optimize_instructions: bool = False

    # WebSocket 配置
    url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    api_key: Optional[str] = None


class QwenRealtimeTTS:
    """
    Qwen-TTS-Realtime WebSocket 流式 TTS 实现

    特点：
    - WebSocket 持久连接，低延迟
    - 事件驱动架构
    - 支持中断和流式播放
    - 完全兼容 Pipecat Pipeline

    使用示例：
    ```python
    tts = QwenRealtimeTTS(
        model="qwen3-tts-flash-realtime",
        voice="Cherry",
        api_key="your-api-key"
    )

    # 流式播放
    await tts.speak("你好，世界")

    # 关闭连接
    await tts.close()
    ```
    """

    def __init__(self, config: RealtimeTTSConfig):
        """
        初始化 Qwen-TTS-Realtime

        Args:
            config: TTS 配置
        """
        self.config = config
        self.qwen_tts = None
        self.callback = None

        # ✅ 用线程安全的普通 queue
        self._audio_queue = queue.Queue()
        self.should_stop = False

        # PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None  # ✅ Stream 一直开着
        self._player_thread = None
        self._loop = None

    def _player_thread_func(self):
        """独立播放线程，持续从队列取数据写入 stream"""
        # ✅ 路一直开着，从 initialize() 到 close()
        while not self.should_stop:
            try:
                audio_data = self._audio_queue.get(timeout=0.1)

                if audio_data is None:
                    # 忽略结束信号，stream 继续开着
                    # 下一批数据来了直接接着跑
                    continue

                # ✅ 数据来了直接走，像水管一样一直流
                self.stream.write(audio_data)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 播放失败: {e}")
                break

    async def initialize(self):
        """初始化 WebSocket 连接"""
        try:
            # 检查 QwenTtsRealtime 是否可用
            if QwenTtsRealtime is None:
                raise ImportError(
                    "❌ dashscope.audio.qwen_tts_realtime 模块未找到\n\n"
                    "请按照以下步骤安装：\n"
                    "1. 检查 dashscope 版本：uv run python -c \"import dashscope; print(dashscope.__version__)\"\n"
                    "2. 如果版本 < 1.25.11，请升级：uv add dashscope --upgrade\n"
                    "3. 确保 dashscope 包含 qwen_tts_realtime 模块\n\n"
                    "或者切换到其他 TTS 引擎：\n"
                    "  - TTS_SERVICE=piper (本地，免费)\n"
                    "  - TTS_SERVICE=dashscope (HTTP 流式，稳定)\n"
                    "  - TTS_SERVICE=edge (微软免费)"
                )

            # 获取 API Key（直接使用传入的配置或环境变量）
            api_key = self.config.api_key
            if not api_key:
                import os
                api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")

            if not api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY not set. "
                    "Please set DASHSCOPE_API_KEY in .env file or environment variable."
                )

            # 检查 dashscope 版本
            try:
                import dashscope
                # 设置 API Key 到 dashscope 模块
                dashscope.api_key = api_key
            except ImportError as e:
                raise ImportError(
                    f"需要安装 dashscope>=1.25.11: uv add dashscope --upgrade\n"
                    f"错误详情: {e}"
                )

            # 创建回调处理器
            self.callback = _RealtimeTTSCallback(self._audio_queue)

            # 创建 QwenTtsRealtime 实例
            self.qwen_tts = QwenTtsRealtime(
                model=self.config.model,
                callback=self.callback,
                url=self.config.url
            )

            # 连接 WebSocket
            self.qwen_tts.connect()

            # 配置会话
            from dashscope.audio.qwen_tts_realtime.qwen_tts_realtime import AudioFormat

            self.qwen_tts.update_session(
                voice=self.config.voice,
                response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,  # 使用枚举而不是字符串
                mode=self.config.mode,
                language_type=self.config.language_type,
                speech_rate=self.config.speech_rate,
                volume=self.config.volume,
                pitch_rate=self.config.pitch_rate
            )

            # 保存事件循环引用
            self._loop = asyncio.get_event_loop()

            # ✅ 路，程序启动就开好，一直开着
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                output=True,
                frames_per_buffer=1024,
            )

            # ✅ 启动播放线程（daemon=True，主程序退出时自动结束）
            self._player_thread = threading.Thread(
                target=self._player_thread_func,
                daemon=True
            )
            self._player_thread.start()

            print(f"✓ Qwen-TTS-Realtime 已连接")
            print(f"  - 模型: {self.config.model}")
            print(f"  - 音色: {self.config.voice}")
            print(f"  - 模式: {self.config.mode}")
            print(f"  - 采样率: {self.config.sample_rate}Hz")

        except ImportError as e:
            raise ImportError(
                f"需要安装 dashscope>=1.25.11: uv add dashscope --upgrade\n"
                f"错误详情: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"初始化 Qwen-TTS-Realtime 失败: {e}")

    async def speak(self, text: str):
        """
        流式语音合成并播放

        Args:
            text: 要合成的文本
        """
        if not self.qwen_tts:
            await self.initialize()

        if not text or not text.strip():
            return

        try:
            # ✅ 发送文本，立即返回
            self.qwen_tts.append_text(text)

            if self.config.mode == "commit":
                self.qwen_tts.commit()

            # ✅ 添加短暂延迟，让 WebSocket 处理
            await asyncio.sleep(0.1)

        except Exception as e:
            print(f"❌ TTS 合成失败: {e}")

    def stop(self):
        """停止播放"""
        self.should_stop = True
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except:
                break

    async def close(self):
        """关闭连接"""
        self.stop()

        # 关闭 WebSocket
        if self.qwen_tts:
            try:
                self.qwen_tts.finish()
                self.qwen_tts.close()
            except:
                pass

        # ✅ 只有程序结束时才关路
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass

        # 清理 PyAudio
        if self.p:
            try:
                self.p.terminate()
            except:
                pass


class _RealtimeTTSCallback:
    """
    Qwen-TTS-Realtime 回调处理器

    处理 WebSocket 事件并转发音频数据到播放队列。
    """

    def __init__(self, audio_queue: queue.Queue):
        """
        初始化回调

        Args:
            audio_queue: 音频数据队列（线程安全）
        """
        self.audio_queue = audio_queue
        self.first_audio_time = None
        self.session_start_time = None

    def on_open(self):
        """WebSocket 连接建立"""
        print("✓ WebSocket 连接已建立")
        self.session_start_time = time.time()

    def on_event(self, response: dict):
        """处理服务端事件"""
        event_type = response.get('type', '')

        # 会话已创建
        if event_type == 'session.created':
            session_id = response.get('session', {}).get('id', '')
            # 减少输出：只在首次连接时打印
            if self.session_start_time and (time.time() - self.session_start_time) < 2:
                print(f"✓ 会话已创建: {session_id}")

        # 会话已更新
        elif event_type == 'session.updated':
            pass  # 减少输出

        # 响应已创建
        elif event_type == 'response.created':
            pass  # 减少输出

        # 接收音频数据
        elif event_type == 'response.audio.delta':
            # 记录首包延迟（只记录一次）
            if self.first_audio_time is None:
                self.first_audio_time = time.time()
                first_audio_delay = (self.first_audio_time - self.session_start_time) * 1000
                print(f"✓ 首包延迟: {first_audio_delay:.0f}ms")

            # 解码 Base64 音频
            delta_b64 = response.get('delta', '')
            if delta_b64:
                try:
                    audio_data = base64.b64decode(delta_b64)
                    # ✅ 线程安全，直接 put
                    self.audio_queue.put(audio_data)

                    # 调试：显示音频块大小
                    if len(audio_data) > 0:
                        print(f"  📦 收到音频块: {len(audio_data)} 字节", end='\r')
                except Exception as e:
                    print(f"❌ 解码失败: {e}")

        elif event_type == 'response.done':
            # 发送结束信号
            self.audio_queue.put(None)

        # 会话完成
        elif event_type == 'session.finished':
            pass  # 减少输出

        # 错误
        elif event_type == 'error':
            error = response.get('error', {})
            error_code = error.get('code', '')
            error_msg = error.get('message', '')
            print(f"❌ 错误 [{error_code}]: {error_msg}")

    def on_close(self, close_status_code, close_msg):
        """WebSocket 连接关闭"""
        print(f"✓ WebSocket 连接已关闭: {close_status_code} - {close_msg}")
        self.audio_queue.put(None)


# ==================== 工厂函数 ====================

def create_realtime_tts(
    model: str = "qwen3-tts-flash-realtime",
    voice: str = "Cherry",
    mode: str = "server_commit",
    api_key: Optional[str] = None,
    **kwargs
) -> QwenRealtimeTTS:
    """
    创建 Qwen-TTS-Realtime 实例（便捷函数）

    Args:
        model: 模型名称
        voice: 音色
        mode: 交互模式（server_commit | commit）
        api_key: API 密钥
        **kwargs: 其他配置参数

    Returns:
        QwenRealtimeTTS 实例

    使用示例：
    ```python
    tts = create_realtime_tts(
        model="qwen3-tts-flash-realtime",
        voice="Cherry",
        mode="server_commit"
    )

    await tts.speak("你好，世界")
    ```
    """
    config = RealtimeTTSConfig(
        model=model,
        voice=voice,
        mode=mode,
        api_key=api_key,
        **kwargs
    )

    return QwenRealtimeTTS(config)
