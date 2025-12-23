"""TTS语音播报管理器（流式版本）"""
import threading
import time
import wave
from pathlib import Path
import pyaudio
import requests

from .config import (
    DASHSCOPE_API_KEY,
    TTS_AUDIO_DIR,
    TTS_CACHE_TIMEOUT_SHORT,
)


class TTSManager:
    """
    流式TTS语音播报管理器（原 TTSManagerStreaming）

    支持：
    - piper（本地，最快，延迟<100ms，推荐）
    - dashscope（阿里云，音质好）
    - edge TTS（免费，但中文支持差）
    - azure Speech Services（高质量，需付费）
    - coqui TTS（本地）
    """

    def __init__(self, engine_type="piper", api_key=None, voice=None, model_path=None):
        """
        初始化流式TTS

        Args:
            engine_type: "piper"（最快）, "dashscope"（推荐）, "edge"（免费）, "azure"（高质量）, "coqui"（本地）
            api_key: DashScope/Azure API key
            voice: 自定义音色名称
            model_path: Piper 模型路径（仅 piper 引擎需要）
        """
        self.is_playing = False
        self.stream = None
        self.engine_type = engine_type

        # Piper 引擎（本地，最快）
        if engine_type == "piper":
            from piper import PiperVoice
            from pathlib import Path

            # 默认模型路径
            if model_path is None:
                model_path = Path(__file__).parent.parent.parent / "models" / "piper" / "zh_CN-huayan-medium.onnx"

            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Piper 模型文件不存在: {model_path}\n"
                    f"请运行: uv run download_piper_model.py"
                )

            print(f"正在加载 Piper 模型: {model_path}")
            self.piper_voice = PiperVoice.load(str(model_path))
            self.p = pyaudio.PyAudio()
            self.should_stop = False
            self.current_stream = None

            print(f"✓ 使用 Piper TTS（本地，超快）- 模型: {Path(model_path).name}")
            return

        # DashScope 引擎
        if engine_type == "dashscope":
            self.api_key = api_key or DASHSCOPE_API_KEY
            self.voice = voice or "Cherry"

            try:
                import dashscope
                dashscope.api_key = self.api_key
                dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
                self.dashscope = dashscope
            except ImportError:
                raise ImportError("需要安装 dashscope: pip install dashscope")

            self.audio_dir = TTS_AUDIO_DIR
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            self.p = pyaudio.PyAudio()
            self.should_stop = False
            self.current_stream = None

            print(f"✓ 使用 DashScope TTS（阿里云）- 音色: {self.voice}")
            return

        # 导入 RealtimeTTS（按需导入）
        try:
            from RealtimeTTS import TextToAudioStream
        except ImportError:
            raise ImportError("需要安装 RealtimeTTS: pip install realtimetts")

        # 选择引擎
        if engine_type == "edge":
            # Edge TTS（免费）
            try:
                from RealtimeTTS import SystemEngine
            except ImportError:
                raise ImportError("需要安装 SystemEngine: pip install realtimetts[system] 或 pip install pyttsx3")

            default_voice = "zh-CN-XiaoxiaoNeural"  # 晓晓音色（温柔女声）
            self.engine = SystemEngine(
                voice=voice or default_voice
            )
            print(f"✓ 使用 Edge TTS（免费）- 音色: {voice or default_voice}")

        elif engine_type == "azure":
            # Azure（音质最好）
            try:
                from RealtimeTTS import AzureEngine
            except ImportError:
                raise ImportError("需要安装 Azure 引擎: pip install realtimetts[azure]")

            if not api_key:
                raise ValueError("Azure 引擎需要 API key")
            default_voice = "zh-CN-XiaoxiaoNeural"
            self.engine = AzureEngine(
                speech_key=api_key,
                speech_region="eastasia",  # 东亚区域
                voice=voice or default_voice
            )
            print(f"✓ 使用 Azure TTS - 音色: {voice or default_voice}")

        elif engine_type == "coqui":
            # Coqui（本地）
            try:
                from RealtimeTTS import CoquiEngine
            except ImportError:
                raise ImportError("需要安装 Coqui 引擎: pip install realtimetts[coqui]")

            self.engine = CoquiEngine(
                language="zh"
            )
            print("✓ 使用 Coqui TTS（本地）")

        else:
            raise ValueError(f"不支持的引擎类型: {engine_type}")

        # 创建流
        self.stream = TextToAudioStream(self.engine)
        print(f"✓ RealtimeTTS 流式引擎已初始化")

    def _play_audio_file(self, audio_file):
        """使用PyAudio直接播放音频文件（DashScope 引擎使用）"""
        stream = None
        try:
            self.is_playing = True
            self.should_stop = False

            with wave.open(str(audio_file), 'rb') as wf:
                stream = self.p.open(
                    format=self.p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                self.current_stream = stream

                chunk_size = 1024
                data = wf.readframes(chunk_size)
                while data and not self.should_stop:
                    stream.write(data)
                    data = wf.readframes(chunk_size)

            if self.should_stop:
                print("   [TTS已打断]")

            time.sleep(0.1)
        except Exception as e:
            if "Broken pipe" not in str(e):
                print(f"播放音频失败: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            self.current_stream = None
            self.is_playing = False
            self.should_stop = False

    def speak(self, text, voice=None, wait=True):
        """
        流式语音播报

        Args:
            text: 要播报的文本
            voice: 音色（暂不支持动态切换）
            wait: 是否等待播放完成
        """
        if not text or not text.strip():
            return

        text = text.strip()
        print(f"📝 文本长度: {len(text)} 字符")
        print(f"   使用TTS引擎: {self.engine_type}")

        try:
            # Piper 引擎（本地流式，最快）
            if self.engine_type == "piper":
                import numpy as np

                self.is_playing = True
                self.should_stop = False

                # 生成音频（返回生成器，产生 AudioChunk 对象）
                audio_generator = self.piper_voice.synthesize(text)

                # 遍历所有 AudioChunk（可能有多个）
                for chunk in audio_generator:
                    if self.should_stop:
                        break

                    # 从 AudioChunk 提取音频数据
                    audio_float = chunk.audio_float_array
                    sample_rate = chunk.sample_rate

                    # 转换为 int16 格式
                    audio_int16 = (audio_float * 32767).astype(np.int16)

                    print(f"[Piper] 播放音频块: {len(audio_float)} samples ({len(audio_float)/sample_rate:.1f}秒)")

                    # 创建 PyAudio 流（第一次）
                    if not self.current_stream:
                        self.current_stream = self.p.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=sample_rate,
                            output=True,
                            frames_per_buffer=512
                        )

                    # 分块播放（可快速中断）
                    chunk_size = 512
                    for i in range(0, len(audio_int16), chunk_size):
                        if self.should_stop:
                            break

                        audio_chunk = audio_int16[i:i + chunk_size]
                        self.current_stream.write(audio_chunk.tobytes())

                    if self.should_stop:
                        break

                # 清理
                if self.current_stream:
                    self.current_stream.stop_stream()
                    self.current_stream.close()
                    self.current_stream = None

                self.is_playing = False
                if self.should_stop:
                    print("   [Piper TTS已打断]")

            # DashScope 引擎
            elif self.engine_type == "dashscope":
                self.is_playing = True

                # 调用 DashScope API
                response = self.dashscope.MultiModalConversation.call(
                    model="qwen3-tts-flash",
                    api_key=self.api_key,
                    text=text,
                    voice=self.voice,
                    language_type="Chinese",
                    stream=False
                )

                if response.status_code == 200:
                    audio_url = response.output.audio.url
                    audio_response = requests.get(audio_url, timeout=10)
                    if audio_response.status_code == 200:
                        audio_file = self.audio_dir / f"tts_{int(time.time())}.wav"
                        with open(audio_file, 'wb') as f:
                            f.write(audio_response.content)

                        if wait:
                            self._play_audio_file(audio_file)
                        else:
                            threading.Thread(
                                target=self._play_audio_file,
                                args=(audio_file,),
                                daemon=True
                            ).start()

                        # 10秒后清理临时文件
                        threading.Timer(
                            TTS_CACHE_TIMEOUT_SHORT,
                            lambda: self._delete_file(audio_file)
                        ).start()
                else:
                    print(f"TTS错误: {response.status_code} - {response.message}")

                if wait:
                    self.is_playing = False

            # RealtimeTTS 引擎（edge/azure/coqui）
            else:
                self.is_playing = True

                # 喂入文本（立即开始生成）
                self.stream.feed(text)

                if wait:
                    # 同步播放（阻塞）
                    self.stream.play()
                    self.is_playing = False
                else:
                    # 异步播放（非阻塞）
                    self.stream.play_async()

        except Exception as e:
            print(f"TTS播放失败: {e}")
            import traceback
            traceback.print_exc()
            if wait:
                self.is_playing = False

    def _delete_file(self, filepath):
        """删除临时文件"""
        try:
            if filepath.exists():
                filepath.unlink()
        except:
            pass

    def speak_async(self, text, voice=None):
        """异步播放（不阻塞）"""
        self.speak(text, voice, wait=False)

    def stop(self):
        """停止播放（立即打断）"""
        if self.is_playing:
            # Piper/DashScope 引擎（使用 PyAudio）
            if self.engine_type in ("piper", "dashscope"):
                self.should_stop = True
                self.is_playing = False
                if self.current_stream:
                    try:
                        self.current_stream.stop_stream()
                        self.current_stream.close()
                    except:
                        pass
                self.current_stream = None
                engine_name = "Piper TTS" if self.engine_type == "piper" else "DashScope TTS"
                print(f"   [{engine_name}已打断]")

            # RealtimeTTS 引擎
            else:
                if self.stream:
                    try:
                        self.stream.stop()
                        self.is_playing = False
                        print("   [流式TTS已打断]")
                    except Exception as e:
                        print(f"停止TTS失败: {e}")

    def __del__(self):
        """清理资源"""
        if self.engine_type in ("piper", "dashscope"):
            try:
                self.p.terminate()
            except:
                pass
        else:
            if self.stream:
                try:
                    self.stream.stop()
                except:
                    pass
