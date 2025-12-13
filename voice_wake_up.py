#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作版本的中文关键词唤醒程序 - 使用正确的API
基于test_recognizer_api.py的发现
"""
import sherpa_onnx
import numpy as np
import pyaudio
import threading
import time
import wave
from pathlib import Path
import queue
import sys
import os

class WorkingWakeWordV3:
    def __init__(self, models_dir="models"):
        """初始化中文语音唤醒系统"""
        self.models_dir = Path(models_dir)
        self.running = False
        self.audio_queue = queue.Queue()

        # 音频参数
        self.sample_rate = 16000
        self.frame_duration = 0.1  # 100ms
        self.frames_per_buffer = int(self.sample_rate * self.frame_duration)
        self.chunk_size = 1024

        print("正在初始化中文语音唤醒系统...")
        self.keywords = ["小智", "你好助手", "智能助手", "小爱同学", "小爱", "助手"]

        # 测试识别器创建
        try:
            self.recognizer = self.create_recognizer()
            print("识别器创建成功!")
            self.recognizer_working = True
        except Exception as e:
            print(f"识别器创建失败: {e}")
            self.recognizer_working = False

        print(f"监听关键词: {', '.join(self.keywords)}")

    def create_recognizer(self):
        """创建离线识别器 - 使用正确的API"""
        # 查找模型文件
        model_file = self.models_dir / "sherpa-onnx-paraformer-zh-2024-03-09" / "model.int8.onnx"
        tokens_file = self.models_dir / "sherpa-onnx-paraformer-zh-2024-03-09" / "tokens.txt"

        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        if not tokens_file.exists():
            raise FileNotFoundError(f"Token文件不存在: {tokens_file}")

        print(f"使用模型文件: {model_file.name}")
        print(f"使用Token文件: {tokens_file.name}")

        # 使用正确的API创建识别器（位置参数）
        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            str(model_file),  # paraformer参数
            str(tokens_file),  # tokens参数
            num_threads=2,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            provider="cpu"
        )
        return recognizer

    def start_listening(self):
        """开始监听"""
        print("\n" + "="*60)
        print("中文语音唤醒系统 - V3版本")
        print("请说出: '小智', '你好助手', '智能助手', '小爱同学'")
        print("按 Ctrl+C 停止")
        print("="*60)

        self.running = True

        try:
            # 初始化PyAudio
            p = pyaudio.PyAudio()

            # 获取默认输入设备
            device_info = p.get_default_input_device_info()
            print(f"使用麦克风: {device_info['name']}")

            # 创建音频流
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )

            # 启动音频处理线程
            process_thread = threading.Thread(target=self._process_audio)
            process_thread.daemon = True
            process_thread.start()

            # 开始录音
            stream.start_stream()
            print("开始监听...")
            print("请说出唤醒词...")

            while self.running:
                time.sleep(0.1)

            # 停止并清理
            stream.stop_stream()
            stream.close()
            p.terminate()

        except KeyboardInterrupt:
            print("\n停止监听...")
        except Exception as e:
            print(f"音频错误: {e}")
        finally:
            self.running = False
            print("监听已停止")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if status:
            print(f"音频状态: {status}")

        # 将音频数据放入队列
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def _process_audio(self):
        """处理音频数据"""
        speech_buffer = []
        is_speaking = False
        silence_count = 0

        while self.running:
            try:
                # 获取音频数据
                if self.audio_queue.empty():
                    time.sleep(0.01)
                    continue

                audio_bytes = self.audio_queue.get(timeout=0.1)
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

                # 使用音量检测
                volume = np.sqrt(np.mean(audio_data**2))

                if volume > 0.05 and not is_speaking:
                    # 开始说话
                    is_speaking = True
                    print(f"检测到声音 (音量: {volume:.3f})...")
                    speech_buffer = [audio_data]
                    silence_count = 0
                elif volume > 0.05 and is_speaking:
                    # 继续说话
                    speech_buffer.append(audio_data)
                    silence_count = 0
                elif volume <= 0.05 and is_speaking:
                    # 声音变小，可能是停顿
                    silence_count += 1
                    speech_buffer.append(audio_data)

                    # 如果静音时间超过一定阈值，认为说话结束
                    if silence_count > 30:  # 约3秒静音
                        is_speaking = False
                        if len(speech_buffer) > 30:  # 至少有1秒的声音
                            self._process_speech_chunk(speech_buffer)
                        speech_buffer = []
                        silence_count = 0

                # 防止缓冲区过长
                if len(speech_buffer) > int(5 * self.sample_rate / self.chunk_size):  # 最多5秒
                    print("语音片段过长，跳过")
                    is_speaking = False
                    speech_buffer = []
                    silence_count = 0

            except Exception as e:
                print(f"音频处理错误: {e}")

    def _process_speech_chunk(self, audio_chunks):
        """处理语音片段"""
        try:
            # 保存音频到临时文件
            audio_data = np.concatenate(audio_chunks)
            temp_file = "temp_speech.wav"

            # 保存为WAV文件
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

            # 使用离线识别器识别
            try:
                if self.recognizer_working:
                    # 识别音频
                    with wave.open(temp_file, 'rb') as wf:
                        sample_rate = wf.getframerate()
                        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                        samples = samples.astype(np.float32) / 32768.0

                    stream = self.recognizer.create_stream()
                    stream.accept_waveform(sample_rate, samples)
                    self.recognizer.decode_stream(stream)
                    text = stream.result.text.strip()

                    if text:
                        print(f"识别结果: {text}")
                        # 检查关键词
                        for keyword in self.keywords:
                            if keyword in text:
                                print(f"\n!!! 检测到唤醒词: {keyword} !!!")
                                print(f"   完整文本: {text}")
                                print(f"   时间: {time.strftime('%H:%M:%S')}")
                                print("-"*60)
                                self._on_wake_up(keyword)
                                break
                else:
                    print("识别器不可用，跳过识别")

            except Exception as e:
                print(f"识别错误: {e}")
            finally:
                # 删除临时文件
                try:
                    Path(temp_file).unlink()
                except:
                    pass

        except Exception as e:
            print(f"处理语音片段错误: {e}")

    def _on_wake_up(self, keyword):
        """唤醒回调函数"""
        print(f"唤醒成功! 关键词: {keyword}")

        # 播放提示音（Windows环境）
        try:
            import winsound
            # 播放系统提示音
            winsound.Beep(800, 200)  # 800Hz, 200ms
            print("已播放唤醒提示音")
        except ImportError:
            # 如果没有winsound，使用系统命令
            try:
                os.system("echo -ne '\\a'")
            except:
                pass

        # 这里可以添加更多唤醒后的操作
        print("语音助手已激活，请继续说话...")

def main():
    """主函数"""
    print("="*60)
    print("中文语音唤醒系统 - V3版本")
    print("使用正确的Sherpa-ONNX API")
    print("="*60)

    try:
        # 创建唤醒系统
        wake_system = WorkingWakeWordV3()

        # 开始监听
        wake_system.start_listening()

    except KeyboardInterrupt:
        print("\n程序退出")
    except Exception as e:
        print(f"启动失败: {e}")
        print("请确保:")
        print("   1. 所有依赖已安装: pip install pyaudio")
        print("   2. 模型文件完整")
        print("   3. 麦克风可用")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()