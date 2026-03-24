#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频设备诊断工具

列出所有可用的音频输入设备，帮助选择正确的麦克风
"""
import pyaudio


def list_audio_devices():
    """列出所有音频设备"""
    p = pyaudio.PyAudio()

    print("=" * 80)
    print("🎤 音频输入设备列表")
    print("=" * 80)

    input_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)

        # 只显示输入设备
        if info['maxInputChannels'] > 0:
            input_devices.append((i, info))

            # 设备类型判断
            device_name = info['name']
            device_type = "未知"

            # 检测常见的问题设备
            if any(keyword in device_name.lower() for keyword in
                   ['stereo mix', '立体声混音', 'what u hear', '您听到的', 'loopback', '回环']):
                device_type = "⚠️  系统混音设备（避免选择）"
            elif any(keyword in device_name.lower() for keyword in
                     ['microphone', '麦克风', 'mic', '输入设备']):
                device_type = "✅ 麦克风设备（推荐）"
            elif any(keyword in device_name.lower() for keyword in
                     ['headset', '耳机', 'headphone']):
                device_type = "🎧 耳机麦克风"

            # 打印设备信息
            print(f"\n设备 #{i}: {device_name}")
            print(f"  类型: {device_type}")
            print(f"  采样率: {int(info['defaultSampleRate'])} Hz")
            print(f"  输入通道: {info['maxInputChannels']}")
            print(f"  默认输入设备: {'是' if info['maxInputChannels'] > 0 else '否'}")

    print("\n" + "=" * 80)
    print("🎤 音频输出设备列表")
    print("=" * 80)

    output_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)

        # 只显示输出设备
        if info['maxOutputChannels'] > 0:
            output_devices.append((i, info))

            device_name = info['name']
            device_type = "扬声器设备"

            if any(keyword in device_name.lower() for keyword in
                   ['headset', '耳机', 'headphone']):
                device_type = "🎧 耳机扬声器"
            elif any(keyword in device_name.lower() for keyword in
                     ['speaker', '扬声器', '音箱']):
                device_type = "🔊 扬声器"

            print(f"\n设备 #{i}: {device_name}")
            print(f"  类型: {device_type}")
            print(f"  采样率: {int(info['defaultSampleRate'])} Hz")
            print(f"  输出通道: {info['maxOutputChannels']}")

    print("\n" + "=" * 80)
    print("💡 使用建议")
    print("=" * 80)
    print("\n1. 查看上面的设备列表，找到你的麦克风设备")
    print("2. 避免选择 '立体声混音'、'Stereo Mix' 或 'What U Hear' 设备")
    print("3. 记下你的麦克风设备编号（例如：设备 #1）")
    print("4. 在 .env 文件中添加配置：")
    print("   AUDIO_INPUT_DEVICE_INDEX=1")
    print("   AUDIO_OUTPUT_DEVICE_INDEX=5")
    print("\n如果不配置，系统将使用默认设备（可能导致问题）")

    p.terminate()


if __name__ == "__main__":
    list_audio_devices()
