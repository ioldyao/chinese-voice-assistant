"""音频设备交互式配置工具"""
import os
import pyaudio
from pathlib import Path


def list_audio_devices():
    """列出所有音频设备"""
    p = pyaudio.PyAudio()

    input_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)

        # 只显示输入设备
        if info['maxInputChannels'] > 0:
            input_devices.append((i, info))

    p.terminate()
    return input_devices


def display_audio_devices():
    """显示音频设备列表"""
    input_devices = list_audio_devices()

    print("\n" + "=" * 80)
    print("🎤 可用的音频输入设备")
    print("=" * 80)

    for idx, info in input_devices:
        device_name = info['name']

        # 设备类型判断
        device_type = "未知"

        # 检测常见的问题设备
        if any(keyword in device_name.lower() for keyword in
               ['stereo mix', '立体声混音', 'what u hear', '您听到的', 'loopback', '回环']):
            device_type = "⚠️  系统混音设备（避免选择）"
        elif any(keyword in device_name.lower() for keyword in
                 ['microphone', '麦克风', 'mic', '输入设备']):
            device_type = "✅ 麦克风设备（推荐）"
        elif any(keyword in device_name.lower() for keyword in
                 ['microsoft 声音映射器', '主声音捕获']):
            device_type = "⚠️  系统混音设备（避免选择）"
        elif any(keyword in device_name.lower() for keyword in
                 ['headset', '耳机', 'headphone']):
            device_type = "🎧 耳机麦克风"

        # 打印设备信息
        print(f"\n设备 #{idx}: {device_name}")
        print(f"  类型: {device_type}")
        print(f"  采样率: {int(info['defaultSampleRate'])} Hz")
        print(f"  输入通道: {info['maxInputChannels']}")

    print("\n" + "=" * 80)
    return input_devices


def setup_audio_device_interactive():
    """
    交互式设置音频设备

    Returns:
        int: 用户选择的设备索引，如果用户选择跳过则返回 None
    """
    input_devices = display_audio_devices()

    # 过滤出推荐的设备
    recommended_devices = []
    for idx, info in input_devices:
        device_name = info['name'].lower()

        # 推荐麦克风设备，排除系统混音
        if any(keyword in device_name for keyword in ['microphone', '麦克风', 'mic']) and \
           not any(keyword in device_name for keyword in
                   ['stereo mix', '立体声混音', 'what u hear', '您听到的',
                    'microsoft 声音映射器', '主声音捕获']):
            recommended_devices.append((idx, info))

    # 如果有推荐的设备，显示建议
    if recommended_devices:
        print("\n💡 推荐设备（根据设备名称自动识别）：")
        for idx, info in recommended_devices[:3]:  # 只显示前 3 个
            print(f"  - 设备 #{idx}: {info['name']}")

    # 提示用户选择
    print("\n" + "=" * 80)
    print("请选择你的麦克风设备")
    print("=" * 80)
    print("\n输入设备编号（例如：1），或按 Enter 跳过使用系统默认设备：")

    try:
        user_input = input("> ").strip()

        if user_input == "":
            print("\n⚠️  已跳过，将使用系统默认设备")
            return None

        device_index = int(user_input)

        # 验证设备索引
        valid_indices = [idx for idx, _ in input_devices]
        if device_index not in valid_indices:
            print(f"\n❌ 无效的设备编号: {device_index}")
            print(f"有效范围: {min(valid_indices)} - {max(valid_indices)}")
            return None

        # 显示选择结果
        selected_device = next(info for idx, info in input_devices if idx == device_index)
        print(f"\n✅ 已选择设备 #{device_index}: {selected_device['name']}")

        return device_index

    except ValueError:
        print("\n❌ 输入无效，将使用系统默认设备")
        return None
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消，将使用系统默认设备")
        return None


def update_env_file(device_index: int):
    """
    更新 .env 文件，添加音频设备配置

    Args:
        device_index: 用户选择的设备索引
    """
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"

    # 检查 .env 文件是否存在
    if not env_file.exists():
        print(f"\n⚠️  .env 文件不存在，将从 .env.example 创建")
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print(f"✓ 已创建 .env 文件")
        else:
            print(f"❌ .env.example 也不存在，跳过配置")
            return

    # 读取现有配置
    env_content = ""
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            env_content = f.read()

    # 检查是否已存在配置
    lines = env_content.split('\n')
    updated_lines = []
    input_device_found = False

    for line in lines:
        if line.startswith('AUDIO_INPUT_DEVICE_INDEX='):
            updated_lines.append(f'AUDIO_INPUT_DEVICE_INDEX={device_index}')
            input_device_found = True
        else:
            updated_lines.append(line)

    # 如果没有找到，添加到文件末尾
    if not input_device_found:
        updated_lines.append(f'\n# 音频设备配置（自动生成）')
        updated_lines.append(f'AUDIO_INPUT_DEVICE_INDEX={device_index}')

    # 写回文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(updated_lines))

    print(f"✓ 已更新 .env 文件")
    print(f"✓ 配置已保存：AUDIO_INPUT_DEVICE_INDEX={device_index}")


def check_and_setup_audio_device():
    """
    检查音频设备配置，如果未设置则交互式配置

    Returns:
        int: 设备索引，如果未设置或用户跳过则返回 None
    """
    from .config import AUDIO_INPUT_DEVICE_INDEX

    # 检查是否已配置
    if AUDIO_INPUT_DEVICE_INDEX is not None:
        return int(AUDIO_INPUT_DEVICE_INDEX)

    # 未配置，启动交互式设置
    print("\n" + "=" * 80)
    print("🔊 音频设备配置")
    print("=" * 80)
    print("\n检测到未配置音频输入设备，现在开始配置...")
    print("(按 Ctrl+C 可跳过配置，使用系统默认设备)\n")

    try:
        device_index = setup_audio_device_interactive()

        if device_index is not None:
            update_env_file(device_index)
            print("\n✅ 配置完成！")
            print("\n💡 提示：配置已保存到 .env 文件")
            print("   下次启动将自动使用此设备")
            print("   如需更换设备，请修改 .env 文件或重新运行此配置\n")

            # 重新加载环境变量
            from dotenv import load_dotenv
            load_dotenv(override=True)

            return device_index

    except KeyboardInterrupt:
        print("\n\n⚠️  配置已取消，将使用系统默认设备\n")

    return None
