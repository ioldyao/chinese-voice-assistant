"""
测试 TTS 配置和导入
"""
import os
import sys

# 设置 UTF-8 编码输出（Windows 兼容）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from voice_assistant.config import (
    TTS_SERVICE,
    DASHSCOPE_REALTIME_MODEL,
    DASHSCOPE_REALTIME_VOICE,
    DASHSCOPE_REALTIME_MODE,
    QWEN_API_KEY,
)

print("=" * 60)
print("TTS 配置检查")
print("=" * 60)
print(f"TTS_SERVICE = {TTS_SERVICE}")
print(f"DASHSCOPE_REALTIME_MODEL = {DASHSCOPE_REALTIME_MODEL}")
print(f"DASHSCOPE_REALTIME_VOICE = {DASHSCOPE_REALTIME_VOICE}")
print(f"DASHSCOPE_REALTIME_MODE = {DASHSCOPE_REALTIME_MODE}")
print(f"QWEN_API_KEY = {QWEN_API_KEY[:10]}..." if QWEN_API_KEY else "QWEN_API_KEY = None")
print()

print("=" * 60)
print("测试延迟导入")
print("=" * 60)

# 测试 WebSocket Realtime TTS 适配器的延迟导入
if TTS_SERVICE == "dashscope_realtime":
    print("✓ 配置为使用 WebSocket Realtime TTS")
    print("  测试导入 QwenRealtimeTTSProcessor...")

    try:
        from voice_assistant.tts_realtime_adapter import QwenRealtimeTTSProcessor
        print("  ✓ 导入成功（延迟导入）")

        # 尝试创建实例（但不初始化）
        tts = QwenRealtimeTTSProcessor(
            model=DASHSCOPE_REALTIME_MODEL,
            voice=DASHSCOPE_REALTIME_VOICE,
            mode=DASHSCOPE_REALTIME_MODE,
            api_key=QWEN_API_KEY
        )
        print("  ✓ 实例创建成功（未初始化连接）")
        print("  注意：WebSocket 连接将在首次使用时建立")

    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"✓ 配置为使用标准 TTS: {TTS_SERVICE}")
    print("  不会触发 WebSocket Realtime TTS")

print()
print("=" * 60)
print("测试完成")
print("=" * 60)
