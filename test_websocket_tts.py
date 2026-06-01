"""
WebSocket Realtime TTS 测试脚本

测试音频播放流畅度，检查是否还有电音和卡顿问题。
"""
import os
import sys
import asyncio

# 设置 UTF-8 编码输出（Windows 兼容）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from voice_assistant.tts_realtime import create_realtime_tts


async def test_websocket_tts():
    """测试 WebSocket Realtime TTS 音频播放"""
    print("=" * 70)
    print("WebSocket Realtime TTS 音频测试")
    print("=" * 70)
    print()

    # 获取配置
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("DASHSCOPE_REALTIME_MODEL", "qwen3-tts-flash-realtime")
    voice = os.getenv("DASHSCOPE_REALTIME_VOICE", "Cherry")
    mode = os.getenv("DASHSCOPE_REALTIME_MODE", "server_commit")

    print(f"📋 配置信息：")
    print(f"  - 模型: {model}")
    print(f"  - 音色: {voice}")
    print(f"  - 模式: {mode}")
    print()

    # 创建 TTS 实例
    print("⏳ 正在初始化 WebSocket 连接...")
    tts = create_realtime_tts(
        model=model,
        voice=voice,
        mode=mode,
        api_key=api_key
    )

    try:
        # 初始化连接
        await tts.initialize()
        print()
        print("✅ WebSocket 连接已建立")
        print()

        # 测试文本（包含多个句子，测试音频块拼接）
        test_texts = [
            "你好，这是一个测试。",
            "我们来检查一下音频播放是否流畅。",
            "WebSocket Realtime TTS 应该能解决电音和卡顿问题。",
            "请注意听每个句子之间的过渡是否平滑。"
        ]

        print("🔊 开始播放测试音频...")
        print("-" * 70)

        # ✅ 流式方式：将所有文本合并，一次性发送
        # WebSocket 会持续返回音频流，播放线程持续播放
        combined_text = " ".join(test_texts)
        print(f"📝 发送文本: {combined_text}")
        await tts.speak(combined_text)

        print("-" * 70)
        print()
        print("⏳ 等待音频播放完成...")
        # ✅ 等待最后一句话的音频播放完
        await asyncio.sleep(10)
        print()
        print("✅ 测试完成！")
        print()
        print("📊 请评估音频质量：")
        print("  - 是否还有电音（咔哒声）？")
        print("  - 句子之间的过渡是否流畅？")
        print("  - 是否还有一段一段的感觉？")
        print()
        print("💡 如果仍有问题，可能需要：")
        print("  1. 检查网络连接质量")
        print("  2. 调整音频缓冲参数")
        print("  3. 添加音频平滑处理器")

    except Exception as e:
        print()
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print()
        print("⏳ 正在关闭连接...")
        await tts.close()
        print("✅ 连接已关闭")


if __name__ == "__main__":
    try:
        asyncio.run(test_websocket_tts())
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
