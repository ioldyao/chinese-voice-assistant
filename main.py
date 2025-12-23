#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能语音助手 - 主程序入口
"""


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 智能语音助手 - 双阶段识别版")
    print("=" * 60)

    print("\n请选择运行模式：")
    print("  1. 传统模式 (原有架构)")
    print("  2. Pipecat 模式 (新架构，实验性)")
    mode_choice = input("请选择 (1/2，默认1): ").strip() or "1"

    if mode_choice == "2":
        # ==================== Pipecat 模式 ====================
        print("\n✨ 启动 Pipecat 模式...")
        try:
            from src.voice_assistant import pipecat_main
            import asyncio
            asyncio.run(pipecat_main.main())
        except ImportError as e:
            print(f"❌ Pipecat 模式不可用: {e}")
            print("💡 请先安装依赖: pip install -e .")
        except Exception as e:
            print(f"❌ Pipecat 模式启动失败: {e}")
            import traceback
            traceback.print_exc()

    else:
        # ==================== 传统模式 ====================
        print("\n✨ 启动传统模式...")
        print("功能: KWS关键词唤醒 + ASR语音识别 + 视觉理解 + 系统控制")

        print("\n是否开启语音播报？")
        print("  1. 是（推荐）")
        print("  2. 否")
        choice = input("请选择 (1/2，默认1): ").strip() or "1"
        enable_voice = (choice == "1")

        try:
            from src.voice_assistant import SmartWakeWordSystem
            system = SmartWakeWordSystem(enable_voice=enable_voice)
            system.start_listening()
        except KeyboardInterrupt:
            print("\n👋 再见！")
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
