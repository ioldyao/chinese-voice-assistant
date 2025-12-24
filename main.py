#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能语音助手 - Pipecat 单一模式
"""


def main():
    """主函数 - 仅 Pipecat 模式"""
    print("=" * 60)
    print("智能语音助手 - Pipecat 模式")
    print("=" * 60)

    try:
        from src.voice_assistant import pipecat_main
        import asyncio
        asyncio.run(pipecat_main.main())
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
