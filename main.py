#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文智能语音助手 v2.3.2

功能特性：
  ✅ 唤醒词检测（小智）
  ✅ 中文语音识别（Sherpa-ONNX）
  ✅ 多 Vision 模型（Qwen-VL / Moondream）
  ✅ 语音合成（Piper TTS）
  ✅ MCP 工具集成（Playwright 浏览器控制）

使用方法：
  uv run python main.py
"""

import asyncio


def main():
    """主函数"""
    print("🚀 启动中文智能语音助手 v2.3.2\n")
    from src.voice_assistant import pipecat_main_v2
    asyncio.run(pipecat_main_v2.main())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已退出")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
