#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文智能语音助手 v2.7.0

功能特性：
  ✅ 唤醒词检测（小智、你好助手、智能助手）
  ✅ 中文语音识别（Sherpa-ONNX）
  ✅ 多 LLM 服务（Qwen/DeepSeek/OpenAI/Anthropic）
  ✅ 多 Vision 模型（Qwen-VL / Moondream）
  ✅ 语音合成（Piper TTS / DashScope WebSocket / Edge TTS）
  ✅ MCP 工具集成（Playwright 浏览器控制）
  ✅ Agent Skills 集成（Claude Code 设计）
  ✅ RNNoise 降噪（soxr 高质量重采样）

架构亮点：
  ✨ 基于 Pipecat 官方框架
  ✨ 自定义 KWS + Silero VAD + Smart Turn Detection
  ✨ RNNoise 深度降噪 + soxr 高质量重采样
  ✨ 工厂模式支持多 LLM 服务
  ✨ 官方 Function Calling API
  ✨ Agent Skills 技能系统

使用方法：
  uv run python main.py
"""

import asyncio


def main():
    """主函数"""
    print("🚀 启动中文智能语音助手 v2.7.0 - Pipecat 官方架构")
    print("✨ 使用 KWS + Silero VAD + Smart Turn Detection + Agent Skills\n")
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
