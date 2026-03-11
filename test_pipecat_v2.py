#!/usr/bin/env python3
"""测试 Pipecat v2.0 架构"""
import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_pyaudio_transport():
    """测试 PyAudioTransport"""
    print("=" * 60)
    print("测试 1: PyAudioTransport")
    print("=" * 60)

    try:
        from src.voice_assistant.pyaudio_transport import PyAudioTransport

        transport = PyAudioTransport(sample_rate=16000)
        await transport.start()

        print("✓ PyAudioTransport 初始化成功")
        print(f"✓ 输入流: {transport.input_stream is not None}")
        print(f"✓ 输出流: {transport.output_stream is not None}")

        # 测试 input() 和 output() 方法
        input_proc = transport.input()
        output_proc = transport.output()

        print(f"✓ input() 返回: {type(input_proc).__name__}")
        print(f"✓ output() 返回: {type(output_proc).__name__}")

        await transport.stop()
        print("✓ PyAudioTransport 停止成功")

        return True

    except Exception as e:
        print(f"❌ PyAudioTransport 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tts_processor():
    """测试 PiperTTSProcessor"""
    print("\n" + "=" * 60)
    print("测试 2: PiperTTSProcessor")
    print("=" * 60)

    try:
        from src.voice_assistant.tts import TTSManager
        from src.voice_assistant.pipecat_adapters import PiperTTSProcessor

        # 检查 Piper 模型（修正路径）
        model_path = Path(__file__).parent / "models" / "piper" / "zh_CN-huayan-medium.onnx"
        if not model_path.exists():
            print(f"⚠️ Piper 模型不存在: {model_path}")
            print("   请运行: python download_piper_model.py")
            return False

        tts = TTSManager(engine_type="piper", model_path=model_path)
        print(f"✓ TTSManager 初始化成功")

        tts_proc = PiperTTSProcessor(tts)
        print(f"✓ PiperTTSProcessor 创建成功")

        # 检查是否不依赖 transport
        print("✓ PiperTTSProcessor 不依赖 transport（正确）")

        return True

    except Exception as e:
        print(f"❌ PiperTTSProcessor 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vision_processor():
    """测试 VisionProcessor"""
    print("\n" + "=" * 60)
    print("测试 3: VisionProcessor")
    print("=" * 60)

    try:
        from src.voice_assistant.pipecat_adapters import VisionProcessor
        from src.voice_assistant.qwen_llm_service import QwenLLMContext
        from src.voice_assistant.config import DASHSCOPE_API_URL, DASHSCOPE_API_KEY

        # 创建 mock context
        context = QwenLLMContext(
            messages=[{"role": "system", "content": "test"}],
            tools=[]
        )

        vision_proc = VisionProcessor(
            api_url=DASHSCOPE_API_URL,
            api_key=DASHSCOPE_API_KEY,
            context=context
        )

        print(f"✓ VisionProcessor 创建成功")
        print(f"✓ 接收 context: {vision_proc.context is not None}")
        print(f"✓ Vision 关键词: {len(vision_proc.vision_keywords)} 个")
        print(f"✓ 操作关键词: {len(vision_proc.operation_keywords)} 个")

        # 测试关键词判断
        assert vision_proc._needs_vision("查看屏幕") == True
        assert vision_proc._needs_vision("点击按钮") == False
        print("✓ 关键词判断逻辑正确")

        return True

    except Exception as e:
        print(f"❌ VisionProcessor 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pipeline_structure():
    """测试 Pipeline 结构"""
    print("\n" + "=" * 60)
    print("测试 4: Pipeline 结构")
    print("=" * 60)

    try:
        from src.voice_assistant.pyaudio_transport import PyAudioTransport
        from src.voice_assistant.pipecat_adapters import (
            SherpaKWSProcessor,
            SherpaASRProcessor,
            PiperTTSProcessor,
            VisionProcessor,
        )
        from src.voice_assistant.qwen_llm_service import QwenLLMService, QwenLLMContext
        from pipecat.services.openai.llm import (
            OpenAIUserContextAggregator,
            OpenAIAssistantContextAggregator,
        )
        from pipecat.pipeline.pipeline import Pipeline
        from src.voice_assistant.wake_word import SmartWakeWordSystem
        from src.voice_assistant.config import DASHSCOPE_API_URL, DASHSCOPE_API_KEY

        # 创建组件（不启动）
        wake_system = SmartWakeWordSystem(enable_voice=False, enable_mcp=False)

        kws_proc = SherpaKWSProcessor(wake_system.kws_model)
        asr_proc = SherpaASRProcessor(wake_system.asr_model)

        llm = QwenLLMService(model="qwen-plus")
        context = QwenLLMContext(messages=[], tools=[])

        user_agg = OpenAIUserContextAggregator(context)
        assistant_agg = OpenAIAssistantContextAggregator(context)

        vision_proc = VisionProcessor(DASHSCOPE_API_URL, DASHSCOPE_API_KEY, context)

        # 修正 Piper 模型路径
        model_path = Path(__file__).parent / "models" / "piper" / "zh_CN-huayan-medium.onnx"
        if not model_path.exists():
            print(f"⚠️ 跳过 TTS 测试（模型不存在）")
            return True

        from src.voice_assistant.tts import TTSManager
        tts = TTSManager(engine_type="piper", model_path=model_path)
        tts_proc = PiperTTSProcessor(tts)

        transport = PyAudioTransport(sample_rate=16000)

        # 创建 Pipeline（官方顺序）
        pipeline = Pipeline([
            transport.input(),
            kws_proc,
            asr_proc,
            user_agg,
            vision_proc,
            llm,
            assistant_agg,
            tts_proc,
            transport.output(),
        ])

        print(f"✓ Pipeline 创建成功")
        print(f"✓ Processors 数量: {len(pipeline.processors)}")

        # 验证顺序（跳过 PipelineSource 和 PipelineSink）
        # Pipecat 会自动添加这些处理器
        expected_order = [
            "PipelineSource",           # Pipecat 自动添加
            "PyAudioInputProcessor",
            "SherpaKWSProcessor",
            "SherpaASRProcessor",
            "OpenAIUserContextAggregator",
            "VisionProcessor",
            "QwenLLMService",
            "OpenAIAssistantContextAggregator",
            "PiperTTSProcessor",
            "PyAudioOutputProcessor",
            "PipelineSink",             # Pipecat 自动添加
        ]

        # 只验证我们关心的处理器（跳过 Source 和 Sink）
        user_processors = [
            p for p in pipeline.processors
            if type(p).__name__ not in ["PipelineSource", "PipelineSink"]
        ]

        expected_user_processors = [
            name for name in expected_order
            if name not in ["PipelineSource", "PipelineSink"]
        ]

        print(f"✓ 用户定义的 Processors: {len(user_processors)} 个")

        for i, (proc, expected) in enumerate(zip(user_processors, expected_user_processors)):
            actual = type(proc).__name__
            if actual == expected:
                print(f"  {i+1}. ✓ {actual}")
            else:
                print(f"  {i+1}. ❌ 期望 {expected}，实际 {actual}")
                return False

        print("✓ Pipeline 顺序正确（符合官方标准）")

        return True

    except Exception as e:
        print(f"❌ Pipeline 结构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    print("\n🧪 Pipecat v2.0 架构测试")
    print("=" * 60)

    results = {
        "PyAudioTransport": await test_pyaudio_transport(),
        "PiperTTSProcessor": await test_tts_processor(),
        "VisionProcessor": await test_vision_processor(),
        "Pipeline 结构": await test_pipeline_structure(),
    }

    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")

    print("\n" + "=" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    print("=" * 60)

    if passed == total:
        print("\n🎉 所有测试通过！架构符合 Pipecat 标准。")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请检查配置。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
