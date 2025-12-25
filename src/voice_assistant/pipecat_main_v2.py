"""Pipecat 主程序 - 符合 Pipecat 官方架构（v2.0）"""
import asyncio
import signal
import sys
from pathlib import Path

# ✅ 在导入 Pipecat 之前配置日志
import os
os.environ["LOGURU_LEVEL"] = "ERROR"  # 只显示 ERROR 级别（隐藏 WARNING）

from loguru import logger

# ✅ 配置自己的日志输出
logger.remove()  # 移除默认 handler
logger.add(
    sys.stderr,
    level="INFO",  # 我们自己的日志显示 INFO
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
    filter=lambda record: not record["name"].startswith("pipecat")  # 过滤掉 pipecat 的日志
)

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import StartFrame, CancelFrame

# ✅ 导入 VAD 相关模块（Pipecat 官方）
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.base_transport import TransportParams

# 导入标准 PyAudio Transport（v2.2 - 支持 VAD）
from .pyaudio_transport import PyAudioTransport

# 导入适配器（已修复）
from .pipecat_adapters import (
    SherpaKWSProcessor,
    SherpaASRProcessor,
    PiperTTSProcessor,
    VisionProcessor,
)

# 导入 Qwen LLM Service（官方框架）
from .qwen_llm_service import (
    QwenLLMService,
    QwenLLMContext,
    mcp_tools_to_openai_format,
    register_mcp_functions,
)

# 导入官方 Context Aggregator
from pipecat.services.openai.llm import (
    OpenAIUserContextAggregator,
    OpenAIAssistantContextAggregator,
)

# 导入现有组件
from .wake_word import SmartWakeWordSystem
from .config import MODELS_DIR


async def create_pipecat_pipeline():
    """
    创建 Pipecat Pipeline - 符合官方架构（v2.0）

    官方推荐结构：
    transport.input() → processors → LLM → TTS → transport.output()

    改进：
    ✅ 使用标准 PyAudioTransport（符合 BaseTransport 接口）
    ✅ Pipeline 顺序：input → KWS → ASR → user_agg → Vision → LLM → assistant_agg → TTS → output
    ✅ TTS 生成标准 OutputAudioRawFrame（不直接播放）
    ✅ Vision 直接修改 context（不推送新 Frame）
    """
    print("\n" + "="*60)
    print("🎙️  中文语音助手 v2.2.1 - Pipecat 官方架构")
    print("="*60)

    # 1. 初始化现有组件
    print("\n⏳ 加载模型...")

    # 创建 wake_word 系统（跳过 MCP 初始化）
    wake_system = SmartWakeWordSystem(enable_voice=False, enable_mcp=False)
    print("✓ KWS/ASR 模型已加载")

    # 手动异步启动 MCP Servers
    print("⏳ 启动 MCP Servers...")

    from .mcp_client import MCPManager
    mcp = MCPManager()

    servers = [
        ("playwright", "npx", ["@playwright/mcp@latest"], 120)
    ]

    success_count = 0
    for name, command, args, timeout in servers:
        try:
            success = await mcp.add_server_async(name, command, args, timeout)
            if success:
                success_count += 1
        except Exception as e:
            print(f"  ❌ {name} Server 启动失败: {e}")
            continue

    if success_count > 0:
        mcp_tools = await mcp.list_all_tools_async()
        playwright_tools = [t for t in mcp_tools if t.get("server") == "playwright"]
        print(f"✓ Playwright MCP 已启动（{len(playwright_tools)} 个工具）")
    else:
        print(f"❌ MCP Server 启动失败")
        raise RuntimeError("MCP Server 启动失败")

    # 2. 初始化 Qwen LLM Service
    print("⏳ 初始化 LLM Service...")

    llm = QwenLLMService(model="qwen-plus")

    # 注册 MCP 函数处理器
    await register_mcp_functions(llm, mcp)

    # 创建 Tools（OpenAI API 格式）
    tools = mcp_tools_to_openai_format(mcp_tools)

    # 创建对话上下文
    messages = [
        {
            "role": "system",
            "content": """你是一个智能语音助手，可以使用浏览器工具和视觉理解能力帮助用户。

可用工具：
- Playwright 浏览器操作（导航、点击、输入、滚动等）

能力：
- 视觉理解：可以看到并描述屏幕内容

重要规则：
1. **操作场景**（用户要求"打开"、"点击"、"输入"等）：
   - **点击元素前必须先调用 browser_snapshot 获取最新页面快照**
   - 使用快照中的 ref 编号进行点击操作
   - 如果点击失败（ref not found），立即重新调用 browser_snapshot 获取新快照
   - 工具调用成功后，用简短的中文确认（如"好的，已经点击"）
   - 不要重复调用同一个工具

2. **视觉理解场景**（用户要求"查看"、"看"、"描述"等）：
   - 如果收到 `[视觉观察]` 开头的消息，说明系统已经完成截图和视觉分析
   - 用自然、简洁的语言向用户描述屏幕内容
   - 突出关键信息，准确描述画面内容

3. **执行流程示例**：
   用户："点击动态按钮"
   → 步骤1：调用 browser_snapshot（获取页面元素和ref）
   → 步骤2：调用 browser_click（使用快照中的ref点击）
   → 步骤3：回复"好的，已经点击"

4. 不要主动询问用户是否需要其他帮助"""
        }
    ]

    context = QwenLLMContext(messages, tools=tools)

    # 创建 Context Aggregators
    user_aggregator = OpenAIUserContextAggregator(context)
    assistant_aggregator = OpenAIAssistantContextAggregator(context)

    print(f"✓ LLM Service 已就绪（{len(tools)} 个工具）")

    # 3. 创建 Pipecat Processors
    print("⏳ 创建 Processors...")

    kws_proc = SherpaKWSProcessor(wake_system.kws_model)
    asr_proc = SherpaASRProcessor(wake_system.asr_model)

    # Vision Processor（传入 context）
    vision_proc = VisionProcessor(
        api_url=wake_system.agent.api_url,
        api_key=wake_system.agent.api_key,
        context=context  # ✅ 传入 context
    )

    # TTS Processor（不传入 transport）
    tts_proc = PiperTTSProcessor(wake_system.agent.tts)

    print("✓ Processors 已创建")

    # 4. ✅ 配置 Silero VAD（Pipecat 官方 VAD）
    print("⏳ 配置 VAD + Turn Detection...")

    # ✅ 使用 Pipecat 官方 Silero VAD + Smart Turn Detection
    # 根据官方文档：配合 Turn Detection 时使用 stop_secs=0.2
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.7,      # VAD 置信度阈值
            start_secs=0.2,      # 确认开始说话的时间（快速响应）
            stop_secs=0.2,       # 快速检测停顿（Turn Detection 会判断是否完成）
            min_volume=0.6,      # 最小音量阈值
        )
    )

    # ✅ 添加 Smart Turn Detection（智能判断对话是否完成）
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

    turn_analyzer = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(
            min_turn_duration_secs=1.0,      # 最短对话时长
            max_silence_secs=2.0,            # 最大停顿时间（incomplete 时）
            confidence_threshold=0.7         # 检测置信度
        )
    )

    print("✓ VAD + Turn Detection 已配置")

    # 5. ✅ 创建标准 PyAudioTransport（配置 VAD）
    print("⏳ 创建 PyAudio Transport...")

    transport = PyAudioTransport(
        sample_rate=16000,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=vad_analyzer,      # ✅ 启用 Silero VAD
            turn_analyzer=turn_analyzer,    # ✅ 启用 Smart Turn Detection
        )
    )
    await transport.start()

    print("✓ Transport 已启动")

    # 6. ✅ 构建 Pipeline（官方标准顺序 + TTS 调整）
    print("⏳ 构建 Pipeline...")

    pipeline = Pipeline([
        transport.input(),              # 1. ✅ 官方音频输入（内置 VAD 处理）
        kws_proc,                       # 2. KWS 唤醒词检测
        asr_proc,                       # 3. ASR 识别（响应 VAD frames）
        user_aggregator,                # 4. ✅ 添加用户消息到 context（紧跟 ASR）
        vision_proc,                    # 5. ✅ Vision（直接修改 context）
        llm,                            # 6. ✅ LLM 生成（已注册 MCP 函数）
        tts_proc,                       # 7. ✅ TTS 合成（在 aggregator 之前，接收 LLMTextFrame）
        assistant_aggregator,           # 8. ✅ 保存助手响应（收集 LLMTextFrame 到 context）
        transport.output(),             # 9. ✅ 官方音频输出
    ])

    print("✓ Pipeline 已构建")
    print("\n" + "="*60)
    print("✅ 启动完成！")
    print("="*60)
    print("\n💬 说出唤醒词开始对话...")
    print("   唤醒词: 小智、你好助手、智能助手")
    print("   按 Ctrl+C 退出")
    print("")

    return pipeline, transport, wake_system, mcp


async def main():
    """Pipecat 主程序 - v2.0（官方架构）"""
    pipeline = None
    transport = None
    wake_system = None
    mcp = None
    task = None
    runner_task = None

    try:
        # 创建 Pipeline
        pipeline, transport, wake_system, mcp = await create_pipecat_pipeline()

        # 创建 PipelineTask
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,  # 启用官方中断机制
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
            )
        )

        # 发送 StartFrame 初始化
        await task.queue_frames([StartFrame()])

        # 创建 PipelineRunner 并运行
        runner = PipelineRunner()

        # ✅ 运行 Pipeline（官方方式）
        # 创建后台任务，以便可以响应 Ctrl+C
        runner_task = asyncio.create_task(runner.run(task))
        await runner_task

    except KeyboardInterrupt:
        print("\n⏹️  收到退出信号...")
    except asyncio.CancelledError:
        print("\n⏹️  Pipeline 已取消")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        print("\n🧹 正在清理资源...")

        # 0. ✅ 停止 Pipeline（发送 CancelFrame）
        if task:
            try:
                print("  ⏳ 正在停止 Pipeline...")
                await asyncio.wait_for(
                    task.queue_frames([CancelFrame()]),
                    timeout=2.0
                )
                print("  ✓ CancelFrame 已发送")
            except asyncio.TimeoutError:
                print("  ⚠️ 发送 CancelFrame 超时")
            except Exception as e:
                print(f"  ⚠️ 发送 CancelFrame 时出错: {e}")

        # 等待 runner 任务完成
        if runner_task and not runner_task.done():
            try:
                await asyncio.wait_for(runner_task, timeout=3.0)
                print("  ✓ Pipeline 已停止")
            except asyncio.TimeoutError:
                print("  ⚠️ Pipeline 停止超时，强制取消")
                runner_task.cancel()
                try:
                    await runner_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                print(f"  ⚠️ 等待 Pipeline 停止时出错: {e}")

        # 1. 停止音频传输
        if transport:
            try:
                await asyncio.wait_for(
                    transport.stop(),
                    timeout=2.0
                )
                print("  ✓ 音频传输已停止")
            except asyncio.TimeoutError:
                print("  ⚠️ 停止音频传输超时")
            except Exception as e:
                print(f"  ⚠️ 停止音频传输时出错: {e}")

        # 2. 停止 MCP Servers
        if mcp:
            try:
                await asyncio.wait_for(
                    mcp.stop_all_async(),
                    timeout=3.0
                )
                print("  ✓ MCP Servers 已停止")
            except asyncio.TimeoutError:
                print("  ⚠️ 停止 MCP Servers 超时")
            except Exception as e:
                print(f"  ⚠️ 停止 MCP Servers 时出错: {e}")

        print("\n👋 再见！")


if __name__ == "__main__":
    asyncio.run(main())
