"""Pipecat 主程序 - 基于 Pipeline 架构的语音助手"""
import asyncio
import signal
import sys
from pathlib import Path

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.base_transport import TransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

# 导入适配器
from .pipecat_adapters import (
    SherpaKWSProcessor,
    SherpaASRProcessor,
    PiperTTSProcessor,
    ScreenshotProcessor,
    QwenVisionProcessor,
)

# 导入 Qwen LLM Service（官方框架）
from .qwen_llm_service import (
    QwenLLMService,
    QwenLLMContext,
    mcp_tools_to_openai_format,
    register_mcp_functions,
)

# 导入官方 Context Aggregator（使用 OpenAI 特定实现）
from pipecat.services.openai.llm import (
    OpenAIUserContextAggregator,      # ✅ 支持函数调用处理
    OpenAIAssistantContextAggregator, # ✅ 自动保存 tool_calls + 结果到 context
)

# 导入现有组件
from .wake_word import SmartWakeWordSystem
from .config import MODELS_DIR


class SimplePyAudioTransport:
    """
    简化的 PyAudio Transport

    在 Phase 1 中，我们使用简化的音频传输实现
    直接使用 PyAudio 进行音频 I/O
    """

    def __init__(self, sample_rate=16000, channels=1):
        import pyaudio
        import numpy as np

        self.pyaudio = pyaudio  # 保存模块引用
        self.np = np
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 512

        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

        self.running = False
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()

    async def start(self):
        """启动音频传输"""
        self.running = True

        # 启动输入流
        self.input_stream = self.p.open(
            format=self.pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=None
        )

        # 启动输出流
        self.output_stream = self.p.open(
            format=self.pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )

        print("✓ 音频传输已启动")

    async def stop(self):
        """停止音频传输"""
        self.running = False

        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()

        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()

        self.p.terminate()
        print("✓ 音频传输已停止")

    async def read_audio_frames(self):
        """读取音频帧（生成器）"""
        from pipecat.frames.frames import AudioRawFrame

        while self.running:
            try:
                # 从麦克风读取音频
                audio_bytes = await asyncio.to_thread(
                    self.input_stream.read,
                    self.chunk_size,
                    exception_on_overflow=False
                )

                # 创建音频帧
                frame = AudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self.sample_rate,
                    num_channels=self.channels
                )

                yield frame

            except Exception as e:
                print(f"❌ 音频读取错误: {e}")
                break

    async def write_audio_frame(self, frame):
        """写入音频帧到扬声器"""
        from pipecat.frames.frames import TTSAudioRawFrame

        if isinstance(frame, TTSAudioRawFrame) and self.output_stream:
            try:
                await asyncio.to_thread(
                    self.output_stream.write,
                    frame.audio
                )
            except Exception as e:
                print(f"❌ 音频播放错误: {e}")


async def create_pipecat_pipeline():
    """
    创建 Pipecat Pipeline - 混合架构版

    保留自定义（官方不支持）：
    - KWS 唤醒词检测（Sherpa-ONNX）
    - ASR 本地识别（Sherpa-ONNX）
    - Piper TTS（本地、免费）
    - Qwen Vision（保持现有 API）

    改用官方（享受官方生态）：
    - QwenLLMService（继承 OpenAILLMService）
    - LLMContextAggregatorPair（自动管理对话历史）
    - Function Calling（MCP 工具无缝集成）

    Pipeline 结构：
    麦克风 → KWS → ASR → context.user() → Screenshot → Vision → LLM → context.assistant() → TTS → 扬声器
                                ↓                                 ↓
                         添加用户消息                       保存助手响应
    """
    print("\n" + "="*60)
    print("🚀 Pipecat 模式 - 混合架构版 - 初始化中...")
    print("="*60)

    # 1. 初始化现有组件
    print("\n⏳ 正在加载模型...")

    # 创建 wake_word 系统（跳过 MCP 初始化，避免事件循环冲突）
    wake_system = SmartWakeWordSystem(enable_voice=False, enable_mcp=False)

    # 手动异步启动 MCP Servers
    print("\n⏳ 正在启动 MCP Servers（异步模式）...")

    # 创建独立的 MCP Manager（不使用 wake_system.agent.mcp）
    from .mcp_client import MCPManager
    mcp = MCPManager()

    servers = [
        # Playwright-MCP: 浏览器操作（主要使用）
        ("playwright", "npx", ["@playwright/mcp@latest"], 120)
    ]

    success_count = 0
    for name, command, args, timeout in servers:
        try:
            success = await mcp.add_server_async(name, command, args, timeout)
            if success:
                success_count += 1
                print(f"  ✓ {name} MCP Server 启动成功")
        except Exception as e:
            print(f"  ❌ {name} Server 启动异常: {e}")
            continue

    if success_count > 0:
        print(f"\n✅ 成功启动 {success_count}/{len(servers)} 个 MCP Server\n")

        # 获取工具列表（使用异步方法）
        mcp_tools = await mcp.list_all_tools_async()
        playwright_tools = [
            tool for tool in mcp_tools
            if tool.get("server") == "playwright"
        ]
        if playwright_tools:
            print(f"  ✓ Playwright-MCP: {len(playwright_tools)} 个工具")

            # 重点打印 browser_snapshot 和 browser_click
            for tool in playwright_tools:
                if 'snapshot' in tool['name'] or 'click' in tool['name']:
                    print(f"\n    📌 {tool['name']}:")
                    print(f"       描述: {tool.get('description', 'N/A')}")
                    print(f"       参数: {list(tool.get('input_schema', {}).get('properties', {}).keys())}")
    else:
        print(f"\n❌ 所有 MCP Server 启动失败\n")
        raise RuntimeError("MCP Server 启动失败")

    # 2. 初始化 Qwen LLM Service（官方框架）
    print("\n⏳ 正在初始化 Qwen LLM Service（官方框架）...")

    llm = QwenLLMService(model="qwen-plus")

    # 注册 MCP 函数处理器
    await register_mcp_functions(llm, mcp)

    # 创建 Tools（OpenAI API 格式，用于 LLM Context）
    tools = mcp_tools_to_openai_format(mcp_tools)

    print(f"\n🔧 转换为 OpenAI 格式后: {len(tools)} 个工具")
    for tool in tools[:5]:  # 只打印前5个
        print(f"  - {tool['function']['name']}: {tool['function']['description'][:60]}...")

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

    print(f"\n📋 LLMContext 中的 tools: {len(context.tools) if context.tools else 0} 个")
    if context.tools:
        for tool in context.tools[:3]:  # 打印前3个
            print(f"  - {tool['function']['name']}")

    # 创建 User Context Aggregator（添加用户消息到上下文）
    user_aggregator = OpenAIUserContextAggregator(context)

    # ✅ 创建 Assistant Context Aggregator（保存工具调用历史）
    assistant_aggregator = OpenAIAssistantContextAggregator(context)

    print("✓ QwenLLMService 已初始化")
    print("✓ MCP 函数已注册")
    print("✓ OpenAIUserContextAggregator 已创建")
    print("✓ OpenAIAssistantContextAggregator 已创建")

    # 3. 创建 Pipecat Processors
    print("\n⏳ 正在创建 Pipecat Processors...")

    kws_proc = SherpaKWSProcessor(wake_system.kws_model)
    asr_proc = SherpaASRProcessor(wake_system.asr_model)

    # Vision Processors（采用 Pipecat 官方模式）
    screenshot_proc = ScreenshotProcessor()  # 截图 → UserImageRawFrame
    qwen_vision_proc = QwenVisionProcessor(
        api_url=wake_system.agent.api_url,
        api_key=wake_system.agent.api_key
    )  # 处理 UserImageRawFrame → TextFrame

    # 创建音频传输（在创建 TTS Processor 之前）
    print("\n⏳ 正在创建音频传输...")
    transport = SimplePyAudioTransport(sample_rate=16000)
    await transport.start()

    # 创建 TTS Processor（传入 transport 用于音频输出）
    tts_proc = PiperTTSProcessor(wake_system.agent.tts, transport)

    print("✓ KWS Processor 已创建（自定义）")
    print("✓ ASR Processor 已创建（自定义）")
    print("✓ Screenshot Processor 已创建（Pipecat 官方模式）")
    print("✓ Qwen Vision Processor 已创建（Pipecat 官方模式）")
    print("✓ TTS Processor 已创建（自定义：Piper）")

    # 4. 构建 Pipeline（混合架构）
    print("\n⏳ 正在构建 Pipeline（混合架构）...")

    pipeline = Pipeline([
        kws_proc,                       # 自定义：KWS 唤醒词检测
        asr_proc,                       # 自定义：ASR 本地识别
        screenshot_proc,                # ✅ 在 user_aggregator 之前判断 Vision
        qwen_vision_proc,               # ✅ 处理 Vision 请求
        user_aggregator,                # 官方：添加用户消息到上下文 ✨
        llm,                            # 官方：Qwen LLM Service（已注册 MCP 函数）✨
        tts_proc,                       # ✅ 先处理 TTS（在 assistant_aggregator 之前）
        assistant_aggregator,           # ✅ 再保存到 context（工具调用历史）
    ])

    print("✓ Pipeline 已构建")
    print("\n" + "="*60)
    print("✓ Pipecat 混合架构启动完成！")
    print("="*60)
    print("\n📋 Pipeline 结构（混合架构）:")
    print("   自定义：KWS → ASR → Screenshot → Vision ✨")
    print("   官方：  context.user() ✨")
    print("   官方：  LLM Service + Function Calling ✨")
    print("   自定义：Piper TTS（先播放）")
    print("   官方：  context.assistant()（再保存历史）✨")
    print("\n💡 技术亮点:")
    print("   ✅ LLM Service 自动管理对话历史")
    print("   ✅ MCP 工具通过 Function Calling 无缝集成")
    print("   ✅ 保留本地 KWS + ASR + TTS（免费、无网络依赖）")
    print("   ✅ Assistant Aggregator 保存工具调用历史")
    print("   ✅ TTS 在 aggregator 之前处理，保证语音输出")
    print("\n💬 说出唤醒词开始对话...")
    print("   默认唤醒词: 小智、你好助手、智能助手")
    print("   按 Ctrl+C 退出\n")

    return pipeline, transport, wake_system, mcp


async def run_pipeline_with_audio(pipeline, transport):
    """
    运行 Pipeline 并处理音频 I/O

    配置官方中断支持：
    - 使用 PipelineParams 启用 allow_interruptions
    - 音频输入通过 queue_frames() 推送到 Pipeline
    """
    from pipecat.frames.frames import StartFrame, EndFrame

    try:
        # ✅ 创建 PipelineTask，配置官方中断支持
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

        # 创建音频输入任务
        async def audio_input_loop():
            """持续读取音频并推送到 Pipeline"""
            async for audio_frame in transport.read_audio_frames():
                await task.queue_frames([audio_frame])

        # 创建 PipelineRunner 并运行
        runner = PipelineRunner()

        # 创建音频输入任务（作为后台任务）
        audio_task = asyncio.create_task(audio_input_loop())

        try:
            # 运行 Pipeline（主任务）
            await runner.run(task)
        finally:
            # Pipeline 结束时，取消音频输入任务
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass

    except asyncio.CancelledError:
        # 发送 EndFrame 结束
        try:
            await task.queue_frames([EndFrame()])
        except:
            pass
        print("\n⏹️  Pipeline 已停止")
        raise  # 重新抛出异常，让 main() 处理
    except Exception as e:
        print(f"\n❌ Pipeline 运行错误: {e}")
        import traceback
        traceback.print_exc()
        raise  # 重新抛出异常


async def main():
    """Pipecat 主程序 - 混合架构版"""
    pipeline = None
    transport = None
    wake_system = None
    mcp = None

    try:
        # 创建 Pipeline（混合架构）
        pipeline, transport, wake_system, mcp = await create_pipecat_pipeline()

        # 运行 Pipeline（让 Pipecat 处理 Ctrl+C）
        await run_pipeline_with_audio(pipeline, transport)

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

        # 1. 停止音频传输
        if transport:
            try:
                await transport.stop()
                print("  ✓ 音频传输已停止")
            except Exception as e:
                print(f"  ⚠️ 停止音频传输时出错: {e}")

        # 2. 停止 MCP Servers
        if mcp:
            try:
                await mcp.stop_all_async()
                print("  ✓ MCP Servers 已停止")
            except Exception as e:
                print(f"  ⚠️ 停止 MCP Servers 时出错: {e}")

        print("\n👋 再见！")


if __name__ == "__main__":
    asyncio.run(main())
