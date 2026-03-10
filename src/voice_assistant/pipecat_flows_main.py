"""Pipecat Flows 集成主程序 (v2.3.0)

这个文件展示了如何将 Pipecat Flows 与当前架构集成：
1. 保留所有现有功能（KWS、ASR、Vision、TTS、MCP）
2. 添加 FlowManager 用于复杂对话流程管理
3. 支持简单指令（快速执行）和复杂流程（多步骤引导）

架构设计：
- FlowManager 作为编排层，通过 Frame 与现有组件通信
- 使用动态流（运行时确定对话路径）
- 函数可以调用现有 MCP 工具
- 不修改 Pipeline 结构
"""

import asyncio
import os
import sys
from typing import Optional, Dict, Any, Tuple

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TranscriptionFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    InterruptionFrame,
    StartFrame,
    CancelFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import (
    OpenAIUserContextAggregator,
    OpenAIAssistantContextAggregator,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.base_transport import TransportParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

# 导入现有组件
try:
    # 尝试相对导入（作为模块运行时）
    from .config import (
        QWEN_MODEL,
        ALIYUN_APPKEY,
        MODELS_DIR,
        load_mcp_servers_config,
    )
    from .pyaudio_transport import PyAudioTransport
    from .pipecat_adapters import (
        SherpaKWSProcessor,
        SherpaASRProcessor,
        VisionProcessor,
        PiperTTSProcessor,
    )
    from .qwen_llm_service import (
        QwenLLMService,
        QwenLLMContext,
        mcp_tools_to_openai_format,
        register_mcp_functions,
    )
    from .mcp_client import MCPManager
    from .wake_word import SmartWakeWordSystem
except ImportError:
    # 尝试绝对导入（直接运行时）
    from config import (
        QWEN_MODEL,
        ALIYUN_APPKEY,
        MODELS_DIR,
    )
    from pyaudio_transport import PyAudioTransport
    from pipecat_adapters import (
        SherpaKWSProcessor,
        SherpaASRProcessor,
        VisionProcessor,
        PiperTTSProcessor,
    )
    from qwen_llm_service import (
        QwenLLMService,
        QwenLLMContext,
        mcp_tools_to_openai_format,
        register_mcp_functions,
    )
    from mcp_client import MCPManager
    from wake_word import SmartWakeWordSystem

# 导入 Pipecat Flows（需要先安装：pip install pipecat-ai-flows）
try:
    from pipecat_flows import (
        FlowManager,
        NodeConfig,
        FlowsFunctionSchema,
        FlowArgs,
        FlowResult,
        ConsolidatedFunctionResult,
    )
    FLOWS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ pipecat-ai-flows 未安装，将禁用对话流程功能")
    FLOWS_AVAILABLE = False


# ==================== 数据类型定义 ====================
# （暂时保留，future use）


# ==================== 函数处理器（对话流程管理）====================

async def end_conversation_handler(
    args: FlowArgs,
    flow_manager: FlowManager
) -> ConsolidatedFunctionResult:
    """结束对话"""
    logger.info("👋 用户请求结束对话")
    return None, None  # 不转换节点，直接结束


# ==================== 函数 Schema 定义 ====================

end_conversation_schema = FlowsFunctionSchema(
    name="end_conversation",
    description="结束当前对话",
    properties={},
    required=[],
    handler=end_conversation_handler,
)


# ==================== 节点配置创建函数 ====================

def create_initial_node(wait_for_user: bool = False) -> NodeConfig:
    """创建初始节点：询问用户想要浏览什么网站"""
    return {
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": (
                    "你是一个智能浏览助手小智。你的任务是帮助用户浏览网页并讲解内容。"
                    "使用简洁、友好的语气与用户交流。这是语音对话，避免使用特殊字符和表情符号。"
                )
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "询问用户想要浏览哪个网站或者输入具体的网址。"
                    "用户说完后，直接使用浏览器工具（如 browser_navigate）打开网页。"
                )
            }
        ],
        "functions": [end_conversation_schema],  # 只保留 Flow 管理函数
        "respond_immediately": not wait_for_user,
    }


def create_analysis_node() -> NodeConfig:
    """创建分析节点：已打开网页，询问是否需要讲解"""
    return {
        "name": "analysis",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "告诉用户已经打开了网页。询问用户是否需要你讲解页面内容，"
                    "或者用户想了解页面的哪些具体信息。"
                )
            }
        ],
        "functions": [analysis_schema, navigate_schema, end_conversation_schema],
    }


def create_completion_node() -> NodeConfig:
    """创建完成节点：讲解完成，询问是否还需要其他帮助"""
    return {
        "name": "completion",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "你已经完成了页面内容的分析和讲解。"
                    "询问用户是否需要浏览其他网站，或者对当前页面还有其他问题。"
                )
            }
        ],
        "functions": [navigate_schema, analysis_schema, end_conversation_schema],
    }


# ==================== 简单指令处理器（快速执行路径）====================

class SimpleCommandProcessor(FrameProcessor):
    """
    简单指令处理器

    检测到特定模式的简单指令时，绕过 FlowManager 直接执行，
    以获得更快的响应速度。

    示例：
    - "打开百度" → 直接调用 MCP Playwright
    - "截图" → 直接调用截图功能
    - "返回" → 直接调用浏览器返回
    """

    def __init__(self, mcp_manager: Optional[MCPManager] = None):
        super().__init__()
        self.mcp_manager = mcp_manager

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """处理帧"""
        await super().process_frame(frame, direction)

        # 只处理用户转录文本
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()

            # 检测简单指令模式
            handled = await self._handle_simple_command(text)

            if handled:
                # 如果已处理，不再传递给 LLM
                logger.info(f"✅ 简单指令已处理: {text}")
                return

        # 未处理的帧继续传递
        await self.push_frame(frame, direction)

    async def _handle_simple_command(self, text: str) -> bool:
        """
        处理简单指令

        Returns:
            bool: 是否成功处理
        """
        if not self.mcp_manager:
            return False

        # 模式匹配：打开网站
        if "打开" in text or "访问" in text:
            # 提取 URL 或网站名称
            url = self._extract_url(text)
            if url:
                try:
                    result = await self.mcp_manager.call_tool_async(
                        "playwright_navigate",
                        {"url": url}
                    )
                    if result.success:
                        logger.info(f"✅ 快速导航成功: {url}")
                        # 可以在这里推送一个成功提示帧
                        return True
                except Exception as e:
                    logger.error(f"❌ 快速导航失败: {e}")

        # 模式匹配：截图
        elif "截图" in text or "截屏" in text:
            try:
                result = await self.mcp_manager.call_tool_async(
                    "playwright_screenshot",
                    {"name": "quick_screenshot"}
                )
                if result.success:
                    logger.info(f"✅ 快速截图成功")
                    return True
            except Exception as e:
                logger.error(f"❌ 快速截图失败: {e}")

        # 模式匹配：返回
        elif "返回" in text or "后退" in text:
            try:
                # 假设有 playwright_back 工具
                result = await self.mcp_manager.call_tool_async("playwright_back", {})
                if result.success:
                    logger.info(f"✅ 快速返回成功")
                    return True
            except Exception as e:
                logger.error(f"❌ 快速返回失败: {e}")

        return False

    def _extract_url(self, text: str) -> Optional[str]:
        """从文本中提取 URL"""
        # 常见网站映射
        site_map = {
            "百度": "https://www.baidu.com",
            "淘宝": "https://www.taobao.com",
            "京东": "https://www.jd.com",
            "知乎": "https://www.zhihu.com",
            "B站": "https://www.bilibili.com",
            "哔哩哔哩": "https://www.bilibili.com",
        }

        # 检查是否包含完整 URL
        if "http://" in text or "https://" in text:
            # 简单提取（实际可以用正则）
            words = text.split()
            for word in words:
                if word.startswith("http"):
                    return word

        # 检查网站名称
        for site_name, url in site_map.items():
            if site_name in text:
                return url

        return None


# ==================== 主程序 ====================

async def main():
    """主程序入口"""

    if not FLOWS_AVAILABLE:
        logger.error("❌ pipecat-ai-flows 未安装，无法启动")
        logger.info("💡 安装方法: pip install pipecat-ai-flows")
        return

    logger.info("🚀 启动 Pipecat Flows 集成版语音助手 (v2.3.0)")

    # 1. 初始化 Wake Word 系统
    logger.info("⏳ 加载模型...")
    wake_system = SmartWakeWordSystem(enable_voice=False, enable_mcp=False)
    logger.info("✓ KWS/ASR 模型已加载")

    # 2. 初始化 MCP Manager
    logger.info("📦 启动 MCP Servers...")
    mcp_manager = MCPManager()

    # 添加 Playwright MCP Server
    # ✅ 从配置文件加载 Server 列表
    servers = load_mcp_servers_config()

    if not servers:
        logger.error("❌ 没有已启用的 MCP Server")
        return

    success_count = 0
    for server_config in servers:
        try:
            success = await mcp_manager.add_server_async(
                name=server_config["name"],
                command=server_config["command"],
                args=server_config["args"],
                timeout=server_config["timeout"]
            )
            if success:
                success_count += 1
                logger.info(f"✅ {server_config['name']} MCP Server 启动成功")
            else:
                logger.warning(f"⚠️ {server_config['name']} MCP Server 启动失败")
        except Exception as e:
            logger.error(f"❌ 启动 {server_config['name']} MCP Server 时出错: {e}")

    if success_count == 0:
        logger.error("❌ 没有 MCP Server 启动成功，无法继续")
        return

    # 3. 初始化 Transport
    logger.info("🎤 初始化音频 Transport...")

    # 创建 VAD Analyzer
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.7,      # VAD 置信度阈值
            start_secs=0.2,      # 确认开始说话的时间（快速响应）
            stop_secs=0.2,       # 快速检测停顿（Turn Detection 会判断是否完成）
            min_volume=0.6,      # 最小音量阈值
        )
    )

    # 创建 Turn Detection（智能判断对话是否完成）
    turn_analyzer = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(
            min_turn_duration_secs=1.0,      # 最短对话时长
            max_silence_secs=2.0,            # 最大停顿时间（incomplete 时）
            confidence_threshold=0.7         # 检测置信度
        )
    )

    # 创建 Transport
    transport = PyAudioTransport(
        sample_rate=16000,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=vad_analyzer,      # ✅ 启用 Silero VAD
            turn_analyzer=turn_analyzer,    # ✅ 启用 Smart Turn Detection
        )
    )

    # 启动 Transport
    await transport.start()
    logger.info("✓ 音频 Transport 已启动（VAD + Turn Detection）")

    # 4. 初始化处理器
    logger.info("⚙️ 初始化处理器...")

    # KWS (关键词唤醒) - 使用 wake_system 的模型对象
    kws_proc = SherpaKWSProcessor(wake_system.kws_model)

    # ASR (语音识别) - 使用 wake_system 的模型对象
    asr_proc = SherpaASRProcessor(wake_system.asr_model)

    # LLM
    llm = QwenLLMService(model=QWEN_MODEL)

    # 注册 MCP 函数处理器（LLM 会自动调用 MCP 工具）
    await register_mcp_functions(llm, mcp_manager)

    # ✅ 获取 MCP 工具并添加到 LLM context（让 LLM 知道有哪些工具可用）
    mcp_tools = await mcp_manager.list_all_tools_async()
    tools = mcp_tools_to_openai_format(mcp_tools)

    # ✅ 创建包含 system message 和 tools 的 context
    messages = [
        {
            "role": "system",
            "content": """你是一个智能浏览助手小智。可以使用浏览器工具帮助用户。

可用工具：
- Playwright 浏览器操作（导航、点击、输入、滚动等）

能力：
- 视觉理解：可以看到并描述屏幕内容

使用简洁、友好的语气与用户交流。这是语音对话，避免使用特殊字符和表情符号。"""
        }
    ]

    context = QwenLLMContext(messages, tools=tools)
    user_aggregator = OpenAIUserContextAggregator(context)
    assistant_aggregator = OpenAIAssistantContextAggregator(context)

    # Vision - 传入 context
    vision_proc = VisionProcessor(context=context)

    # TTS - 使用 wake_system 的 TTS 对象
    tts_proc = PiperTTSProcessor(wake_system.agent.tts)

    # 简单指令处理器（可选，用于快速执行）
    simple_cmd_proc = SimpleCommandProcessor(mcp_manager=mcp_manager)

    # 4. 构建 Pipeline
    logger.info("🔧 构建 Pipeline...")

    pipeline = Pipeline([
        transport.input(),              # PyAudio 输入 + VAD
        kws_proc,                       # 唤醒词检测
        asr_proc,                       # 语音识别
        simple_cmd_proc,                # 简单指令快速处理（可选）
        user_aggregator,                # 添加用户消息到 context
        vision_proc,                    # Vision 分析
        llm,                            # Qwen LLM + Function Calling
        tts_proc,                       # Piper TTS
        assistant_aggregator,           # 保存助手响应
        transport.output(),             # PyAudio 输出
    ])

    # 5. 创建 Task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True)
    )

    # 6. 初始化 FlowManager
    logger.info("🌊 初始化 FlowManager...")

    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=user_aggregator,  # 使用 user_aggregator
        transport=transport,
    )

    # 将 mcp_manager 保存到 FlowManager 的状态中，供函数使用
    flow_manager.state["mcp_manager"] = mcp_manager

    # 7. 启动 FlowManager（带初始节点）
    logger.info("🎬 启动 FlowManager...")

    # 注意：这里 wait_for_user=True，表示等待用户先说话
    # 如果希望机器人先打招呼，设置为 False
    initial_node = create_initial_node(wait_for_user=True)
    await flow_manager.initialize(initial_node)

    # 8. 运行 Pipeline
    logger.info("✅ 一切就绪，开始运行...")
    logger.info("💡 说 '小智' 唤醒助手")
    logger.info("💡 按 Ctrl+C 退出")

    runner_task = None

    try:
        # 发送 StartFrame 初始化
        await task.queue_frames([StartFrame()])

        # 创建 PipelineRunner 并运行
        runner = PipelineRunner()
        runner_task = asyncio.create_task(runner.run(task))
        await runner_task

    except KeyboardInterrupt:
        logger.info("\n👋 收到退出信号，正在清理...")
    except asyncio.CancelledError:
        logger.info("\n⏹️  Pipeline 已取消")
    except Exception as e:
        logger.error(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        logger.info("\n🧹 正在清理资源...")

        # 停止 Pipeline
        if task:
            try:
                logger.info("  ⏳ 正在停止 Pipeline...")
                await asyncio.wait_for(
                    task.queue_frames([CancelFrame()]),
                    timeout=2.0
                )
                logger.info("  ✓ CancelFrame 已发送")
            except asyncio.TimeoutError:
                logger.warning("  ⚠️ 发送 CancelFrame 超时")
            except Exception as e:
                logger.warning(f"  ⚠️ 发送 CancelFrame 时出错: {e}")

        # 等待 runner 任务完成
        if runner_task and not runner_task.done():
            try:
                await asyncio.wait_for(runner_task, timeout=3.0)
                logger.info("  ✓ Pipeline 已停止")
            except asyncio.TimeoutError:
                logger.warning("  ⚠️ Pipeline 停止超时，强制取消")
                runner_task.cancel()
                try:
                    await runner_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                logger.warning(f"  ⚠️ 等待 Pipeline 停止时出错: {e}")

        # 停止音频传输
        if transport:
            try:
                await asyncio.wait_for(
                    transport.stop(),
                    timeout=2.0
                )
                logger.info("  ✓ 音频传输已停止")
            except asyncio.TimeoutError:
                logger.warning("  ⚠️ 停止音频传输超时")
            except Exception as e:
                logger.warning(f"  ⚠️ 停止音频传输时出错: {e}")

        # 停止 MCP Servers
        if mcp_manager:
            try:
                await asyncio.wait_for(
                    mcp_manager.stop_all_async(),
                    timeout=3.0
                )
                logger.info("  ✓ MCP Servers 已停止")
            except asyncio.TimeoutError:
                logger.warning("  ⚠️ 停止 MCP Servers 超时")
            except Exception as e:
                logger.warning(f"  ⚠️ 停止 MCP Servers 时出错: {e}")

        logger.info("\n👋 再见！")


if __name__ == "__main__":
    asyncio.run(main())
