"""
Anthropic Messages API LLM Service - 纯 Anthropic API 实现（包装器链模式）

使用 Pipecat 的 Adapter 模式来处理不同的 LLM API。

架构：
- 继承 LLMService（Pipecat 基类）
- 使用 AnthropicLLMAdapter 进行格式转换
- 与 Context Aggregator 配合工作
- 调用纯 Anthropic Messages API
"""
import asyncio
from typing import Optional, AsyncGenerator

from anthropic import AsyncAnthropic
from loguru import logger

from pipecat.services.llm_service import LLMService
from pipecat.frames.frames import (
    Frame,
    LLMTextFrame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext

from .anthropic_adapter import AnthropicLLMAdapter


class AnthropicLLMService(LLMService):
    """
    Anthropic Messages API LLM Service - 包装器链模式

    架构说明：
    1. 继承 LLMService（Pipecat 基类）
    2. 使用 AnthropicLLMAdapter 进行格式转换
    3. 接收 LLMContextFrame（从 Context Aggregator）
    4. 调用纯 Anthropic Messages API
    5. 推送 LLMTextFrame 到下游

    Pipeline 位置：
    transport.input() → KWS → ASR → user_aggregator → Vision → LLM → TTS → assistant_aggregator → output

    与 OpenAI 的区别：
    - OpenAI: 继承 BaseOpenAILLMService（使用 AsyncOpenAI 客户端）
    - Anthropic: 继承 LLMService（使用 AsyncAnthropic 客户端）
    - 两者都使用 Context Aggregator 架构
    """

    # 设置 Anthropic 适配器
    adapter_class = AnthropicLLMAdapter

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        thinking_effort: str = "medium",
        **kwargs
    ):
        """
        初始化 Anthropic LLM Service

        Args:
            api_key: Anthropic API Key
            model: 模型名称
                - claude-opus-4-6（最强推理）
                - claude-sonnet-4-5-20250929（推荐）
                - claude-sonnet-4-6（最新）
                - glm-4.7（智谱 GLM）
            base_url: API 地址
            max_tokens: 最大生成 token 数（默认 4096）
            enable_thinking: 是否启用思考模式（仅 Opus/Sonnet 4.6）
            thinking_effort: 思考强度（high/medium/low）
            **kwargs: 传递给 LLMService 的其他参数
        """
        super().__init__(**kwargs)

        # 保存配置
        self._model_name = model
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._thinking_effort = thinking_effort

        # 创建 Anthropic 客户端
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0
        )

        # 创建 Anthropic 适配器（用于格式转换）
        self._adapter = AnthropicLLMAdapter()

        print(f"✓ AnthropicLLMService 初始化完成")
        print(f"  - 模型: {model}")
        print(f"  - API: {base_url}")
        print(f"  - 思考模式: {'启用' if enable_thinking else '禁用'}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        处理帧 - 接收 LLMContextFrame

        这是包装器链的关键：从 Context Aggregator 接收聚合后的上下文。
        """
        await super().process_frame(frame, direction)

        # 处理 LLMContextFrame（从 Context Aggregator 发送）
        if isinstance(frame, LLMContextFrame):
            context = frame.context
            if isinstance(context, LLMContext):
                await self._process_context(context)

        # 其他帧直接传递
        await self.push_frame(frame, direction)

    async def _process_context(self, context: LLMContext):
        """
        处理 LLM Context - 调用 Anthropic API

        使用 Adapter 将通用格式转换为 Anthropic 格式。
        """
        try:
            # 使用 Adapter 转换格式
            invocation_params = self._adapter.get_llm_invocation_params(context)

            # 构建 Anthropic API 请求参数
            request_params = {
                "model": self._model_name,
                "max_tokens": self._max_tokens,
                **invocation_params,  # 包含 messages, system, tools 等
            }

            # 添加思考模式（如果启用）
            if self._enable_thinking:
                request_params["thinking"] = {"type": "adaptive"}
                request_params["output_config"] = {"effort": self._thinking_effort}

            # 推送开始帧
            await self.push_frame(LLMFullResponseStartFrame())

            # 流式调用 Anthropic API
            assistant_message = ""
            async with self.client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    # 推送每个 token 到下游
                    await self.push_frame(LLMTextFrame(text))
                    assistant_message += text

            # 推送结束帧
            await self.push_frame(LLMFullResponseEndFrame())

            logger.info(f"AnthropicLLM: 响应完成")

        except Exception as e:
            logger.error(f"AnthropicLLM: API 调用失败: {e}")
            import traceback
            traceback.print_exc()

    def get_model_name(self) -> str:
        """返回模型显示名称"""
        return f"Anthropic ({self._model_name})"
