"""
Anthropic Messages API LLM Service - 纯 Anthropic API 实现

使用 Anthropic Messages API 规范，支持任何兼容该规范的服务。

特点：
- ✅ 纯 Anthropic Messages API（完全不依赖 OpenAI）
- ✅ 直接处理 TranscriptionFrame（不使用 Context Aggregator）
- ✅ 自己维护 Anthropic 格式的对话历史
- ✅ 支持官方 Claude API（api.anthropic.com）
- ✅ 支持智谱 GLM（https://open.bigmodel.cn/api/anthropic）
- ✅ 流式响应
- ✅ 自适应思考模式（Adaptive Thinking）

兼容的服务：
- Anthropic Claude（官方）：https://api.anthropic.com
- 智谱 GLM：https://open.bigmodel.cn/api/anthropic
- 本地模型：Ollama、vLLM 等（需兼容 Messages API）

官方文档：
- https://docs.anthropic.com/en/api/messages
"""
import asyncio
from typing import Optional, AsyncGenerator

from anthropic import AsyncAnthropic
from loguru import logger

from pipecat.services.llm_service import LLMService
from pipecat.frames.frames import (
    Frame,
    LLMTextFrame,
    TranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    InterruptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection


class AnthropicLLMService(LLMService):
    """
    Anthropic Messages API LLM Service - 纯 Anthropic 实现

    特点：
    - ✅ 继承 LLMService 基类（不是 OpenAILLMService）
    - ✅ 直接处理 TranscriptionFrame
    - ✅ 维护 Anthropic 格式的对话历史
    - ✅ 调用纯 Anthropic API（不转换）

    架构：
    1. process_frame() 接收 TranscriptionFrame
    2. 将用户消息添加到 Anthropic 格式的历史记录
    3. 调用 _process_context() 使用 Anthropic API
    4. 推送 LLMTextFrame 到下游

    使用示例：
    ```python
    llm = AnthropicLLMService(
        api_key="sk-ant-xxxxx",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-5-20250929"
    )
    ```
    """

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

        # ✅ Anthropic 格式的对话历史（不使用 OpenAI 格式）
        self._messages = []
        self._system_prompt = None

        # 创建 Anthropic 客户端
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0
        )

        # 追踪 bot 说话状态
        self._bot_speaking = False

        print(f"✓ AnthropicLLMService 初始化完成")
        print(f"  - 模型: {model}")
        print(f"  - API: {base_url}")
        print(f"  - 思考模式: {'启用' if enable_thinking else '禁用'}")

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self._system_prompt = prompt

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """处理帧 - 纯 Anthropic 架构"""
        await super().process_frame(frame, direction)

        # 追踪 bot 说话状态
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        # 中断处理
        if isinstance(frame, InterruptionFrame):
            # 清空未完成的输入
            await self.push_frame(frame, direction)
            return

        # ✅ 直接处理 TranscriptionFrame（不使用 Context Aggregator）
        if isinstance(frame, TranscriptionFrame):
            # 添加用户消息到 Anthropic 格式的历史记录
            self._messages.append({
                "role": "user",
                "content": frame.text
            })

            logger.info(f"AnthropicLLM: 收到用户消息: {frame.text}")

            # 调用 Anthropic API 生成响应
            await self._process_context()

            await self.push_frame(frame, direction)
        else:
            # 其他帧直接传递
            await self.push_frame(frame, direction)

    async def _process_context(self):
        """
        调用 Anthropic API 生成响应

        使用纯 Anthropic Messages API，不做任何格式转换。
        """
        if not self._messages:
            return

        try:
            # 构建请求参数（纯 Anthropic 格式）
            request_params = {
                "model": self._model_name,
                "max_tokens": self._max_tokens,
                "messages": self._messages,
            }

            # 添加 system（如果有）
            if self._system_prompt:
                request_params["system"] = self._system_prompt

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

            # 添加助手响应到历史记录
            self._messages.append({
                "role": "assistant",
                "content": assistant_message
            })

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
