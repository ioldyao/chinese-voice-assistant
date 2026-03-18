"""
Anthropic LLM Adapter - 将通用格式转换为 Anthropic Messages API 格式

包装器链模式：
- BaseLLMAdapter 定义接口
- AnthropicLLMAdapter 实现 Anthropic 格式转换
- LLMService 使用 Adapter 处理不同 API
"""
from typing import Any, Dict, List

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMSpecificMessage,
    NotGiven,
)


class AnthropicLLMAdapter(BaseLLMAdapter):
    """
    Anthropic Messages API 适配器

    功能：
    - 将通用 LLMContext 转换为 Anthropic Messages API 格式
    - 处理 Anthropic 特定的字段（system, thinking, beta 等）
    - 支持工具调用的格式转换
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """LLMSpecificMessage 的标识符"""
        return "anthropic"

    def to_provider_tools_format(self, tools_schema) -> List[Dict[str, Any]]:
        """
        转换工具格式：Pipecat 标准格式 → Anthropic 格式

        Anthropic 工具格式：
        {
            "name": "tool_name",
            "description": "Tool description",
            "input_schema": {...}  # JSON Schema
        }

        注意：当前版本暂未实现工具调用，返回空列表。
        """
        # TODO: 实现完整的工具格式转换
        # 暂时返回空列表
        return []

    def get_llm_invocation_params(self, context: LLMContext) -> Dict[str, Any]:
        """
        获取 Anthropic API 调用参数

        将通用 LLMContext 转换为 Anthropic Messages API 格式：
        - 提取 system 消息
        - 转换 messages 格式
        - 转换 tools 格式（如果需要）
        """
        messages = self.get_messages(context)

        # 分离 system 消息和对话消息
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if isinstance(msg, LLMSpecificMessage):
                # 跳过 Anthropic 特定的消息（由其他地方处理）
                continue
            elif msg.get("role") == "system":
                system_message = msg.get("content")
            elif msg.get("role") in ["user", "assistant"]:
                # Anthropic 格式：直接使用
                anthropic_messages.append(msg)

        # 构建请求参数
        params: Dict[str, Any] = {
            "messages": anthropic_messages,
        }

        # 添加 system（如果有）
        if system_message:
            params["system"] = system_message

        # 添加 tools（如果有）
        if hasattr(context, 'tools') and context.tools and context.tools is not NotGiven:
            params["tools"] = self._convert_tools_to_anthropic_format(context.tools)

        # 添加 tool_choice（如果有）
        if hasattr(context, 'tool_choice') and context.tool_choice is not NotGiven:
            params["tool_choice"] = self._convert_tool_choice_to_anthropic_format(context.tool_choice)

        return params

    def _convert_tools_to_anthropic_format(self, tools) -> List[Dict[str, Any]]:
        """
        转换工具格式：通用格式 → Anthropic 格式

        Anthropic 工具格式：
        {
            "name": "tool_name",
            "description": "Tool description",
            "input_schema": {...}  # JSON Schema
        }
        """
        # TODO: 实现工具格式转换
        # 暂时返回空列表
        return []

    def _convert_tool_choice_to_anthropic_format(self, tool_choice) -> Any:
        """
        转换 tool_choice：通用格式 → Anthropic 格式

        Anthropic tool_choice 格式：
        - "auto" - 自动选择
        - "none" - 不使用工具
        - {"type": "tool", "name": "tool_name"} - 指定工具
        - {"type": "tool", "name": "tool1"} - 多个工具
        """
        # TODO: 实现 tool_choice 转换
        return tool_choice

    def get_messages_for_logging(self, context: LLMContext) -> List[Dict[str, Any]]:
        """获取用于日志记录的消息（去除敏感数据）"""
        messages = []
        for msg in self.get_messages(context):
            if isinstance(msg, LLMSpecificMessage):
                continue

            # 创建副本以避免修改原始数据
            log_msg = dict(msg)

            # 去除敏感数据
            if "content" in log_msg and isinstance(log_msg["content"], list):
                for item in log_msg["content"]:
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image/"):
                            item["image_url"]["url"] = "data:image/..."

            messages.append(log_msg)

        return messages
