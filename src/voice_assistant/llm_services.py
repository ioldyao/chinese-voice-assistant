"""LLM Services - 支持多种 LLM 模型的统一接口"""
from typing import Optional

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext


# ==================== Qwen LLM Service ====================

class QwenLLMService(OpenAILLMService):
    """
    Qwen LLM Service - 阿里云 DashScope API

    特点：
    - 完全兼容 OpenAI 格式
    - 中文理解优秀
    - 支持 Function Calling
    - 支持流式响应
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "qwen-plus",
        **kwargs
    ):
        """
        初始化 Qwen LLM Service

        Args:
            api_key: DashScope API Key
            base_url: DashScope API URL
            model: 模型名称（qwen-plus, qwen-max, qwen-turbo, 或本地模型）
            **kwargs: 传递给 OpenAILLMService 的其他参数
        """
        # 保存模型名称（用于显示）
        self._model_name = model

        # Qwen3 特殊参数：禁用思考模式
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        if "body" not in kwargs["extra"]:
            kwargs["extra"]["body"] = {}

        # 添加 chat_template_kwargs 参数（Qwen3 专用）
        kwargs["extra"]["body"]["chat_template_kwargs"] = {"enable_thinking": False}

        # 调用父类构造函数
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )

    def build_chat_completion_params(self, params_from_context) -> dict:
        """
        构建请求参数并修正消息格式

        修复：本地 Qwen 要求 assistant message 有 tool_calls 时必须包含 content 字段
        """
        # 调用父类方法获取原始参数
        params = super().build_chat_completion_params(params_from_context)

        # 修正 messages：确保 assistant message 有 tool_calls 时也有 content
        if "messages" in params:
            for msg in params["messages"]:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    if "content" not in msg:
                        msg["content"] = ""  # 添加空 content

        return params

    def get_model_name(self) -> str:
        """返回模型显示名称"""
        return f"Qwen ({self._model_name})"


# ==================== DeepSeek LLM Service ====================

class DeepSeekLLMService(OpenAILLMService):
    """
    DeepSeek LLM Service - DeepSeek API

    特点：
    - 兼容 OpenAI 格式
    - 强大的推理能力（DeepSeek-R1）
    - 成本低
    - 支持长上下文
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        **kwargs
    ):
        """
        初始化 DeepSeek LLM Service

        Args:
            api_key: DeepSeek API Key
            base_url: DeepSeek API URL
            model: 模型名称（deepseek-chat, deepseek-reasoner）
            **kwargs: 传递给 OpenAILLMService 的其他参数
        """
        # 保存模型名称（用于显示）
        self._model_name = model

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )

    def get_model_name(self) -> str:
        """返回模型显示名称"""
        return f"DeepSeek ({self._model_name})"


# ==================== OpenAI LLM Service（原生） ====================

class OpenAILLMServiceWrapper(OpenAILLMService):
    """
    OpenAI LLM Service - 官方 OpenAI API

    特点：
    - 官方 API
    - 最新的模型（GPT-4, GPT-4o, o1 等）
    - 稳定可靠
    - 多模态支持
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        **kwargs
    ):
        """
        初始化 OpenAI LLM Service

        Args:
            api_key: OpenAI API Key
            base_url: OpenAI API URL（支持代理）
            model: 模型名称（gpt-4o, gpt-4, gpt-3.5-turbo, o1 等）
            **kwargs: 传递给 OpenAILLMService 的其他参数
        """
        # 保存模型名称（用于显示）
        self._model_name = model

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )

    def get_model_name(self) -> str:
        """返回模型显示名称"""
        return f"OpenAI ({self._model_name})"


# ==================== LLM Context（统一） ====================

class UnifiedLLMContext(OpenAILLMContext):
    """
    统一 LLM Context - 兼容所有 OpenAI 格式的 API

    由于所有服务都兼容 OpenAI 格式，可以直接使用 OpenAILLMContext
    """

    def __init__(self, messages: list[dict] | None = None, tools: list[dict] | None = None):
        """
        初始化 LLM Context

        Args:
            messages: 对话历史（OpenAI 格式）
            tools: Function Calling 工具（OpenAI 格式）
        """
        super().__init__(messages=messages, tools=tools)


# ==================== LLM Factory ====================

class LLMFactory:
    """LLM 服务工厂"""

    @staticmethod
    def create_llm_service(
        service: str,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> OpenAILLMService:
        """
        根据配置创建 LLM 服务

        Args:
            service: 服务名称（qwen, deepseek, openai）
            api_key: API 密钥
            base_url: API 地址（可选，有默认值）
            model: 模型名称（可选，有默认值）
            **kwargs: 传递给 LLMService 的其他参数

        Returns:
            OpenAILLMService: LLM 服务实例
        """
        service_lower = service.lower()

        # 1. Qwen (阿里云 DashScope)
        if service_lower == "qwen":
            base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            model = model or "qwen-plus"
            return QwenLLMService(
                api_key=api_key,
                base_url=base_url,
                model=model,
                **kwargs
            )

        # 2. DeepSeek
        elif service_lower == "deepseek":
            base_url = base_url or "https://api.deepseek.com/v1"
            model = model or "deepseek-chat"
            return DeepSeekLLMService(
                api_key=api_key,
                base_url=base_url,
                model=model,
                **kwargs
            )

        # 3. OpenAI
        elif service_lower == "openai":
            base_url = base_url or "https://api.openai.com/v1"
            model = model or "gpt-4o"
            return OpenAILLMServiceWrapper(
                api_key=api_key,
                base_url=base_url,
                model=model,
                **kwargs
            )

        else:
            raise ValueError(
                f"不支持的 LLM 服务: {service}。"
                f"支持的服务：qwen, deepseek, openai"
            )

    @staticmethod
    def get_model_display_name(llm_service: OpenAILLMService) -> str:
        """获取模型显示名称"""
        if hasattr(llm_service, 'get_model_name'):
            return llm_service.get_model_name()
        elif hasattr(llm_service, '_model_name'):
            return f"LLM ({llm_service._model_name})"
        else:
            return "LLM (Unknown)"


# 便捷函数
def create_llm_service(**kwargs) -> OpenAILLMService:
    """创建 LLM 服务的便捷函数"""
    return LLMFactory.create_llm_service(**kwargs)


def create_llm_context(messages: list[dict] | None = None, tools: list[dict] | None = None) -> UnifiedLLMContext:
    """创建 LLM Context 的便捷函数"""
    return UnifiedLLMContext(messages=messages, tools=tools)
