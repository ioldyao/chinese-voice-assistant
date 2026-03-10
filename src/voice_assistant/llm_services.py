"""
LLM Services - 基于官方实现 + 用户优化

使用 Pipecat 官方推荐的继承模式，保留用户的 Qwen3 优化和工厂模式设计。

架构说明：
- 所有服务继承自官方 OpenAILLMService
- 保留用户的 Qwen3 特殊优化（禁用思考模式）
- 保留用户的 Bug 修复（tool_calls content 字段）
- 使用工厂模式统一创建 LLM 服务
- 支持 Qwen、DeepSeek、OpenAI 等多种服务

官方文档：
- https://docs.pipecat.ai/guides/learn/llm
- https://github.com/pipecat-ai/pipecat/blob/main/COMMUNITY_INTEGRATIONS.md
"""
from typing import Optional

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext


# ==================== Qwen LLM Service ====================

class QwenLLMService(OpenAILLMService):
    """
    Qwen LLM Service - 阿里云 DashScope API

    基于 Pipecat 官方 OpenAILLMService，添加 Qwen3 特殊优化。

    特点：
    - ✅ 完全兼容 OpenAI 格式
    - ✅ 中文理解优秀
    - ✅ 支持 Function Calling（官方机制）
    - ✅ 支持流式响应
    - ✅ Qwen3 优化：禁用思考模式
    - ✅ Bug 修复：tool_calls content 字段

    官方兼容性：
    - 阿里云 DashScope API 完全兼容 OpenAI 格式
    - 可直接使用 OpenAILLMService 的所有功能
    - 支持 Context Aggregator、Function Calling 等

    用户优化：
    1. Qwen3 禁用思考模式：chat_template_kwargs: {enable_thinking: False}
    2. 修复本地 Qwen 严格要求：assistant message 有 tool_calls 时必须有 content

    使用示例：
    ```python
    llm = QwenLLMService(
        api_key="your-api-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus"
    )
    ```
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
        # 这是用户的优化，官方实现中没有
        # 禁用思考模式可以加快响应速度，减少延迟
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        if "body" not in kwargs["extra"]:
            kwargs["extra"]["body"] = {}

        # 添加 chat_template_kwargs 参数（Qwen3 专用）
        kwargs["extra"]["body"]["chat_template_kwargs"] = {"enable_thinking": False}

        # 调用父类构造函数（官方实现）
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )

        print(f"✓ QwenLLMService 初始化完成")
        print(f"  - 模型: {model}")
        print(f"  - API: {base_url}")
        print(f"  - 思考模式: 已禁用（用户优化）")

    def build_chat_completion_params(self, params_from_context) -> dict:
        """
        构建请求参数并修正消息格式

        修复：本地 Qwen 要求 assistant message 有 tool_calls 时必须包含 content 字段
        这是用户的 Bug 修复，官方实现中没有
        """
        # 调用父类方法获取原始参数
        params = super().build_chat_completion_params(params_from_context)

        # 修正 messages：确保 assistant message 有 tool_calls 时也有 content
        # 本地部署的 Qwen 模型严格要求这个格式，否则会返回 400 错误
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

    完全兼容 OpenAI 格式，直接使用官方实现。

    特点：
    - ✅ 兼容 OpenAI 格式
    - ✅ 强大的推理能力（DeepSeek-R1）
    - ✅ 成本低
    - ✅ 支持长上下文

    使用示例：
    ```python
    llm = DeepSeekLLMService(
        api_key="your-api-key",
        model="deepseek-chat"  # 或 "deepseek-reasoner"
    )
    ```
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

        print(f"✓ DeepSeekLLMService 初始化完成")
        print(f"  - 模型: {model}")
        print(f"  - API: {base_url}")

    def get_model_name(self) -> str:
        """返回模型显示名称"""
        return f"DeepSeek ({self._model_name})"


# ==================== OpenAI LLM Service（原生） ====================

class OpenAILLMServiceWrapper(OpenAILLMService):
    """
    OpenAI LLM Service - 官方 OpenAI API

    使用官方实现，仅添加显示名称方法。

    特点：
    - ✅ 官方 API
    - ✅ 最新的模型（GPT-4, GPT-4o, o1 等）
    - ✅ 稳定可靠
    - ✅ 多模态支持

    使用示例：
    ```python
    llm = OpenAILLMServiceWrapper(
        api_key="your-api-key",
        model="gpt-4o"  # 或 "gpt-4", "o1-preview" 等
    )
    ```
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

        print(f"✓ OpenAILLMService 初始化完成")
        print(f"  - 模型: {model}")
        print(f"  - API: {base_url}")

    def get_model_name(self) -> str:
        """返回模型显示名称"""
        return f"OpenAI ({self._model_name})"


# ==================== LLM Context（统一） ====================

class UnifiedLLMContext(OpenAILLMContext):
    """
    统一 LLM Context - 兼容所有 OpenAI 格式的 API

    直接使用官方 OpenAILLMContext，无需修改。

    特点：
    - ✅ 完全兼容 OpenAI 格式
    - ✅ 支持 Function Calling 工具定义
    - ✅ 自动管理对话历史
    - ✅ 支持流式响应

    使用示例：
    ```python
    messages = [
        {"role": "system", "content": "你是一个智能助手"},
        {"role": "user", "content": "你好"}
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "城市"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    context = UnifiedLLMContext(messages, tools)
    ```
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
    """
    LLM 服务工厂

    提供统一的 LLM 服务创建接口，支持通过 .env 文件配置切换。

    设计模式：
    - 工厂模式：统一创建不同 LLM 服务
    - 策略模式：根据 service 参数选择不同实现
    - 配置驱动：通过环境变量控制服务选择

    支持的服务：
    - qwen: 阿里云 Qwen（中文优化）
    - deepseek: DeepSeek（强推理，低成本）
    - openai: OpenAI 官方（最新模型）

    使用示例：
    ```python
    # 方式 1: 使用工厂方法
    llm = LLMFactory.create_llm_service(
        service="qwen",
        api_key="your-key",
        base_url="https://...",
        model="qwen-plus"
    )

    # 方式 2: 使用便捷函数
    llm = create_llm_service(
        service="qwen",
        api_key="your-key"
    )

    # 方式 3: 从 .env 配置读取
    # LLM_SERVICE=qwen
    # QWEN_API_KEY=your-key
    llm = create_llm_service(
        service=LLM_SERVICE,
        api_key=QWEN_API_KEY,
        base_url=QWEN_API_URL,
        model=QWEN_MODEL
    )
    ```
    """

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

        Raises:
            ValueError: 不支持的服务名称

        示例：
        ```python
        # Qwen
        llm = LLMFactory.create_llm_service(
            service="qwen",
            api_key="sk-...",
            model="qwen-plus"
        )

        # DeepSeek
        llm = LLMFactory.create_llm_service(
            service="deepseek",
            api_key="sk-...",
            model="deepseek-chat"
        )

        # OpenAI
        llm = LLMFactory.create_llm_service(
            service="openai",
            api_key="sk-...",
            model="gpt-4o"
        )
        ```
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
        """
        获取模型显示名称

        Args:
            llm_service: LLM 服务实例

        Returns:
            str: 模型显示名称
        """
        if hasattr(llm_service, 'get_model_name'):
            return llm_service.get_model_name()
        elif hasattr(llm_service, '_model_name'):
            return f"LLM ({llm_service._model_name})"
        else:
            return "LLM (Unknown)"


# ==================== 便捷函数 ====================

def create_llm_service(**kwargs) -> OpenAILLMService:
    """
    创建 LLM 服务的便捷函数

    Args:
        **kwargs: 传递给 LLMFactory.create_llm_service 的参数

    Returns:
        OpenAILLMService: LLM 服务实例

    示例：
    ```python
    llm = create_llm_service(
        service="qwen",
        api_key="sk-...",
        model="qwen-plus"
    )
    ```
    """
    return LLMFactory.create_llm_service(**kwargs)


def create_llm_context(
    messages: list[dict] | None = None,
    tools: list[dict] | None = None
) -> UnifiedLLMContext:
    """
    创建 LLM Context 的便捷函数

    Args:
        messages: 对话历史（OpenAI 格式）
        tools: Function Calling 工具（OpenAI 格式）

    Returns:
        UnifiedLLMContext: LLM Context 实例

    示例：
    ```python
    messages = [
        {"role": "system", "content": "你是一个智能助手"},
        {"role": "user", "content": "你好"}
    ]
    tools = [...]  # OpenAI 格式的工具列表

    context = create_llm_context(messages, tools)
    ```
    """
    return UnifiedLLMContext(messages=messages, tools=tools)
