"""Qwen LLM Service - 基于 Pipecat 官方框架的 Qwen-Plus 集成

完全兼容 Pipecat 的 LLMService 接口，享受官方 Context Aggregator 的便利。
"""
import os
from typing import Any

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from .config import DASHSCOPE_API_KEY, DASHSCOPE_API_URL


def mcp_tools_to_function_schemas(mcp_tools: list[dict]) -> list[FunctionSchema]:
    """
    将 MCP 工具列表转换为 Pipecat FunctionSchema 列表

    MCP 工具格式（来自 MCP Protocol）：
    {
        "name": "browser_navigate",
        "description": "Navigate to a URL",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to navigate to"}
            },
            "required": ["url"]
        }
    }

    Pipecat FunctionSchema 格式：
    FunctionSchema(
        name="browser_navigate",
        description="Navigate to a URL",
        properties={"url": {"type": "string", "description": "URL to navigate to"}},
        required=["url"]
    )

    Args:
        mcp_tools: MCP 工具列表

    Returns:
        Pipecat FunctionSchema 列表
    """
    schemas = []
    for tool in mcp_tools:
        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("input_schema", {})

        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # 创建 FunctionSchema
        schema = FunctionSchema(
            name=name,
            description=description,
            properties=properties,
            required=required
        )
        schemas.append(schema)

    return schemas


def create_tools_schema_from_mcp(mcp_tools: list[dict]) -> ToolsSchema:
    """
    从 MCP 工具创建 Pipecat ToolsSchema

    Args:
        mcp_tools: MCP 工具列表

    Returns:
        ToolsSchema 对象
    """
    function_schemas = mcp_tools_to_function_schemas(mcp_tools)
    return ToolsSchema(standard_tools=function_schemas)


def mcp_tools_to_openai_format(mcp_tools: list[dict]) -> list[dict]:
    """
    将 MCP 工具列表转换为 OpenAI API 格式（用于 LLMContext）

    OpenAI API 格式：
    [
        {
            "type": "function",
            "function": {
                "name": "browser_navigate",
                "description": "Navigate to a URL",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
    ]

    Args:
        mcp_tools: MCP 工具列表

    Returns:
        OpenAI API 格式的工具列表
    """
    openai_tools = []
    for tool in mcp_tools:
        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("input_schema", {})

        openai_tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema  # MCP input_schema 已经是 JSON Schema 格式
            }
        }
        openai_tools.append(openai_tool)

    return openai_tools


async def register_mcp_functions(llm_service: OpenAILLMService, mcp_manager):
    """
    将 MCP 工具注册为 LLM 函数处理器

    Args:
        llm_service: QwenLLMService 实例
        mcp_manager: MCPManager 实例
    """
    from pipecat.services.llm_service import FunctionCallParams

    # 创建通用 MCP 函数处理器
    async def mcp_function_handler(params: FunctionCallParams):
        """统一的 MCP 函数调用处理器"""
        function_name = params.function_name
        arguments = params.arguments

        print(f"🔧 调用 MCP 工具: {function_name}")
        print(f"   参数: {arguments}")

        try:
            # 调用 MCP 工具（异步）
            result = await mcp_manager.call_tool_async(function_name, arguments)

            if result.success:
                print(f"✓ 工具执行成功")

                # 简化输出：移除冗长的浏览器 console 日志
                content = str(result.content) if result.content else "操作成功完成"

                # 过滤掉浏览器 console messages（只保留关键信息）
                if "### New console messages" in content:
                    # 只保留第一部分（Ran Playwright code + Page state）
                    parts = content.split("### New console messages")
                    content = parts[0].strip()
                    # 添加简短说明
                    content += "\n\n[浏览器操作成功]"

                # 限制输出长度（最多 500 字符）
                if len(content) > 500:
                    content = content[:500] + "..."

                print(f"   结果: {content[:100]}...")
                await params.result_callback(content)
            else:
                print(f"✗ 工具执行失败: {result.error}")
                error_msg = f"错误: {result.error}"
                await params.result_callback(error_msg)

        except Exception as e:
            print(f"❌ MCP 调用异常: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"异常: {str(e)}"
            await params.result_callback(error_msg)

    # 注册所有 MCP 工具（使用统一处理器）
    # 使用 None 作为 function_name 表示捕获所有函数调用
    llm_service.register_function(None, mcp_function_handler)

    print(f"✓ 已注册 MCP 函数处理器（捕获所有函数调用）")


class QwenLLMService(OpenAILLMService):
    """
    Qwen LLM Service - 继承自 OpenAILLMService

    阿里云 DashScope API 完全兼容 OpenAI 格式，因此可以直接复用 OpenAILLMService 的所有功能：
    - Function Calling（函数调用）
    - Streaming（流式响应）
    - Context Management（上下文管理）

    技术优势：
    - ✅ 符合 Pipecat 官方最佳实践
    - ✅ 自动管理对话历史（LLMContextAggregatorPair）
    - ✅ 支持 Function Calling（MCP 工具）
    - ✅ 支持流式响应
    - ✅ 保持 Qwen 生态（中文效果好）
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "qwen-plus",
        **kwargs
    ):
        """
        初始化 Qwen LLM Service

        Args:
            api_key: DashScope API Key（默认从环境变量读取）
            base_url: DashScope API URL（默认使用 config.py 配置）
            model: Qwen 模型名称（qwen-plus, qwen-max, qwen-turbo）
            **kwargs: 传递给 OpenAILLMService 的其他参数
        """
        # 使用 DashScope API 配置
        api_key = api_key or DASHSCOPE_API_KEY
        base_url = base_url or DASHSCOPE_API_URL

        # 调用父类构造函数（OpenAILLMService）
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )

        print(f"✓ QwenLLMService 初始化完成")
        print(f"  - 模型: {model}")
        print(f"  - API: {base_url}")


class QwenLLMContext(OpenAILLMContext):
    """
    Qwen LLM Context - 继承自 OpenAILLMContext

    由于 Qwen API 完全兼容 OpenAI 格式，可以直接使用 OpenAILLMContext
    无需任何修改。

    使用示例：
    ```python
    messages = [
        {"role": "system", "content": "你是一个智能助手"},
        {"role": "user", "content": "你好"}
    ]
    context = QwenLLMContext(messages, tools)
    ```
    """

    def __init__(self, messages: list[dict] | None = None, tools: Any = None):
        """
        初始化 Qwen LLM Context

        Args:
            messages: 对话历史（OpenAI 格式）
            tools: Function Calling 工具（OpenAI 格式）
        """
        super().__init__(messages=messages, tools=tools)
