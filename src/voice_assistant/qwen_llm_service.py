"""
MCP 工具转换器 - 使用官方 Function Calling 机制

将 MCP (Model Context Protocol) 工具转换为 Pipecat 官方格式，
并使用官方 register_function API 注册函数处理器。

功能说明：
1. MCP 工具格式转换：MCP → OpenAI / FunctionSchema / ToolsSchema
2. 函数注册：使用官方 register_function API
3. 统一处理器：catch-all 处理所有 MCP 工具调用
4. 事件处理：官方事件处理器（on_function_calls_started 等）

官方文档：
- https://docs.pipecat.ai/guides/learn/llm
- https://reference-server.pipecat.ai/en/latest/api/pipecat.services.llm_service

MCP 协议：
- https://modelcontextprotocol.io/
"""
from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.llm_service import FunctionCallParams


# ==================== MCP 工具格式转换 ====================

def mcp_tools_to_openai_format(mcp_tools: list[dict]) -> list[dict]:
    """
    将 MCP 工具列表转换为 OpenAI API 格式

    MCP 工具格式（来自 MCP Protocol）：
    ```json
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
    ```

    OpenAI API 格式：
    ```json
    [
        {
            "type": "function",
            "function": {
                "name": "browser_navigate",
                "description": "Navigate to a URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to navigate to"}
                    },
                    "required": ["url"]
                }
            }
        }
    ]
    ```

    Args:
        mcp_tools: MCP 工具列表

    Returns:
        OpenAI API 格式的工具列表

    使用示例：
    ```python
    mcp_tools = await mcp.list_all_tools_async()
    openai_tools = mcp_tools_to_openai_format(mcp_tools)
    context = OpenAILLMContext(messages, tools=openai_tools)
    ```
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


def mcp_tools_to_function_schemas(mcp_tools: list[dict]) -> list[FunctionSchema]:
    """
    将 MCP 工具列表转换为 Pipecat FunctionSchema 列表

    使用官方 FunctionSchema 类，提供类型安全的工具定义。

    FunctionSchema 格式：
    ```python
    FunctionSchema(
        name="browser_navigate",
        description="Navigate to a URL",
        properties={
            "url": {
                "type": "string",
                "description": "URL to navigate to"
            }
        },
        required=["url"]
    )
    ```

    Args:
        mcp_tools: MCP 工具列表

    Returns:
        Pipecat FunctionSchema 列表

    使用示例：
    ```python
    mcp_tools = await mcp.list_all_tools_async()
    schemas = mcp_tools_to_function_schemas(mcp_tools)
    tools_schema = ToolsSchema(standard_tools=schemas)
    ```
    """
    schemas = []
    for tool in mcp_tools:
        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("input_schema", {})

        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # 使用官方 FunctionSchema
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

    使用官方 ToolsSchema 类，管理工具集合。

    Args:
        mcp_tools: MCP 工具列表

    Returns:
        ToolsSchema 对象

    使用示例：
    ```python
    mcp_tools = await mcp.list_all_tools_async()
    tools_schema = create_tools_schema_from_mcp(mcp_tools)
    ```
    """
    function_schemas = mcp_tools_to_function_schemas(mcp_tools)
    return ToolsSchema(standard_tools=function_schemas)


# ==================== MCP 函数注册（官方 API）====================

async def register_mcp_functions(
    llm_service: OpenAILLMService,
    mcp_manager,
    skill_manager=None
) -> None:
    """
    将 MCP 工具注册为 LLM 函数处理器

    使用 Pipecat 官方 register_function API 和 FunctionCallParams。

    官方 API 说明：
    - llm_service.register_function(function_name, handler)
    - function_name: None 表示 catch-all 处理器（捕获所有函数调用）
    - handler: async def handler(params: FunctionCallParams)

    FunctionCallParams 属性：
    - function_name: str - 函数名称
    - tool_call_id: str - 工具调用 ID
    - arguments: dict - 函数参数
    - llm: LLMService - LLM 服务实例
    - context: OpenAILLMContext - LLM 上下文
    - result_callback: Callable - 结果回调函数

    Args:
        llm_service: LLM 服务实例（QwenLLMService、OpenAILLMService 等）
        mcp_manager: MCP 管理器实例（MCPManager）
        skill_manager: 技能管理器实例（可选，用于处理 skill_execute）

    使用示例：
    ```python
    # 在主程序中
    await register_mcp_functions(llm, mcp, skill_manager)
    ```

    工作流程：
    1. LLM 决定调用工具
    2. 触发 mcp_function_handler
    3. 如果是 skill_execute，委托给 skill_manager
    4. 否则调用 mcp_manager.call_tool_async()
    5. 通过 result_callback 返回结果
    6. LLM 接收结果并继续对话
    """

    # 创建通用 MCP 函数处理器
    # 使用官方 FunctionCallParams 类型
    async def mcp_function_handler(params: FunctionCallParams):
        """
        统一的 MCP 函数调用处理器

        处理所有 MCP 工具的调用，包括：
        - skill_execute：技能执行
        - openmeteo_weather：天气查询 API（Open-Meteo，免费无需密钥）
        - Playwright 浏览器操作
        - Windows 系统操作
        - 文件系统操作
        - GitHub 操作
        - 其他 MCP 工具
        """
        function_name = params.function_name
        arguments = params.arguments

        # ✅ 特殊处理：skill_execute
        if function_name == "skill_execute" and skill_manager:
            skill_name = arguments.get("skill_name")
            user_input = arguments.get("user_input", "")

            print(f"🔧 调用技能: {skill_name}")
            print(f"   用户输入: {user_input}")

            try:
                result = await skill_manager.execute_skill(skill_name, user_input)

                if result.success:
                    print(f"✓ 技能执行成功")
                    await params.result_callback(result.content)
                else:
                    print(f"✗ 技能执行失败: {result.error}")
                    await params.result_callback(f"错误: {result.error}")

            except Exception as e:
                print(f"❌ 技能调用异常: {e}")
                import traceback
                traceback.print_exc()
                await params.result_callback(f"异常: {str(e)}")

            return

        # ✅ 特殊处理：Open-Meteo 天气 API（免费，无需API密钥）
        if function_name == "openmeteo_weather":
            city = arguments.get("city", "")
            print(f"🌤️ 查询天气: {city}")

            try:
                import aiohttp

                # 第一步：地理编码（城市 → 经纬度）
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=zh&format=json"

                async with aiohttp.ClientSession() as session:
                    async with session.get(geo_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status != 200:
                            await params.result_callback(f"无法查询 {city} 的天气，请检查城市名称。")
                            return

                        geo_data = await response.json()

                        if not geo_data.get("results"):
                            await params.result_callback(f"未找到城市：{city}")
                            return

                        # 获取经纬度和城市名
                        location = geo_data["results"][0]
                        latitude = location["latitude"]
                        longitude = location["longitude"]
                        city_name = location.get("name", city)
                        country = location.get("country", "")

                        # 第二步：查询实时天气
                        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true&timezone=auto&hourly=relativehumidity_2m"

                        async with session.get(weather_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status != 200:
                                await params.result_callback(f"无法获取 {city_name} 的天气数据。")
                                return

                            weather_data = await response.json()

                            # 解析当前天气
                            current = weather_data.get("current_weather", {})
                            temp = current.get("temperature", "N/A")
                            wind_speed = current.get("windspeed", "N/A")
                            wind_direction = current.get("winddirection", "N/A")
                            weather_code = current.get("weathercode", 0)

                            # 天气代码映射（Open-Meteo WMO code）
                            weather_desc_map = {
                                0: "晴朗", 1: "大部晴朗", 2: "多云", 3: "阴天",
                                45: "雾", 48: "雾凇",
                                51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
                                61: "小雨", 63: "中雨", 65: "大雨",
                                71: "小雪", 73: "中雪", 75: "大雪",
                                80: "阵雨", 81: "阵雨", 82: "暴雨",
                                95: "雷阵雨", 96: "雷阵雨", 99: "雷阵雨"
                            }
                            weather_desc = weather_desc_map.get(weather_code, "未知")

                            # 获取湿度（从hourly数据中提取当前小时的）
                            hourly = weather_data.get("hourly", {})
                            humidity = "N/A"
                            if "relativehumidity_2m" in hourly and "time" in hourly:
                                # 获取当前时间戳（用于匹配最近的小时数据）
                                import datetime
                                now = datetime.datetime.now(datetime.timezone.utc)
                                current_time_str = now.strftime("%Y-%m-%dT%H:00")

                                # 找到最接近当前时间的索引
                                times = hourly.get("time", [])
                                humidities = hourly.get("relativehumidity_2m", [])

                                if times and humidities:
                                    # 简单做法：取第一个（通常是当前或最近的小时）
                                    humidity = humidities[0] if len(humidities) > 0 else "N/A"

                            # 风向映射
                            wind_dir_map = {
                                (0, 23): "北", (23, 68): "东北", (68, 113): "东",
                                (113, 158): "东南", (158, 203): "南", (203, 248): "西南",
                                (248, 293): "西", (293, 338): "西北", (338, 361): "北"
                            }
                            wind_dir_str = "北"
                            for (low, high), direction in wind_dir_map.items():
                                if low <= wind_direction < high:
                                    wind_dir_str = direction
                                    break

                            # 格式化输出
                            result = f"""📍 {city_name} ({country}) 当前天气

🌤️ 天气：{weather_desc}
🌡️ 温度：{temp}°C
💧 湿度：{humidity}%
💨 风速：{wind_speed} km/h，{wind_dir_str}风"""

                            print(f"✓ 天气查询成功")
                            await params.result_callback(result)

            except Exception as e:
                print(f"❌ 天气查询失败: {e}")
                import traceback
                traceback.print_exc()
                await params.result_callback(f"天气查询失败: {str(e)}")

            return

        # MCP 工具处理
        print(f"🔧 调用 MCP 工具: {function_name}")
        print(f"   参数: {arguments}")

        try:
            # 调用 MCP 工具（异步）
            result = await mcp_manager.call_tool_async(function_name, arguments)

            if result.success:
                print(f"✓ 工具执行成功")

                # 获取完整内容（不截断）
                content = str(result.content) if result.content else "操作成功完成"

                # 过滤掉浏览器 console messages（只保留关键信息）
                # Playwright MCP 会返回大量 console messages，影响 LLM 理解
                if "### New console messages" in content:
                    parts = content.split("### New console messages")
                    content = parts[0].strip()
                    content += "\n\n[浏览器操作成功]"

                print(f"   结果: {content[:100]}...")

                # 使用官方 result_callback 返回结果
                # 结果会被传递给 LLM，LLM 会基于结果生成回复
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
    # 官方 API: 使用 None 作为 function_name 表示捕获所有函数调用
    # 这样无论 LLM 调用哪个工具，都会由 mcp_function_handler 处理
    llm_service.register_function(None, mcp_function_handler)

    print(f"✓ 已注册 MCP 函数处理器（使用官方 API，捕获所有函数调用）")


# ==================== 事件处理器示例（官方 API）====================

def setup_function_call_event_handlers(
    llm_service: OpenAILLMService,
    task
) -> None:
    """
    设置函数调用事件处理器

    使用官方事件处理器 API，监听函数调用生命周期事件。

    官方事件：
    - on_function_calls_started: 函数调用开始时触发
    - on_completion_timeout: LLM 完成超时时触发

    Args:
        llm_service: LLM 服务实例
        task: PipelineTask 实例

    使用示例：
    ```python
    # 在创建 Pipeline 后
    task = PipelineTask(pipeline, params=...)
    setup_function_call_event_handlers(llm, task)
    ```

    事件处理器用途：
    - 日志记录：记录函数调用详情
    - 监控：监控函数调用性能
    - 调试：调试函数调用流程
    - 自定义逻辑：在特定事件时执行自定义逻辑
    """

    @task.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        """
        函数调用开始时的处理

        Args:
            service: LLM 服务实例
            function_calls: 函数调用列表
        """
        print(f"🎯 开始执行 {len(function_calls)} 个函数调用")
        for fc in function_calls:
            print(f"   - {fc.function_name}")

    @task.event_handler("on_completion_timeout")
    async def on_completion_timeout(service):
        """
        LLM 完成超时处理

        Args:
            service: LLM 服务实例
        """
        print("⏱️ LLM 完成超时")

    print(f"✓ 已设置函数调用事件处理器")


# ==================== 工具调用增强 ====================

class MCPFunctionCallLogger:
    """
    MCP 函数调用日志记录器

    记录所有 MCP 工具调用的详细信息，用于调试和监控。

    使用示例：
    ```python
    logger = MCPFunctionCallLogger()

    @task.event_handler("on_function_calls_started")
    async def log_function_calls(service, function_calls):
        logger.log_calls(function_calls)
    ```
    """

    def __init__(self):
        self.call_history = []

    def log_call(self, function_name: str, arguments: dict, result: Any):
        """记录单次函数调用"""
        self.call_history.append({
            "function_name": function_name,
            "arguments": arguments,
            "result": str(result)[:200],  # 限制结果长度
            "timestamp": __import__("time").time()
        })

    def log_calls(self, function_calls: list):
        """记录多个函数调用"""
        for fc in function_calls:
            self.log_call(fc.function_name, {}, "pending")

    def get_summary(self) -> dict:
        """获取调用摘要"""
        return {
            "total_calls": len(self.call_history),
            "unique_functions": len(set(c["function_name"] for c in self.call_history)),
            "last_call": self.call_history[-1] if self.call_history else None
        }
