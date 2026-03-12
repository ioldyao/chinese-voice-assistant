"""
智能语音助手 - 混合架构版
阶段1: KWS轻量级关键词检测（持续监听）
阶段2: ASR完整语音识别（唤醒后）
集成: Qwen LLM + MCP + Pipecat 官方框架 + Vision + Agent Skills

注意：使用延迟导入以避免循环依赖和测试时的导入问题
"""

__version__ = "2.7.0"

# 延迟导入辅助函数
def __getattr__(name: str):
    """延迟导入模块，避免在顶层导入时触发复杂依赖链"""
    import importlib

    module_map = {
        # Wake Word & ASR
        "SmartWakeWordSystem": (".wake_word", "SmartWakeWordSystem"),
        # TTS
        "TTSManager": (".tts", "TTSManager"),
        # Vision
        "VisionUnderstanding": (".vision", "VisionUnderstanding"),
        # MCP
        "MCPClient": (".mcp_client", "MCPClient"),
        "MCPManager": (".mcp_client", "MCPManager"),
        "MCPResponse": (".mcp_client", "MCPResponse"),
        # React Agent
        "ReactAgent": (".react_agent", "ReactAgent"),
        "ReActParser": (".react_agent", "ReActParser"),
        "ReActStep": (".react_agent", "ReActStep"),
        # LLM Services
        "QwenLLMService": (".llm_services", "QwenLLMService"),
        "DeepSeekLLMService": (".llm_services", "DeepSeekLLMService"),
        "OpenAILLMServiceWrapper": (".llm_services", "OpenAILLMServiceWrapper"),
        "UnifiedLLMContext": (".llm_services", "UnifiedLLMContext"),
        "create_llm_service": (".llm_services", "create_llm_service"),
        "create_llm_context": (".llm_services", "create_llm_context"),
        "LLMFactory": (".llm_services", "LLMFactory"),
        # MCP 工具
        "mcp_tools_to_function_schemas": (".qwen_llm_service", "mcp_tools_to_function_schemas"),
        "create_tools_schema_from_mcp": (".qwen_llm_service", "create_tools_schema_from_mcp"),
        "mcp_tools_to_openai_format": (".qwen_llm_service", "mcp_tools_to_openai_format"),
        "register_mcp_functions": (".qwen_llm_service", "register_mcp_functions"),
        "setup_function_call_event_handlers": (".qwen_llm_service", "setup_function_call_event_handlers"),
        "MCPFunctionCallLogger": (".qwen_llm_service", "MCPFunctionCallLogger"),
        # Agent Skills
        "AgentSkill": (".skills", "AgentSkill"),
        "SkillManager": (".skills", "SkillManager"),
        "SkillProcessor": (".skills", "SkillProcessor"),
        "SkillLoader": (".skills", "SkillLoader"),
        "SkillExecutor": (".skills", "SkillExecutor"),
        "SkillExecutionContext": (".skills", "SkillExecutionContext"),
        "SkillResult": (".skills", "SkillResult"),
        "SkillState": (".skills", "SkillState"),
        "SkillMetadata": (".skills", "SkillMetadata"),
    }

    if name in module_map:
        module_path, attr_name = module_map[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SmartWakeWordSystem",
    "TTSManager",
    "VisionUnderstanding",
    "MCPClient",
    "MCPManager",
    "MCPResponse",
    "ReactAgent",
    "ReActParser",
    "ReActStep",
    # LLM 服务
    "QwenLLMService",
    "DeepSeekLLMService",
    "OpenAILLMServiceWrapper",
    "UnifiedLLMContext",
    "create_llm_service",
    "create_llm_context",
    "LLMFactory",
    # MCP 工具
    "mcp_tools_to_function_schemas",
    "create_tools_schema_from_mcp",
    "mcp_tools_to_openai_format",
    "register_mcp_functions",
    "setup_function_call_event_handlers",
    "MCPFunctionCallLogger",
    # Agent Skills
    "AgentSkill",
    "SkillManager",
    "SkillProcessor",
    "SkillLoader",
    "SkillExecutor",
    "SkillExecutionContext",
    "SkillResult",
    "SkillState",
    "SkillMetadata",
]

# pipecat_main 使用延迟导入，避免初始化时的阻塞
# 使用时通过 from src.voice_assistant import pipecat_main 导入

__version__ = "2.7.0"
__all__ = [
    "SmartWakeWordSystem",
    "TTSManager",
    "VisionUnderstanding",
    "MCPClient",
    "MCPManager",
    "MCPResponse",
    "ReactAgent",
    "ReActParser",
    "ReActStep",
    # LLM 服务
    "QwenLLMService",
    "DeepSeekLLMService",
    "OpenAILLMServiceWrapper",
    "UnifiedLLMContext",
    "create_llm_service",
    "create_llm_context",
    "LLMFactory",
    # MCP 工具
    "mcp_tools_to_function_schemas",
    "create_tools_schema_from_mcp",
    "mcp_tools_to_openai_format",
    "register_mcp_functions",
    "setup_function_call_event_handlers",
    "MCPFunctionCallLogger",
    # Agent Skills
    "AgentSkill",
    "SkillManager",
    "SkillProcessor",
    "SkillLoader",
    "SkillExecutor",
    "SkillExecutionContext",
    "SkillResult",
    "SkillState",
    "SkillMetadata",
]
