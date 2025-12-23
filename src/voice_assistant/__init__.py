"""
智能语音助手 - 双阶段识别版
阶段1: KWS轻量级关键词检测（持续监听）
阶段2: ASR完整语音识别（唤醒后）
集成: Windows-MCP + React Agent + Vision
"""

from .wake_word import SmartWakeWordSystem
from .tts import TTSManager
from .vision import VisionUnderstanding
from .mcp_client import MCPClient, MCPManager, MCPResponse
from .react_agent import ReactAgent, ReActParser, ReActStep

# pipecat_main 使用延迟导入，避免初始化时的阻塞
# 使用时通过 from src.voice_assistant import pipecat_main 导入

__version__ = "2.0.0"
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
]
