"""配置文件"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 尝试加载 .env 文件（如果 dotenv 可用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv 未安装时，跳过 .env 文件加载
    # 依赖环境变量或其他配置方式
    pass

# ==================== LLM 配置 - 从环境变量读取 ====================
# 指定使用哪个 LLM 服务：qwen | deepseek | openai | anthropic
LLM_SERVICE = os.getenv("LLM_SERVICE", "qwen")  # 默认使用 qwen

# Qwen (阿里云 DashScope) 配置
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_API_URL = os.getenv("QWEN_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")  # qwen-plus, qwen-max, qwen-turbo, 或本地模型

# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # deepseek-chat, deepseek-reasoner

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # gpt-4o, gpt-4, gpt-3.5-turbo, o1

# Anthropic Claude 配置
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
ANTHROPIC_ENABLE_THINKING = os.getenv("ANTHROPIC_ENABLE_THINKING", "false").lower() == "true"
ANTHROPIC_THINKING_EFFORT = os.getenv("ANTHROPIC_THINKING_EFFORT", "medium")  # high, medium, low

# ==================== 向后兼容配置（旧变量名） ====================
DASHSCOPE_API_KEY = QWEN_API_KEY  # 向后兼容
DASHSCOPE_API_URL = QWEN_API_URL  # 向后兼容

# ==================== TTS 配置 ====================
ALIYUN_APPKEY = os.getenv("ALIYUN_APPKEY")
ALIYUN_TTS_URL = os.getenv("ALIYUN_TTS_URL", "https://nls-gateway-cn-shanghai.aliyuncs.com/rest/v1/tts/async")  # 默认阿里云 TTS URL

# ==================== Vision 配置 - 从环境变量读取 ====================
# 指定使用哪个 Vision 服务
VISION_SERVICE = os.getenv("VISION_SERVICE", "moondream")  # 默认使用 moondream 本地模型

# Moondream 本地模型配置
MOONDREAM_USE_CPU = os.getenv("MOONDREAM_USE_CPU", "false").lower() == "true"

# Qwen-VL API 配置
QWEN_VL_API_URL = os.getenv("QWEN_VL_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_VL_API_KEY = os.getenv("QWEN_VL_API_KEY", DASHSCOPE_API_KEY)  # 默认使用 DASHSCOPE_API_KEY

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
TTS_AUDIO_DIR = DATA_DIR / "tts_audio"

# 音频配置
SAMPLE_RATE = 16000
CHUNK_SIZE = 512

# 音频设备配置（可选）
# 通过 list_audio_devices.py 查看设备列表
# 留空表示使用系统默认设备
AUDIO_INPUT_DEVICE_INDEX = os.getenv("AUDIO_INPUT_DEVICE_INDEX", None)
AUDIO_OUTPUT_DEVICE_INDEX = os.getenv("AUDIO_OUTPUT_DEVICE_INDEX", None)

# ==================== TTS 配置 ====================
# 指定使用哪个 TTS 引擎：piper | dashscope | dashscope_realtime | edge | azure | coqui
TTS_SERVICE = os.getenv("TTS_SERVICE", "piper")  # 默认使用 piper

# DashScope Realtime WebSocket 配置（低延迟，推荐）
DASHSCOPE_REALTIME_MODEL = os.getenv("DASHSCOPE_REALTIME_MODEL", "qwen3-tts-flash-realtime")
DASHSCOPE_REALTIME_VOICE = os.getenv("DASHSCOPE_REALTIME_VOICE", "Cherry")
DASHSCOPE_REALTIME_MODE = os.getenv("DASHSCOPE_REALTIME_MODE", "server_commit")  # server_commit | commit
DASHSCOPE_REALTIME_URL = os.getenv("DASHSCOPE_REALTIME_URL", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime")

# Piper TTS 配置（本地，最快）
PIPER_TTS_MODEL_PATH = os.getenv(
    "PIPER_TTS_MODEL_PATH",
    str(MODELS_DIR / "piper" / "zh_CN-huayan-medium.onnx")
)

# Piper TTS 合成参数（调节音色和语速）
PIPER_VOLUME = float(os.getenv("PIPER_VOLUME", "1.0"))  # 音量 (0.1-2.0, 1.0 = 正常)
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))  # 语速 (0.5-2.0, 1.0 = 正常, <1 = 快, >1 = 慢)
PIPER_NOISE_SCALE = float(os.getenv("PIPER_NOISE_SCALE", "0.667"))  # 音频变化 (0.1-1.0, 0.667 = 自然)
PIPER_NOISE_W_SCALE = float(os.getenv("PIPER_NOISE_W_SCALE", "0.8"))  # 说话变化 (0.1-1.0, 0.8 = 自然)
PIPER_NORMALIZE_AUDIO = os.getenv("PIPER_NORMALIZE_AUDIO", "true").lower() == "true"  # 是否标准化音频

# DashScope TTS 配置（阿里云，音质好，HTTP 流式）
DASHSCOPE_TTS_MODEL = os.getenv("DASHSCOPE_TTS_MODEL", "qwen3-tts-flash")
DASHSCOPE_TTS_VOICE = os.getenv("DASHSCOPE_TTS_VOICE", "Cherry")

# DashScope TTS 可选模型（HTTP 流式）：
# - qwen3-tts-flash（默认，快速，高质量）
# - qwen3-tts-instruct-flash（支持指令控制）
# - qwen-audio-turbo（极速）
# - sambert-zhichu-v1（经典模型）

# DashScope Realtime TTS 可选模型（WebSocket 流式，推荐）：
# - qwen3-tts-flash-realtime（推荐，低延迟）
# - qwen3-tts-instruct-flash-realtime（支持指令控制）
# - qwen-tts-realtime（经典模型）

# Edge TTS 配置（微软免费）
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")

# Azure TTS 配置（高质量，需付费）
AZURE_TTS_API_KEY = os.getenv("AZURE_TTS_API_KEY")
AZURE_TTS_REGION = os.getenv("AZURE_TTS_REGION", "eastasia")
AZURE_TTS_VOICE = os.getenv("AZURE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")

# TTS缓存配置
TTS_SHORT_TEXT_LIMIT = 280  # 短文本TTS字符限制
TTS_CACHE_TIMEOUT_SHORT = 10  # 短文本缓存清理时间（秒）
TTS_CACHE_TIMEOUT_LONG = 30   # 长文本缓存清理时间（秒）

# 录音配置
RECORD_SECONDS = 10  # 最大录音时长（秒），支持更长的指令
SILENCE_THRESHOLD = 0.02  # 静音阈值
MAX_SILENCE_FRAMES = 20  # 连续静音帧数（约1.3秒），说完即停
MIN_RECORD_FRAMES = 15  # 最小录音保护帧数（约1秒），防止误判

# 唤醒词配置（格式：拼音音节 @中文）
DEFAULT_WAKE_WORDS = [
    "x iǎo zh ì @小智",
    "n ǐ h ǎo zh ù sh ǒu @你好助手",
    "zh ì n éng zh ù sh ǒu @智能助手"
]

# ==================== MCP Server 配置 ====================
MCP_SERVERS_CONFIG_FILE = CONFIG_DIR / "mcp_servers.json"


def load_mcp_servers_config() -> List[Dict[str, Any]]:
    """
    加载 MCP Server 配置

    Returns:
        已启用的 Server 配置列表
        格式: [
            {
                "name": "playwright",
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
                "timeout": 120,
                "env": {...}  # 可选
            },
            ...
        ]
    """
    if not MCP_SERVERS_CONFIG_FILE.exists():
        print(f"⚠️ MCP Server 配置文件不存在: {MCP_SERVERS_CONFIG_FILE}")
        print("   将使用默认配置（仅启用 playwright）")
        return [
            {
                "name": "playwright",
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
                "timeout": 120,
            }
        ]

    try:
        with open(MCP_SERVERS_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        servers = config.get("servers", [])

        # 只返回已启用的 Server
        enabled_servers = [
            {
                "name": server["name"],
                "command": server["command"],
                "args": server["args"],
                "timeout": server.get("timeout", 60),
                "env": server.get("env"),  # 可选环境变量
            }
            for server in servers
            if server.get("enabled", False)
        ]

        return enabled_servers

    except Exception as e:
        print(f"❌ 加载 MCP Server 配置失败: {e}")
        print("   将使用默认配置（仅启用 playwright）")
        return [
            {
                "name": "playwright",
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
                "timeout": 120,
            }
        ]


def get_mcp_server_info() -> str:
    """
    获取 MCP Server 配置信息（用于显示）

    Returns:
        配置信息的格式化字符串
    """
    servers = load_mcp_servers_config()
    if not servers:
        return "无已启用的 MCP Server"

    info_lines = [f"已启用 {len(servers)} 个 MCP Server:"]
    for server in servers:
        info_lines.append(f"  - {server['name']}: {server['command']} {' '.join(server['args'])}")

    return "\n".join(info_lines)
