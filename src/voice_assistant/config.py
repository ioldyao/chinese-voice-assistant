"""配置文件"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ==================== LLM 配置 - 从环境变量读取 ====================
# 指定使用哪个 LLM 服务
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

# TTS配置
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
