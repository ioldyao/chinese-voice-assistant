"""配置文件"""
import os
from pathlib import Path

# API配置 - 建议使用环境变量
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-49d20b6630984acabb4f28aa0bc7ab17")
ALIYUN_APPKEY = os.getenv("ALIYUN_APPKEY", "YOUR_APPKEY")

# API URL
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALIYUN_TTS_URL = "https://nls-gateway-cn-shanghai.aliyuncs.com/rest/v1/tts/async"

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
