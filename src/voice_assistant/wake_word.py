"""智能语音唤醒系统 - 模型加载器（用于 Pipecat 模式）"""
import numpy as np
import sherpa_onnx
from pathlib import Path

from .config import (
    MODELS_DIR,
    SAMPLE_RATE,
    DEFAULT_WAKE_WORDS,
    CONFIG_DIR,
)
from .react_agent import ReactAgent


class SmartWakeWordSystem:
    """智能语音唤醒系统 - 模型加载器（仅用于 Pipecat 模式）"""

    def __init__(self, models_dir=None, enable_voice=False, enable_mcp=False):
        """
        初始化语音助手模型加载器

        Args:
            models_dir: 模型目录路径
            enable_voice: 启用语音播报（Pipecat 模式中由 TTS Processor 处理）
            enable_mcp: 启用 MCP（Pipecat 模式中将异步启动，此参数被忽略）
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.sample_rate = SAMPLE_RATE

        print("正在初始化智能语音助手...")

        # 阶段1: KWS模型（轻量级）
        self.kws_model = self.create_kws_model()

        # 阶段2: ASR模型（重量级）
        self.asr_model = self.create_asr_model()

        # React Agent（MCP 将在 Pipecat 模式中异步启动）
        self.agent = ReactAgent()

        print(f"✓ KWS模型已加载")
        print(f"✓ ASR模型已加载")
        print(f"✓ React Agent 已创建（MCP 将稍后异步启动）")

    def create_kws_model(self):
        """创建KWS关键词检测模型"""
        kws_dir = self.models_dir / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"

        if not kws_dir.exists():
            raise FileNotFoundError(f"KWS模型目录不存在: {kws_dir}")

        # 创建关键词文件（格式：拼音音节 @中文）
        keywords_file = CONFIG_DIR / "keywords.txt"
        if not keywords_file.exists():
            print("⚠️  创建默认关键词文件...")
            keywords_file.parent.mkdir(parents=True, exist_ok=True)
            with open(keywords_file, 'w', encoding='utf-8') as f:
                # 格式：拼音音节(空格分隔) @中文
                # 使用带声调的拼音韵母，空格分隔
                f.write("x iǎo zh ì @小智\n")
                f.write("n ǐ h ǎo zh ù sh ǒu @你好助手\n")
                f.write("zh ì n éng zh ù sh ǒu @智能助手\n")

        kws = sherpa_onnx.KeywordSpotter(
            tokens=str(kws_dir / "tokens.txt"),
            encoder=str(kws_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            decoder=str(kws_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            joiner=str(kws_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
            num_threads=2,
            keywords_file=str(keywords_file),
            provider="cpu",
        )

        print(f"📋 加载关键词: {keywords_file}")
        return kws

    def create_asr_model(self):
        """创建ASR完整识别模型"""
        model_file = self.models_dir / "sherpa-onnx-paraformer-zh-2024-03-09" / "model.int8.onnx"
        tokens_file = self.models_dir / "sherpa-onnx-paraformer-zh-2024-03-09" / "tokens.txt"

        if not model_file.exists():
            raise FileNotFoundError(f"ASR模型文件不存在: {model_file}")

        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            str(model_file),
            str(tokens_file),
            num_threads=2,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            provider="cpu"
        )
        return recognizer
