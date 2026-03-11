"""Vision Services - 支持多种视觉模型的统一接口"""
import asyncio
from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image

from .config import (
    VISION_SERVICE,
    MOONDREAM_USE_CPU,
    QWEN_VL_API_URL,
    QWEN_VL_API_KEY
)


class BaseVisionService(ABC):
    """视觉服务抽象基类"""

    @abstractmethod
    async def analyze_image(self, image: Image.Image, question: str) -> str:
        """
        分析图片并返回描述

        Args:
            image: PIL Image 对象
            question: 用户问题（中文）

        Returns:
            str: 图片描述结果
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型名称"""
        pass


class MoondreamVisionService(BaseVisionService):
    """Moondream 本地视觉模型服务"""

    def __init__(self, use_cpu: bool = False):
        """
        初始化 Moondream 服务

        Args:
            use_cpu: 是否强制使用 CPU
        """
        from pipecat.services.moondream.vision import MoondreamService
        print("⏳ 加载 Moondream 视觉模型...")
        self.moondream = MoondreamService(use_cpu=use_cpu)
        print("✓ Moondream 模型已加载")

    async def analyze_image(self, image: Image.Image, question: str) -> str:
        """使用 Moondream 分析图片"""
        from pipecat.frames.frames import UserImageRawFrame, VisionTextFrame

        try:
            # 1. 转换为 RGB 格式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 2. 将中文问题翻译为英文
            question_en = self._translate_to_english(question)
            print(f"💬 英文提示: {question_en}")

            # 3. 创建 UserImageRawFrame
            frame = UserImageRawFrame(
                image=image.tobytes(),
                format=image.mode,
                size=image.size,
                text=question_en
            )

            # 4. 调用 Moondream
            description = ""
            async for output_frame in self.moondream.run_vision(frame):
                if isinstance(output_frame, VisionTextFrame):
                    description += output_frame.text

            return description if description else "未能识别图片内容"

        except Exception as e:
            return f"Moondream 处理失败: {str(e)}"

    def _translate_to_english(self, chinese_text: str) -> str:
        """将常见中文视觉问题翻译为英文"""
        text_lower = chinese_text.lower()
        if any(kw in text_lower for kw in ["看", "查看", "描述", "显示"]):
            return "Describe what you see in this image in detail."
        elif "分析" in text_lower:
            return "Analyze the content of this image."
        elif "界面" in text_lower or "屏幕" in text_lower:
            return "Describe the user interface shown in this screenshot."
        else:
            return "What do you see in this image?"

    def get_model_name(self) -> str:
        return "Moondream (Local)"


class QwenVLVisionService(BaseVisionService):
    """Qwen-VL 视觉模型服务（阿里云 API）"""

    def __init__(self, model: str = "qwen-vl-plus", api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        初始化 Qwen-VL 服务

        Args:
            model: 模型名称（qwen-vl-plus 或 qwen-vl-max）
            api_url: API 地址
            api_key: API 密钥
        """
        self.model = model
        self.api_url = api_url or DASHSCOPE_API_URL
        self.api_key = api_key or VISION_API_KEY
        print(f"✓ Qwen-VL 服务已初始化: {model}")

    async def analyze_image(self, image: Image.Image, question: str) -> str:
        """使用 Qwen-VL API 分析图片"""
        import httpx
        import base64
        import io

        try:
            # 1. 转换为 base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()

            print(f"📸 调用 {self.model} API...")

            # 2. 调用 API
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.api_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ]
                        }],
                        "max_tokens": 2000,
                        "temperature": 0.7
                    }
                )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"API错误 {response.status_code}: {response.text}"

        except Exception as e:
            return f"Qwen-VL API 调用失败: {str(e)}"

    def get_model_name(self) -> str:
        return f"Qwen-VL ({self.model})"


class VisionFactory:
    """Vision 服务工厂"""

    @staticmethod
    def create_vision_service(
        service: Optional[str] = None,
        use_cpu: Optional[bool] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> BaseVisionService:
        """
        根据配置创建 Vision 服务

        Args:
            service: 服务名称（None 则从环境变量 VISION_SERVICE 读取）
                支持：moondream, qwen-vl-plus, qwen-vl-max
            use_cpu: 是否使用 CPU（仅 Moondream，None 则从环境变量读取）
            api_url: API URL（仅 Qwen-VL，None 则从环境变量读取）
            api_key: API 密钥（仅 Qwen-VL，None 则从环境变量读取）

        Returns:
            BaseVisionService: 视觉服务实例
        """
        # 1. 确定使用哪个服务
        service = service or VISION_SERVICE
        service_lower = service.lower()

        # 2. 根据服务类型创建实例
        if service_lower == "moondream":
            # Moondream 本地模型
            cpu = use_cpu if use_cpu is not None else MOONDREAM_USE_CPU
            return MoondreamVisionService(use_cpu=cpu)

        elif service_lower in ["qwen-vl-plus", "qwen-vl-max"]:
            # Qwen-VL API 模型
            url = api_url or QWEN_VL_API_URL
            key = api_key or QWEN_VL_API_KEY
            return QwenVLVisionService(model=service_lower, api_url=url, api_key=key)

        else:
            raise ValueError(
                f"不支持的 Vision 服务: {service}。"
                f"支持的服务：moondream, qwen-vl-plus, qwen-vl-max"
            )


# 便捷函数
def create_vision_service(**kwargs) -> BaseVisionService:
    """创建 Vision 服务的便捷函数"""
    return VisionFactory.create_vision_service(**kwargs)
