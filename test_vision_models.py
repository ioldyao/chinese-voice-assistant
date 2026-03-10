"""测试多 Vision 模型切换"""
import asyncio
from PIL import ImageGrab
from src.voice_assistant.vision_services import create_vision_service


async def test_vision_models():
    """测试不同的 Vision 模型"""
    print("=" * 70)
    print("多 Vision 模型切换测试")
    print("=" * 70)

    # 截取当前屏幕
    print("\n📸 截取当前屏幕...")
    screenshot = ImageGrab.grab()
    print(f"✓ 截图完成，尺寸: {screenshot.size}")

    # 测试问题
    question = "看一下屏幕显示什么"

    # ==================== 测试 1: Moondream (本地) ====================
    print("\n" + "=" * 70)
    print("测试 1: Moondream 本地模型")
    print("=" * 70)

    try:
        vision_service = create_vision_service(service="moondream", use_cpu=False)
        print(f"✓ 模型加载: {vision_service.get_model_name()}")

        result = await vision_service.analyze_image(screenshot, question)
        print(f"\n📊 Moondream 结果:")
        print("-" * 70)
        print(result)
        print("-" * 70)
    except Exception as e:
        print(f"❌ Moondream 测试失败: {e}")

    # ==================== 测试 2: Qwen-VL-Plus (API) ====================
    print("\n" + "=" * 70)
    print("测试 2: Qwen-VL-Plus API 模型")
    print("=" * 70)

    try:
        # 注意：需要配置正确的 API URL 和 API Key
        vision_service = create_vision_service(
            service="qwen-vl-plus",
            # api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 可选：自定义 URL
            # api_key="your-api-key"          # 可选：自定义 API Key
        )
        print(f"✓ 模型加载: {vision_service.get_model_name()}")

        result = await vision_service.analyze_image(screenshot, question)
        print(f"\n📊 Qwen-VL-Plus 结果:")
        print("-" * 70)
        print(result)
        print("-" * 70)
    except Exception as e:
        print(f"❌ Qwen-VL-Plus 测试失败: {e}")

    # ==================== 测试 3: 从 .env 读取配置 ====================
    print("\n" + "=" * 70)
    print("测试 3: 从 .env 读取配置（默认服务）")
    print("=" * 70)

    try:
        from src.voice_assistant.config import VISION_SERVICE
        print(f"⚙️ 配置: VISION_SERVICE={VISION_SERVICE}")

        vision_service = create_vision_service()  # 不传参数，从环境变量读取
        print(f"✓ 模型加载: {vision_service.get_model_name()}")

        result = await vision_service.analyze_image(screenshot, question)
        print(f"\n📊 默认服务结果:")
        print("-" * 70)
        print(result)
        print("-" * 70)
    except Exception as e:
        print(f"❌ 默认服务测试失败: {e}")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_vision_models())
