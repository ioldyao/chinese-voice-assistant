"""测试 Moondream 本地视觉模型"""
import asyncio
from PIL import Image, ImageGrab
from pipecat.services.moondream.vision import MoondreamService
from pipecat.frames.frames import UserImageRawFrame, VisionTextFrame


async def test_moondream():
    """测试 Moondream 视觉理解"""
    print("=" * 60)
    print("Moondream 本地视觉模型测试")
    print("=" * 60)

    # 1. 初始化 Moondream
    print("\n⏳ 加载 Moondream 模型...")
    moondream = MoondreamService(use_cpu=False)  # 使用 GPU 加速
    print("✓ Moondream 模型已加载")

    # 2. 截取当前屏幕
    print("\n📸 截取当前屏幕...")
    screenshot = ImageGrab.grab()
    print(f"✓ 截图完成，原始尺寸: {screenshot.size}")

    # 3. 缩小图片尺寸（Moondream 推荐 < 1024x1024）
    max_size = 800
    if max(screenshot.size) > max_size:
        ratio = max_size / max(screenshot.size)
        new_size = (int(screenshot.size[0] * ratio), int(screenshot.size[1] * ratio))
        screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)
        print(f"✓ 已缩放到: {screenshot.size}")

    # 4. 转换为 RGB 格式
    if screenshot.mode != 'RGB':
        screenshot = screenshot.convert('RGB')
        print(f"✓ 已转换为 RGB 格式")

    # 5. 创建 UserImageRawFrame（使用英文问题）
    question = "Describe what you see in this image in detail."
    frame = UserImageRawFrame(
        image=screenshot.tobytes(),
        format=screenshot.mode,
        size=screenshot.size,
        text=question
    )
    print(f"\n❓ 问题: {question}")

    # 5. 调用 Moondream
    print("\n⏳ 调用 Moondream 分析图片...")
    description = ""
    async for output_frame in moondream.run_vision(frame):
        if isinstance(output_frame, VisionTextFrame):
            description += output_frame.text

    # 6. 输出结果
    print("\n" + "=" * 60)
    print("📊 Moondream 分析结果:")
    print("=" * 60)
    print(description)
    print("=" * 60)

    if description:
        print("\n✅ Moondream 测试成功！")
    else:
        print("\n❌ Moondream 未返回结果")


if __name__ == "__main__":
    asyncio.run(test_moondream())
