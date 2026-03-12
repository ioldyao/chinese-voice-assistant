"""快速启动脚本 - Pipecat Flows 版本

使用说明：
python scripts/run_flows_version.py
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查依赖...")

    missing_deps = []

    try:
        import pipecat_flows
        print("  ✅ pipecat-ai-flows 已安装")
    except ImportError:
        missing_deps.append("pipecat-ai-flows")
        print("  ❌ pipecat-ai-flows 未安装")

    try:
        import sherpa_onnx
        print("  ✅ sherpa-onnx 已安装")
    except ImportError:
        missing_deps.append("sherpa-onnx")
        print("  ❌ sherpa-onnx 未安装")

    try:
        import dashscope
        print("  ✅ dashscope 已安装")
    except ImportError:
        missing_deps.append("dashscope")
        print("  ❌ dashscope 未安装")

    try:
        from pipecat.frames.frames import Frame
        print("  ✅ pipecat-ai 已安装")
    except ImportError:
        missing_deps.append("pipecat-ai[local,moondream]")
        print("  ❌ pipecat-ai 未安装")

    if missing_deps:
        print("\n❌ 缺少以下依赖：")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n💡 请运行以下命令安装：")
        print(f"  uv pip install {' '.join(missing_deps)}")
        print("  或者：")
        print("  uv sync")
        return False

    print("✅ 所有依赖已安装\n")
    return True


def check_env_file():
    """检查环境变量配置"""
    print("🔍 检查环境变量配置...")

    env_file = project_root / ".env"

    if not env_file.exists():
        print(f"  ❌ .env 文件不存在")
        print(f"  💡 请复制 .env.example 为 .env 并配置必要的参数")
        return False

    print("  ✅ .env 文件存在")

    # 检查关键配置
    try:
        from dotenv import load_dotenv
        import os

        load_dotenv(env_file)

        required_vars = {
            "DASHSCOPE_API_KEY": "Qwen LLM API Key",
            "ALIYUN_APPKEY": "Piper TTS AppKey",
        }

        missing_vars = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"{var} ({description})")
                print(f"  ⚠️ {var} 未配置")
            else:
                print(f"  ✅ {var} 已配置")

        if missing_vars:
            print("\n⚠️ 以下环境变量未配置（可能影响功能）：")
            for var in missing_vars:
                print(f"  - {var}")
            print("\n💡 请在 .env 文件中配置这些变量")
            # 不强制返回 False，允许继续运行
    except Exception as e:
        print(f"  ⚠️ 环境变量检查失败: {e}")

    print()
    return True


def check_models():
    """检查模型文件"""
    print("🔍 检查模型文件...")

    models_dir = project_root / "models"

    if not models_dir.exists():
        print(f"  ⚠️ models 目录不存在: {models_dir}")
        print(f"  💡 请参考 README.md 下载必要的模型文件")
        return False

    # 检查关键模型
    models = {
        "KWS 模型": models_dir / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
        "ASR 模型": models_dir / "sherpa-onnx-paraformer-zh-2023-09-14",
        "Piper TTS 模型": models_dir / "zh_CN-huayan-medium.onnx",
    }

    missing_models = []
    for name, path in models.items():
        if path.exists():
            print(f"  ✅ {name}: {path.name}")
        else:
            missing_models.append(name)
            print(f"  ❌ {name}: {path.name} (不存在)")

    if missing_models:
        print("\n❌ 缺少以下模型：")
        for model in missing_models:
            print(f"  - {model}")
        print("\n💡 请参考 README.md 的 '模型下载' 章节")
        return False

    print("✅ 所有模型文件已就绪\n")
    return True


async def main():
    """主函数"""
    print("=" * 70)
    print("🚀 Pipecat Flows 版本快速启动")
    print("=" * 70)
    print()

    # 1. 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 2. 检查环境变量
    if not check_env_file():
        sys.exit(1)

    # 3. 检查模型
    if not check_models():
        sys.exit(1)

    # 4. 启动主程序
    print("=" * 70)
    print("🎬 启动 Pipecat Flows 集成版语音助手...")
    print("=" * 70)
    print()

    try:
        from voice_assistant.pipecat_flows_main import main as flows_main
        await flows_main()
    except KeyboardInterrupt:
        print("\n👋 收到退出信号")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
