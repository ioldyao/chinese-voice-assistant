"""
Anthropic LLM Service 测试

测试 Anthropic Claude Messages API 集成。

使用方法：
1. 设置环境变量 ANTHROPIC_API_KEY
2. 运行测试：pytest tests/test_anthropic_llm.py -v
"""
import asyncio
import os
from pathlib import Path

import pytest

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root / "src"))

from voice_assistant.anthropic_llm_service import AnthropicLLMService
from voice_assistant.config import ANTHROPIC_API_KEY


@pytest.mark.asyncio
async def test_anthropic_llm_instantiation():
    """测试 Anthropic LLM Service 初始化"""
    if not ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY 未设置")

    llm = AnthropicLLMService(
        api_key=ANTHROPIC_API_KEY,
        model="claude-3-5-haiku-20241022"  # 使用便宜的模型测试
    )

    assert llm is not None
    assert llm._model_name == "claude-3-5-haiku-20241022"
    print("✓ Anthropic LLM Service 初始化成功")


@pytest.mark.asyncio
async def test_anthropic_message_conversion():
    """测试消息格式转换"""
    if not ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY 未设置")

    llm = AnthropicLLMService(
        api_key=ANTHROPIC_API_KEY,
        model="claude-3-5-haiku-20241022"
    )

    # OpenAI 格式消息
    openai_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]

    # 转换为 Anthropic 格式
    system, anthropic_messages, tools = \
        await llm._convert_to_anthropic_format(openai_messages)

    # 验证转换结果
    assert system == "You are a helpful assistant."
    assert len(anthropic_messages) == 1
    assert anthropic_messages[0]["role"] == "user"
    assert anthropic_messages[0]["content"] == "Hello!"
    print("✓ 消息格式转换成功")


@pytest.mark.asyncio
async def test_anthropic_stream_chat():
    """测试流式对话（需要 API 调用）"""
    if not ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY 未设置")

    llm = AnthropicLLMService(
        api_key=ANTHROPIC_API_KEY,
        model="claude-3-5-haiku-20241022"
    )

    messages = [
        {"role": "user", "content": "Say 'Hello, Anthropic!' in Chinese."}
    ]

    # 收集流式输出
    collected_text = []
    async for text_chunk in llm.stream_chat_completion(messages):
        collected_text.append(text_chunk)

    result = "".join(collected_text)

    # 验证输出包含预期内容
    assert len(result) > 0
    assert "你好" in result or "Hello" in result
    print(f"✓ 流式对话成功: {result}")


if __name__ == "__main__":
    # 直接运行测试
    print("🧪 测试 Anthropic LLM Service\n")

    asyncio.run(test_anthropic_llm_instantiation())
    asyncio.run(test_anthropic_message_conversion())
    asyncio.run(test_anthropic_stream_chat())

    print("\n✅ 所有测试通过！")
