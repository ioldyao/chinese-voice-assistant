"""测试 Anthropic LLM 服务集成"""
import asyncio
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair


async def test_anthropic_service():
    """测试 Anthropic LLM 服务"""

    print("=" * 60)
    print("测试: 官方 Claude API")
    print("=" * 60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("跳过：ANTHROPIC_API_KEY 未设置")
        return

    try:
        # 创建 Anthropic LLM 服务
        llm = AnthropicLLMService(
            api_key=api_key,
            model="claude-sonnet-4-5-20250929"
        )

        print("OK: AnthropicLLMService created")
        print(f"  - Model: claude-sonnet-4-5-20250929")

        # 创建 Context
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        context = LLMContext(messages=messages)

        print("OK: LLMContext created")

        # 创建 Aggregators（正确方式）
        aggregator_pair = LLMContextAggregatorPair(context)
        user_agg = aggregator_pair.user()
        assistant_agg = aggregator_pair.assistant()

        print("OK: LLMContextAggregatorPair created")
        print(f"  - User aggregator: {type(user_agg).__name__}")
        print(f"  - Assistant aggregator: {type(assistant_agg).__name__}")
        print("\nAll tests passed!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_anthropic_service())
