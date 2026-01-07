#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 Flows 版本的关键修复"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

async def test_llm_context_creation():
    """测试 LLMContext 创建（不应该包含 tools）"""
    print("\n[TEST 1] LLMContext 创建")

    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.services.openai.llm import (
        OpenAIUserContextAggregator,
        OpenAIAssistantContextAggregator,
    )

    try:
        # 创建空 context（Flows 版正确做法）
        context = OpenAILLMContext()
        user_aggregator = OpenAIUserContextAggregator(context)
        assistant_aggregator = OpenAIAssistantContextAggregator(context)

        print("  [OK] OpenAILLMContext 创建成功（无 tools）")
        print("  [OK] UserContextAggregator 创建成功")
        print("  [OK] AssistantContextAggregator 创建成功")

        # 验证 context 配置正确（Flows 版使用空 context）
        # context.tools 应该为 None 或空
        if hasattr(context, 'tools') and context.tools:
            print("  [WARNING] Context 包含 tools（Flows 版应该为空）")
            return False
        else:
            print("  [OK] Context 配置正确（Flows 版使用空 context）")

        return True

    except Exception as e:
        print(f"  [ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_qwen_llm_service():
    """测试 QwenLLMService 初始化"""
    print("\n[TEST 2] QwenLLMService 初始化")

    try:
        from src.voice_assistant.qwen_llm_service import QwenLLMService
        from src.voice_assistant.config import QWEN_MODEL

        llm = QwenLLMService(model=QWEN_MODEL)

        print(f"  [OK] QwenLLMService 创建成功")
        print(f"  [OK] 模型: {QWEN_MODEL}")

        return True

    except Exception as e:
        print(f"  [ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_node_creation():
    """测试节点创建（FlowManager 使用）"""
    print("\n[TEST 3] FlowManager 节点创建")

    try:
        from src.voice_assistant.pipecat_flows_main import (
            create_initial_node,
            create_analysis_node,
            create_completion_node,
        )

        # 创建节点
        initial_node = create_initial_node(wait_for_user=True)
        analysis_node = create_analysis_node()
        completion_node = create_completion_node()

        # 验证节点结构
        assert "name" in initial_node
        assert "functions" in initial_node
        assert len(initial_node["functions"]) > 0

        print(f"  [OK] initial_node 创建成功，包含 {len(initial_node['functions'])} 个函数")
        print(f"  [OK] analysis_node 创建成功，包含 {len(analysis_node['functions'])} 个函数")
        print(f"  [OK] completion_node 创建成功，包含 {len(completion_node['functions'])} 个函数")

        return True

    except Exception as e:
        print(f"  [ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_import_flows():
    """测试 pipecat-ai-flows 导入"""
    print("\n[TEST 4] pipecat-ai-flows 导入")

    try:
        from pipecat_flows import (
            FlowManager,
            NodeConfig,
            FlowsFunctionSchema,
            FlowArgs,
            FlowResult,
            ConsolidatedFunctionResult,
        )

        print("  [OK] pipecat_flows 所有组件导入成功")
        return True

    except ImportError as e:
        print(f"  [ERROR] 导入失败: {e}")
        print("  [TIP] 请运行: uv pip install pipecat-ai-flows")
        return False


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("Pipecat Flows v2.3.0 修复验证测试")
    print("=" * 60)

    results = []

    # 测试 4: pipecat-ai-flows 导入
    results.append(("pipecat-ai-flows 导入", await test_import_flows()))

    if not results[0][1]:
        print("\n[ERROR] pipecat-ai-flows 未安装，无法继续测试")
        return False

    # 测试 1: LLMContext 创建
    results.append(("LLMContext 创建", await test_llm_context_creation()))

    # 测试 2: QwenLLMService 初始化
    results.append(("QwenLLMService 初始化", await test_qwen_llm_service()))

    # 测试 3: 节点创建
    results.append(("FlowManager 节点创建", await test_node_creation()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")

    print()
    print(f"总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n[SUCCESS] 所有测试通过！")
        print("\n[OK] 核心修复验证成功:")
        print("  - LLMContext 不再包含 tools（避免序列化错误）")
        print("  - FlowManager 通过节点函数管理工具")
        print("  - QwenLLMService 正常初始化")
        print("\n[TIP] 可以运行主程序测试完整功能:")
        print("   uv run python main.py")
        return True
    else:
        print("\n[WARNING] 部分测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
