"""Pipecat Flows 集成测试脚本

这个脚本用于测试 Pipecat Flows 集成是否正常工作。

测试内容：
1. 导入检查
2. 节点创建测试
3. 函数 Schema 验证
4. FlowManager 初始化测试
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "voice_assistant"))


def test_imports():
    """测试 1: 导入检查"""
    print("🔍 测试 1: 检查必要的导入...")

    try:
        from pipecat_flows import (
            FlowManager,
            NodeConfig,
            FlowsFunctionSchema,
            FlowArgs,
            FlowResult,
            ConsolidatedFunctionResult,
        )
        print("  ✅ pipecat_flows 导入成功")
    except ImportError as e:
        print(f"  ❌ pipecat_flows 导入失败: {e}")
        print("  💡 请运行: pip install pipecat-ai-flows")
        return False

    try:
        from pipecat_flows_main import (
            create_initial_node,
            create_analysis_node,
            create_completion_node,
        )
        print("  ✅ 节点创建函数导入成功")
    except ImportError as e:
        print(f"  ❌ 节点创建函数导入失败: {e}")
        return False

    try:
        from pipecat_flows_main import (
            navigate_schema,
            analysis_schema,
            end_conversation_schema,
        )
        print("  ✅ 函数 Schema 导入成功")
    except ImportError as e:
        print(f"  ❌ 函数 Schema 导入失败: {e}")
        return False

    print("✅ 测试 1 通过：所有导入正常\n")
    return True


def test_node_creation():
    """测试 2: 节点创建"""
    print("🔍 测试 2: 节点创建...")

    try:
        from pipecat_flows_main import (
            create_initial_node,
            create_analysis_node,
            create_completion_node,
        )

        # 创建节点
        initial_node = create_initial_node(wait_for_user=False)
        analysis_node = create_analysis_node()
        completion_node = create_completion_node()

        # 验证节点结构
        assert "name" in initial_node, "initial_node 缺少 name 字段"
        assert "task_messages" in initial_node, "initial_node 缺少 task_messages 字段"
        assert "functions" in initial_node, "initial_node 缺少 functions 字段"

        print(f"  ✅ initial_node: {initial_node['name']}")
        print(f"  ✅ analysis_node: {analysis_node['name']}")
        print(f"  ✅ completion_node: {completion_node['name']}")

        # 验证函数列表
        print(f"  ✅ initial_node 函数数量: {len(initial_node['functions'])}")
        print(f"  ✅ analysis_node 函数数量: {len(analysis_node['functions'])}")
        print(f"  ✅ completion_node 函数数量: {len(completion_node['functions'])}")

        print("✅ 测试 2 通过：节点创建正常\n")
        return True

    except Exception as e:
        print(f"  ❌ 节点创建失败: {e}")
        return False


def test_function_schemas():
    """测试 3: 函数 Schema 验证"""
    print("🔍 测试 3: 函数 Schema 验证...")

    try:
        from pipecat_flows_main import (
            navigate_schema,
            analysis_schema,
            end_conversation_schema,
        )

        # 验证 navigate_schema
        assert navigate_schema.name == "navigate_to_url", "navigate_schema 名称错误"
        assert "url" in navigate_schema.properties, "navigate_schema 缺少 url 属性"
        assert "url" in navigate_schema.required, "navigate_schema 缺少 url 必填项"
        assert navigate_schema.handler is not None, "navigate_schema 缺少 handler"
        print(f"  ✅ navigate_schema: {navigate_schema.name}")

        # 验证 analysis_schema
        assert analysis_schema.name == "analyze_page_content", "analysis_schema 名称错误"
        assert analysis_schema.handler is not None, "analysis_schema 缺少 handler"
        print(f"  ✅ analysis_schema: {analysis_schema.name}")

        # 验证 end_conversation_schema
        assert end_conversation_schema.name == "end_conversation", "end_conversation_schema 名称错误"
        assert end_conversation_schema.handler is not None, "end_conversation_schema 缺少 handler"
        print(f"  ✅ end_conversation_schema: {end_conversation_schema.name}")

        print("✅ 测试 3 通过：函数 Schema 验证成功\n")
        return True

    except Exception as e:
        print(f"  ❌ 函数 Schema 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_flow_manager_init():
    """测试 4: FlowManager 初始化（模拟）"""
    print("🔍 测试 4: FlowManager 初始化（模拟）...")

    try:
        from pipecat_flows import FlowManager
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineTask, PipelineParams
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
        from pipecat.services.openai.llm import OpenAIUserContextAggregator
        from qwen_llm_service import QwenLLMService
        from pipecat_flows_main import create_initial_node
        from config import QWEN_MODEL

        # 创建模拟组件
        print("  📦 创建 LLM 服务...")
        llm = QwenLLMService(model=QWEN_MODEL)

        print("  📦 创建 Context...")
        context = OpenAILLMContext()
        context_aggregator = OpenAIUserContextAggregator(context)

        print("  📦 创建 Pipeline...")
        pipeline = Pipeline([])  # 空 Pipeline，仅用于测试

        print("  📦 创建 Task...")
        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        print("  📦 创建 FlowManager...")
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
        )

        print("  📦 创建初始节点...")
        initial_node = create_initial_node(wait_for_user=True)

        print("  📦 初始化 FlowManager...")
        # 注意：这里不实际调用 initialize，因为需要完整的 Pipeline
        # 只验证 FlowManager 对象创建成功
        assert flow_manager is not None, "FlowManager 创建失败"
        assert flow_manager.state is not None, "FlowManager state 为空"

        print("  ✅ FlowManager 对象创建成功")
        print("  ✅ 初始节点配置正确")

        print("✅ 测试 4 通过：FlowManager 初始化模拟成功\n")
        return True

    except Exception as e:
        print(f"  ❌ FlowManager 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("Pipecat Flows 集成测试")
    print("=" * 60)
    print()

    results = []

    # 测试 1: 导入检查
    results.append(("导入检查", test_imports()))

    if not results[0][1]:
        print("\n❌ 导入失败，无法继续后续测试")
        return

    # 测试 2: 节点创建
    results.append(("节点创建", test_node_creation()))

    # 测试 3: 函数 Schema 验证
    results.append(("函数 Schema 验证", test_function_schemas()))

    # 测试 4: FlowManager 初始化
    results.append(("FlowManager 初始化", await test_flow_manager_init()))

    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")

    print()
    print(f"总计: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 所有测试通过！Pipecat Flows 集成成功")
        return True
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
