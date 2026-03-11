"""
Unit Tests for SkillIntegrationLayer

测试技能集成层的核心功能。
"""
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest


@pytest.fixture
def temp_skills_dir():
    """创建临时技能目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir)

        # 创建测试技能
        _create_test_skill(
            skills_dir,
            "weather",
            "天气查询",
            "查询全球各地的天气信息",
        )
        _create_test_skill(
            skills_dir,
            "calculator",
            "计算器",
            "执行数学计算",
        )

        yield skills_dir


def _create_test_skill(skills_dir: Path, name: str, display_name: str, description: str):
    """创建测试技能目录和 SKILL.md"""
    skill_dir = skills_dir / name
    skill_dir.mkdir()

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"""---
name: {name}
display_name: {display_name}
description: {description}
version: 1.0.0
author: Test Author
tags: [test, utility]
category: utility
requires_tools: []
---

# {display_name}

这是 {display_name} 的技能说明。

## 使用方法

当用户询问相关问题时，使用此技能。
""", encoding="utf-8")


@pytest.fixture
def mock_llm_service():
    """创建模拟的 LLM 服务"""
    llm = MagicMock()
    llm.register_function = MagicMock()
    return llm


@pytest.fixture
def mock_task():
    """创建模拟的 PipelineTask"""
    task = MagicMock()
    task.queue_frames = AsyncMock()
    return task


@pytest.mark.asyncio
async def test_integration_layer_initialize(temp_skills_dir, mock_llm_service):
    """测试初始化"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry
    from src.voice_assistant.skills.skill_matcher import SkillMatcher
    from src.voice_assistant.skills.skill_manager import SkillManager
    from src.voice_assistant.skills.skill_integration_layer import SkillIntegrationLayer

    # 创建组件
    registry = SkillRegistry(temp_skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(temp_skills_dir)

    # 创建集成层
    layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=mock_llm_service,
    )

    # 初始化
    await layer.initialize()

    # 验证
    assert len(registry.skills) == 2
    assert "weather" in registry.skills
    assert "calculator" in registry.skills


@pytest.mark.asyncio
async def test_integration_layer_process_user_input(temp_skills_dir, mock_llm_service, mock_task):
    """测试处理用户输入"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry
    from src.voice_assistant.skills.skill_matcher import SkillMatcher
    from src.voice_assistant.skills.skill_manager import SkillManager
    from src.voice_assistant.skills.skill_integration_layer import SkillIntegrationLayer

    # 创建组件
    registry = SkillRegistry(temp_skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(temp_skills_dir)

    # 创建集成层
    layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=mock_llm_service,
    )

    # 初始化
    await layer.initialize()

    # 设置 task
    layer.task = mock_task

    # 处理用户输入
    await layer.process_user_input("今天天气怎么样")

    # 验证：应该激活 weather 技能
    assert registry.is_active("weather")


@pytest.mark.asyncio
async def test_integration_layer_manually_activate_skill(temp_skills_dir, mock_llm_service, mock_task):
    """测试手动激活技能"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry
    from src.voice_assistant.skills.skill_matcher import SkillMatcher
    from src.voice_assistant.skills.skill_manager import SkillManager
    from src.voice_assistant.skills.skill_integration_layer import SkillIntegrationLayer

    # 创建组件
    registry = SkillRegistry(temp_skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(temp_skills_dir)

    # 创建集成层
    layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=mock_llm_service,
    )

    # 初始化
    await layer.initialize()

    # 设置 task
    layer.task = mock_task

    # 手动激活技能
    success = await layer.manually_activate_skill("weather")

    # 验证
    assert success
    assert registry.is_active("weather")


@pytest.mark.asyncio
async def test_integration_layer_deactivate_skill(temp_skills_dir, mock_llm_service):
    """测试停用技能"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry
    from src.voice_assistant.skills.skill_matcher import SkillMatcher
    from src.voice_assistant.skills.skill_manager import SkillManager
    from src.voice_assistant.skills.skill_integration_layer import SkillIntegrationLayer

    # 创建组件
    registry = SkillRegistry(temp_skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(temp_skills_dir)

    # 创建集成层
    layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=mock_llm_service,
    )

    # 初始化
    await layer.initialize()

    # 激活技能
    await layer.manually_activate_skill("weather")
    assert registry.is_active("weather")

    # 停用技能
    await layer.deactivate_skill("weather")
    assert not registry.is_active("weather")


@pytest.mark.asyncio
async def test_integration_layer_deactivate_all_skills(temp_skills_dir, mock_llm_service):
    """测试停用所有技能"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry
    from src.voice_assistant.skills.skill_matcher import SkillMatcher
    from src.voice_assistant.skills.skill_manager import SkillManager
    from src.voice_assistant.skills.skill_integration_layer import SkillIntegrationLayer

    # 创建组件
    registry = SkillRegistry(temp_skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(temp_skills_dir)

    # 创建集成层
    layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=mock_llm_service,
    )

    # 初始化
    await layer.initialize()

    # 激活所有技能
    await layer.manually_activate_skill("weather")
    await layer.manually_activate_skill("calculator")

    # 停用所有技能
    await layer.deactivate_all_skills()

    # 验证
    assert len(registry.active_skills) == 0


@pytest.mark.asyncio
async def test_integration_layer_set_auto_activate(temp_skills_dir, mock_llm_service):
    """测试设置自动激活"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry
    from src.voice_assistant.skills.skill_matcher import SkillMatcher
    from src.voice_assistant.skills.skill_manager import SkillManager
    from src.voice_assistant.skills.skill_integration_layer import SkillIntegrationLayer

    # 创建组件
    registry = SkillRegistry(temp_skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(temp_skills_dir)

    # 创建集成层
    layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=mock_llm_service,
    )

    # 禁用自动激活
    layer.set_auto_activate(False)

    # 初始化
    await layer.initialize()

    # 处理用户输入
    await layer.process_user_input("今天天气怎么样")

    # 验证：不应该自动激活
    assert not registry.is_active("weather")


@pytest.mark.asyncio
async def test_integration_layer_set_match_threshold(temp_skills_dir, mock_llm_service):
    """测试设置匹配阈值"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry
    from src.voice_assistant.skills.skill_matcher import SkillMatcher
    from src.voice_assistant.skills.skill_manager import SkillManager
    from src.voice_assistant.skills.skill_integration_layer import SkillIntegrationLayer

    # 创建组件
    registry = SkillRegistry(temp_skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(temp_skills_dir)

    # 创建集成层
    layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=mock_llm_service,
    )

    # 设置高阈值
    layer.set_match_threshold(0.9)

    # 验证
    assert layer.match_threshold == 0.9


@pytest.mark.asyncio
async def test_integration_layer_set_max_active_skills(temp_skills_dir, mock_llm_service):
    """测试设置最大激活技能数"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry
    from src.voice_assistant.skills.skill_matcher import SkillMatcher
    from src.voice_assistant.skills.skill_manager import SkillManager
    from src.voice_assistant.skills.skill_integration_layer import SkillIntegrationLayer

    # 创建组件
    registry = SkillRegistry(temp_skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(temp_skills_dir)

    # 创建集成层
    layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=mock_llm_service,
    )

    # 设置最大激活数
    layer.set_max_active_skills(5)

    # 验证
    assert layer.max_active_skills == 5


@pytest.mark.asyncio
async def test_integration_layer_get_status(temp_skills_dir, mock_llm_service):
    """测试获取状态"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry
    from src.voice_assistant.skills.skill_matcher import SkillMatcher
    from src.voice_assistant.skills.skill_manager import SkillManager
    from src.voice_assistant.skills.skill_integration_layer import SkillIntegrationLayer

    # 创建组件
    registry = SkillRegistry(temp_skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(temp_skills_dir)

    # 创建集成层
    layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=mock_llm_service,
    )

    # 初始化
    await layer.initialize()

    # 获取状态
    status = await layer.get_status()

    # 验证
    assert "registry" in status
    assert "matcher" in status
    assert "skill_manager" in status
    assert "config" in status
    assert status["registry"]["total_skills"] == 2


@pytest.mark.asyncio
async def test_integration_layer_repr(temp_skills_dir, mock_llm_service):
    """测试字符串表示"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry
    from src.voice_assistant.skills.skill_matcher import SkillMatcher
    from src.voice_assistant.skills.skill_manager import SkillManager
    from src.voice_assistant.skills.skill_integration_layer import SkillIntegrationLayer

    # 创建组件
    registry = SkillRegistry(temp_skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(temp_skills_dir)

    # 创建集成层
    layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=mock_llm_service,
    )

    # 初始化
    await layer.initialize()

    # 获取字符串表示
    repr_str = repr(layer)

    # 验证
    assert "SkillIntegrationLayer" in repr_str
    assert "skills=2" in repr_str


@pytest.mark.asyncio
async def test_integration_layer_set_task(temp_skills_dir, mock_llm_service, mock_task):
    """测试设置 task"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry
    from src.voice_assistant.skills.skill_matcher import SkillMatcher
    from src.voice_assistant.skills.skill_manager import SkillManager
    from src.voice_assistant.skills.skill_integration_layer import SkillIntegrationLayer

    # 创建组件
    registry = SkillRegistry(temp_skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(temp_skills_dir)

    # 创建集成层
    layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=mock_llm_service,
    )

    # 初始化
    await layer.initialize()

    # 设置 task
    layer.set_task(mock_task)

    # 验证
    assert layer.task == mock_task
    assert skill_manager.task == mock_task


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
