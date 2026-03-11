"""
Unit Tests for SkillRegistry

测试技能注册表的核心功能。
"""
import asyncio
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_skills_dir():
    """创建临时技能目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir)

        # 创建测试技能
        _create_test_skill(skills_dir, "weather", "天气查询", "查询全球各地的天气信息")
        _create_test_skill(skills_dir, "calendar", "日历管理", "管理日程和提醒")
        _create_test_skill(skills_dir, "calculator", "计算器", "执行数学计算")

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

## 功能

- 功能1
- 功能2
""", encoding="utf-8")


@pytest.mark.asyncio
async def test_registry_discover_all(temp_skills_dir):
    """测试发现所有技能"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    assert len(registry.skills) == 3
    assert "weather" in registry.skills
    assert "calendar" in registry.skills
    assert "calculator" in registry.skills


@pytest.mark.asyncio
async def test_registry_get_skill(temp_skills_dir):
    """测试获取技能"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    weather_skill = registry.get_skill("weather")
    assert weather_skill is not None
    assert weather_skill.metadata.name == "weather"
    assert weather_skill.metadata.display_name == "天气查询"


@pytest.mark.asyncio
async def test_registry_get_skill_not_found(temp_skills_dir):
    """测试获取不存在的技能"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    skill = registry.get_skill("nonexistent")
    assert skill is None


@pytest.mark.asyncio
async def test_registry_is_active(temp_skills_dir):
    """测试检查激活状态"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    # 初始状态：未激活
    assert not registry.is_active("weather")

    # 标记为激活
    registry.mark_active("weather")
    assert registry.is_active("weather")

    # 标记为未激活
    registry.mark_inactive("weather")
    assert not registry.is_active("weather")


@pytest.mark.asyncio
async def test_registry_get_active_skills(temp_skills_dir):
    """测试获取激活的技能"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    # 激活两个技能
    registry.mark_active("weather")
    registry.mark_active("calendar")

    active_skills = registry.get_active_skills()
    assert len(active_skills) == 2

    active_names = [s.metadata.name for s in active_skills]
    assert "weather" in active_names
    assert "calendar" in active_names


@pytest.mark.asyncio
async def test_registry_list_all_skills(temp_skills_dir):
    """测试列出所有技能"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    all_skills = registry.list_all_skills()
    assert len(all_skills) == 3
    assert "weather" in all_skills
    assert "calendar" in all_skills
    assert "calculator" in all_skills


@pytest.mark.asyncio
async def test_registry_list_active_skill_names(temp_skills_dir):
    """测试列出激活的技能名称"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    registry.mark_active("weather")
    registry.mark_active("calendar")

    active_names = registry.list_active_skill_names()
    assert len(active_names) == 2
    assert "weather" in active_names
    assert "calendar" in active_names


@pytest.mark.asyncio
async def test_registry_list_inactive_skill_names(temp_skills_dir):
    """测试列出未激活的技能名称"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    registry.mark_active("weather")

    inactive_names = registry.list_inactive_skill_names()
    assert len(inactive_names) == 2
    assert "calendar" in inactive_names
    assert "calculator" in inactive_names


@pytest.mark.asyncio
async def test_registry_get_skill_metadata(temp_skills_dir):
    """测试获取技能元数据"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    metadata = registry.get_skill_metadata("weather")
    assert metadata is not None
    assert metadata["name"] == "weather"
    assert metadata["display_name"] == "天气查询"
    assert metadata["description"] == "查询全球各地的天气信息"


@pytest.mark.asyncio
async def test_registry_filter_by_category(temp_skills_dir):
    """测试按类别过滤"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    utility_skills = registry.filter_by_category("utility")
    assert len(utility_skills) == 3


@pytest.mark.asyncio
async def test_registry_filter_by_tag(temp_skills_dir):
    """测试按标签过滤"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    test_skills = registry.filter_by_tag("test")
    assert len(test_skills) == 3


@pytest.mark.asyncio
async def test_registry_search_skills(temp_skills_dir):
    """测试搜索技能"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    # 搜索 "天气"
    results = registry.search_skills("天气")
    assert "weather" in results

    # 搜索 "计算"
    results = registry.search_skills("计算")
    assert "calculator" in results


@pytest.mark.asyncio
async def test_registry_get_status(temp_skills_dir):
    """测试获取状态"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    registry.mark_active("weather")
    registry.mark_active("calendar")

    status = await registry.get_status()
    assert status["total_skills"] == 3
    assert status["active_skills"] == 2
    assert status["inactive_skills"] == 1
    assert "weather" in status["active_skill_names"]


@pytest.mark.asyncio
async def test_registry_repr(temp_skills_dir):
    """测试字符串表示"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()

    registry.mark_active("weather")
    registry.mark_active("calendar")

    repr_str = repr(registry)
    assert "SkillRegistry" in repr_str
    assert "total=3" in repr_str
    assert "active=2" in repr_str


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
