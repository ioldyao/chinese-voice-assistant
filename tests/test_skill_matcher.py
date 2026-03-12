"""
Unit Tests for SkillMatcher

测试技能匹配器的核心功能。
"""
import tempfile
from pathlib import Path

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
            "查询全球各地的天气信息，包括温度、湿度、风力等",
            tags=["weather", "api"]
        )
        _create_test_skill(
            skills_dir,
            "calculator",
            "计算器",
            "执行数学计算，支持加减乘除和复杂运算",
            tags=["math", "utility"]
        )
        _create_test_skill(
            skills_dir,
            "calendar",
            "日历管理",
            "管理日程安排和提醒事项",
            tags=["productivity", "schedule"]
        )

        yield skills_dir


def _create_test_skill(skills_dir: Path, name: str, display_name: str, description: str, tags: list[str] = None):
    """创建测试技能目录和 SKILL.md"""
    skill_dir = skills_dir / name
    skill_dir.mkdir()

    tags_str = str(tags) if tags else "[]"

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"""---
name: {name}
display_name: {display_name}
description: {description}
version: 1.0.0
author: Test Author
tags: {tags_str}
category: utility
requires_tools: []
---

# {display_name}

这是 {display_name} 的技能说明。
""", encoding="utf-8")


@pytest.fixture
async def registry(temp_skills_dir):
    """创建技能注册表"""
    from src.voice_assistant.skills.skill_registry import SkillRegistry

    registry = SkillRegistry(temp_skills_dir)
    await registry.discover_all()
    return registry


@pytest.fixture
def matcher(registry):
    """创建技能匹配器"""
    from src.voice_assistant.skills.skill_matcher import SkillMatcher

    return SkillMatcher(registry)


@pytest.mark.asyncio
async def test_matcher_exact_name_match(matcher):
    """测试精确名称匹配"""
    # 精确匹配 "weather"
    results = await matcher.match_skills("weather")

    assert len(results) > 0
    assert results[0].skill_name == "weather"
    assert results[0].score == 1.0  # 精确匹配得满分


@pytest.mark.asyncio
async def test_matcher_keyword_match(matcher):
    """测试关键词匹配"""
    # 匹配 "天气"（在显示名称和描述中）
    results = await matcher.match_skills("今天天气怎么样")

    assert len(results) > 0
    assert results[0].skill_name == "weather"
    assert results[0].score > 0.5


@pytest.mark.asyncio
async def test_matcher_multiple_skills(matcher):
    """测试多技能匹配"""
    # 匹配多个相关技能
    results = await matcher.match_skills("管理日程")

    assert len(results) > 0
    # 应该匹配到 calendar
    skill_names = [r.skill_name for r in results]
    assert "calendar" in skill_names


@pytest.mark.asyncio
async def test_matcher_threshold(matcher):
    """测试阈值过滤"""
    # 设置高阈值
    results = await matcher.match_skills("xyz", threshold=0.8)

    # 应该没有匹配或很少匹配
    assert len(results) <= 1


@pytest.mark.asyncio
async def test_matcher_top_k(matcher):
    """测试 Top-K 限制"""
    # 请求最多 2 个结果
    results = await matcher.match_skills("管理", top_k=2)

    assert len(results) <= 2


@pytest.mark.asyncio
async def test_matcher_cache(matcher):
    """测试缓存功能"""
    # 第一次匹配
    results1 = await matcher.match_skills("天气")

    # 第二次匹配（应该从缓存读取）
    results2 = await matcher.match_skills("天气")

    assert len(results1) == len(results2)
    assert matcher.get_cache_size() > 0


@pytest.mark.asyncio
async def test_matcher_clear_cache(matcher):
    """测试清空缓存"""
    # 产生一些缓存
    await matcher.match_skills("天气")
    assert matcher.get_cache_size() > 0

    # 清空缓存
    matcher.clear_cache()
    assert matcher.get_cache_size() == 0


@pytest.mark.asyncio
async def test_matcher_weights(matcher):
    """测试权重设置"""
    # 获取默认权重
    weights = matcher.get_weights()
    assert "exact_name_match" in weights
    assert "keyword_in_name" in weights

    # 设置新权重
    new_weights = {"exact_name_match": 0.9}
    matcher.set_weights(new_weights)

    # 验证权重已更新
    updated_weights = matcher.get_weights()
    assert updated_weights["exact_name_match"] == 0.9


@pytest.mark.asyncio
async def test_matcher_no_match(registry):
    """测试无匹配情况"""
    from src.voice_assistant.skills.skill_matcher import SkillMatcher

    matcher = SkillMatcher(registry)

    # 使用不相关的关键词
    results = await matcher.match_skills("xyzabc", threshold=0.5)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_matcher_get_status(matcher):
    """测试获取状态"""
    status = await matcher.get_status()

    assert "registry_total_skills" in status
    assert status["registry_total_skills"] == 3
    assert "enable_cache" in status
    assert "weights" in status


@pytest.mark.asyncio
async def test_matcher_repr(matcher):
    """测试字符串表示"""
    repr_str = repr(matcher)

    assert "SkillMatcher" in repr_str
    assert "skills=3" in repr_str


@pytest.mark.asyncio
async def test_matcher_extract_keywords(matcher):
    """测试关键词提取"""
    keywords = matcher._extract_keywords("今天北京天气怎么样")

    # 应该提取出有意义的关键词
    assert len(keywords) > 0
    # 停用词应该被过滤
    assert "怎么样" not in keywords


@pytest.mark.asyncio
async def test_matcher_calculate_score(matcher):
    """测试评分计算"""
    registry = matcher.registry
    skill = registry.get_skill("weather")

    # 测试精确匹配
    score, reason = matcher._calculate_match_score("weather", skill)
    assert score == 1.0
    assert "Exact" in reason

    # 测试关键词匹配
    score, reason = matcher._calculate_match_score("今天天气", skill)
    assert score > 0.5


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
