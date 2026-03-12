"""
Unit Tests for Progressive Disclosure (Claude Code Standard)

测试渐进式披露功能：
- references/ 目录扫描
- examples/ 目录扫描
- scripts/ 目录扫描
- 资源文件加载
"""
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_skill_with_resources():
    """创建包含资源的临时技能目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir)

        # 创建技能目录结构
        skill_dir = skills_dir / "weather"
        skill_dir.mkdir()
        (skill_dir / "references").mkdir()
        (skill_dir / "examples").mkdir()
        (skill_dir / "scripts").mkdir()

        # 创建 SKILL.md
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: weather
display_name: 天气查询
description: This skill should be used when the user asks to "查询天气", "今天天气怎么样", "天气如何"
version: 1.0.0
author: Test Author
tags: [weather, api]
category: utility
requires_tools: []
---

# 天气查询技能

提供全球天气信息查询功能。

## Additional Resources

### Reference Files
- **`references/api-docs.md`** - API 文档

### Examples
- **`examples/query.py`** - 查询示例
""", encoding="utf-8")

        # 创建 references 文件
        (skill_dir / "references" / "api-docs.md").write_text(
            "# API 文档\n\n详细的 API 使用说明。", encoding="utf-8"
        )
        (skill_dir / "references" / "errors.md").write_text(
            "# 错误处理\n\n常见错误及解决方案。", encoding="utf-8"
        )

        # 创建 examples 文件
        (skill_dir / "examples" / "query.py").write_text(
            "# 查询示例\nprint('Weather query')", encoding="utf-8"
        )
        (skill_dir / "examples" / "response.json").write_text(
            '{"temperature": 25}', encoding="utf-8"
        )

        # 创建 scripts 文件
        (skill_dir / "scripts" / "test.sh").write_text(
            "#!/bin/bash\necho 'Test script'", encoding="utf-8"
        )

        yield skills_dir


@pytest.mark.asyncio
async def test_progressive_disclosure_metadata(temp_skill_with_resources):
    """测试元数据包含渐进式资源列表"""
    from src.voice_assistant.skills.base_skill import AgentSkill

    skill_dir = temp_skill_with_resources / "weather"
    skill = AgentSkill(skill_dir)

    # 验证元数据包含资源列表
    assert skill.metadata is not None
    assert len(skill.metadata.references_files) == 2
    assert len(skill.metadata.examples_files) == 2
    assert len(skill.metadata.scripts_files) == 1

    # 验证文件路径格式
    assert any("api-docs.md" in f for f in skill.metadata.references_files)
    assert any("query.py" in f for f in skill.metadata.examples_files)
    assert any("test.sh" in f for f in skill.metadata.scripts_files)


@pytest.mark.asyncio
async def test_load_reference_file(temp_skill_with_resources):
    """测试加载 references 文件"""
    from src.voice_assistant.skills.base_skill import AgentSkill

    skill_dir = temp_skill_with_resources / "weather"
    skill = AgentSkill(skill_dir)

    # 加载 reference 文件
    content = await skill.load_reference("api-docs.md")

    assert "API 文档" in content
    assert "详细的 API 使用说明" in content


@pytest.mark.asyncio
async def test_load_example_file(temp_skill_with_resources):
    """测试加载 examples 文件"""
    from src.voice_assistant.skills.base_skill import AgentSkill

    skill_dir = temp_skill_with_resources / "weather"
    skill = AgentSkill(skill_dir)

    # 加载 example 文件
    content = await skill.load_example("query.py")

    assert "查询示例" in content
    assert "print" in content


@pytest.mark.asyncio
async def test_get_script_path(temp_skill_with_resources):
    """测试获取脚本路径"""
    from src.voice_assistant.skills.base_skill import AgentSkill

    skill_dir = temp_skill_with_resources / "weather"
    skill = AgentSkill(skill_dir)

    # 获取脚本路径
    script_path = skill.get_script_path("test.sh")

    assert script_path.exists()
    assert script_path.name == "test.sh"
    assert "scripts" in str(script_path)


@pytest.mark.asyncio
async def test_has_progressive_resources(temp_skill_with_resources):
    """测试检查是否有渐进式资源"""
    from src.voice_assistant.skills.base_skill import AgentSkill

    skill_dir = temp_skill_with_resources / "weather"
    skill = AgentSkill(skill_dir)

    # 验证 has_progressive_resources 属性
    assert skill.metadata.has_progressive_resources is True


@pytest.mark.asyncio
async def test_skill_without_resources():
    """测试没有资源的技能"""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir)
        skill_dir = skills_dir / "simple"
        skill_dir.mkdir()

        # 只创建 SKILL.md，不创建资源目录
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: simple
display_name: 简单技能
description: This skill should be used when testing
version: 1.0.0
tags: [test]
category: utility
requires_tools: []
---

# 简单技能

没有额外资源的技能。
""", encoding="utf-8")

        from src.voice_assistant.skills.base_skill import AgentSkill

        skill = AgentSkill(skill_dir)

        # 验证资源列表为空
        assert len(skill.metadata.references_files) == 0
        assert len(skill.metadata.examples_files) == 0
        assert len(skill.metadata.scripts_files) == 0
        assert skill.metadata.has_progressive_resources is False


@pytest.mark.asyncio
async def test_get_resource_path_file_not_found():
    """测试获取不存在的资源文件"""
    from src.voice_assistant.skills.base_skill import AgentSkill

    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir)
        skill_dir = skills_dir / "weather"
        skill_dir.mkdir()
        (skill_dir / "references").mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: weather
display_name: 天气查询
description: This skill should be used when testing
version: 1.0.0
tags: [test]
category: utility
requires_tools: []
---

# 天气查询
""", encoding="utf-8")

        skill = AgentSkill(skill_dir)

        # 尝试获取不存在的文件
        with pytest.raises(FileNotFoundError):
            skill.get_resource_path("references", "nonexistent.md")


@pytest.mark.asyncio
async def test_metadata_to_dict_includes_resources():
    """测试元数据转换为字典包含资源列表"""
    from src.voice_assistant.skills.base_skill import AgentSkill

    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir)
        skill_dir = skills_dir / "weather"
        skill_dir.mkdir()
        (skill_dir / "references").mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: weather
display_name: 天气查询
description: This skill should be used when testing
version: 1.0.0
tags: [test]
category: utility
requires_tools: []
---

# 天气查询
""", encoding="utf-8")

        # 创建一个 reference 文件
        (skill_dir / "references" / "api.md").write_text("# API", encoding="utf-8")

        skill = AgentSkill(skill_dir)
        metadata_dict = skill.metadata.to_dict()

        # 验证字典包含资源字段
        assert "references_files" in metadata_dict
        assert "examples_files" in metadata_dict
        assert "scripts_files" in metadata_dict
        assert len(metadata_dict["references_files"]) == 1
