"""
Agent Skills - 技能加载器

负责从文件系统加载技能，支持批量加载和过滤。

功能：
- 扫描 skills 目录发现所有技能
- 批量加载技能元数据（Discovery 阶段）
- 支持按标签、类别过滤
- 自动验证 SKILL.md 文件格式

使用示例：
```python
from src.voice_assistant.skills.skill_loader import SkillLoader

loader = SkillLoader(Path("skills"))

# 发现所有技能（仅加载 metadata）
skills = loader.discover_all()
print(f"发现 {len(skills)} 个技能")

# 按类别过滤
utility_skills = loader.discover_by_category("utility")

# 按标签过滤
api_skills = loader.discover_by_tag("api")

# 获取特定技能
weather_skill = loader.load_skill("weather")
```
"""
from pathlib import Path
from typing import Optional

from .base_skill import AgentSkill


class SkillLoader:
    """
    技能加载器

    提供技能发现和加载功能，支持批量操作和过滤。

    Attributes:
        skills_dir: 技能根目录
        recursive: 是否递归搜索子目录

    使用示例：
    ```python
    # 基础用法
    loader = SkillLoader(Path("skills"))
    all_skills = loader.discover_all()

    # 递归搜索
    loader = SkillLoader(Path("skills"), recursive=True)
    all_skills = loader.discover_all()

    # 过滤
    weather_skill = loader.discover_by_name("weather")
    utility_skills = loader.discover_by_category("utility")
    ```
    """

    def __init__(self, skills_dir: Path, recursive: bool = False):
        """
        初始化技能加载器

        Args:
            skills_dir: 技能根目录
            recursive: 是否递归搜索子目录（默认 False）

        Raises:
            FileNotFoundError: skills_dir 不存在
        """
        if not skills_dir.exists():
            raise FileNotFoundError(f"Skills directory not found: {skills_dir}")

        self.skills_dir = skills_dir
        self.recursive = recursive

    def discover_all(self) -> dict[str, AgentSkill]:
        """
        发现所有技能

        扫描 skills 目录，加载所有技能的元数据（Discovery 阶段）。

        Returns:
            dict: {skill_name: AgentSkill} 字典

        示例：
        ```python
        loader = SkillLoader(Path("skills"))
        skills = loader.discover_all()
        for name, skill in skills.items():
            print(f"{skill.metadata.display_name}: {skill.metadata.description}")
        ```
        """
        skills = {}

        # 确定搜索模式
        if self.recursive:
            skill_dirs = [d for d in self.skills_dir.rglob("*") if d.is_dir() and (d / "SKILL.md").exists()]
        else:
            skill_dirs = [d for d in self.skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]

        # 加载每个技能
        for skill_dir in skill_dirs:
            try:
                skill = AgentSkill(skill_dir)
                if skill.metadata and skill.metadata.name:
                    skills[skill.metadata.name] = skill
            except Exception as e:
                print(f"⚠️  Failed to load skill from {skill_dir}: {e}")
                continue

        print(f"✓ Discovered {len(skills)} skills from {self.skills_dir}")
        return skills

    def discover_by_name(self, skill_name: str) -> Optional[AgentSkill]:
        """
        通过名称加载特定技能

        Args:
            skill_name: 技能名称（如 weather, calendar）

        Returns:
            AgentSkill: 技能实例，如果不存在返回 None

        示例：
        ```python
        loader = SkillLoader(Path("skills"))
        weather_skill = loader.discover_by_name("weather")
        if weather_skill:
            print(f"Found: {weather_skill.metadata.display_name}")
        ```
        """
        if self.recursive:
            # 递归搜索
            for skill_dir in self.skills_dir.rglob("*"):
                if skill_dir.is_dir() and skill_dir.name == skill_name:
                    skill_md = skill_dir / "SKILL.md"
                    if skill_md.exists():
                        try:
                            return AgentSkill(skill_dir)
                        except Exception:
                            return None
        else:
            # 直接搜索
            skill_dir = self.skills_dir / skill_name
            skill_md = skill_dir / "SKILL.md"

            if skill_md.exists():
                try:
                    return AgentSkill(skill_dir)
                except Exception:
                    return None

        return None

    def discover_by_category(self, category: str) -> dict[str, AgentSkill]:
        """
        按类别过滤技能

        Args:
            category: 类别名称（如 utility, productivity）

        Returns:
            dict: {skill_name: AgentSkill} 字典

        示例：
        ```python
        loader = SkillLoader(Path("skills"))
        utility_skills = loader.discover_by_category("utility")
        print(f"Found {len(utility_skills)} utility skills")
        ```
        """
        all_skills = self.discover_all()
        return {
            name: skill
            for name, skill in all_skills.items()
            if skill.metadata and skill.metadata.category == category
        }

    def discover_by_tag(self, tag: str) -> dict[str, AgentSkill]:
        """
        按标签过滤技能

        Args:
            tag: 标签名称（如 api, browser, mcp）

        Returns:
            dict: {skill_name: AgentSkill} 字典

        示例：
        ```python
        loader = SkillLoader(Path("skills"))
        api_skills = loader.discover_by_tag("api")
        print(f"Found {len(api_skills)} API skills")
        ```
        """
        all_skills = self.discover_all()
        return {
            name: skill
            for name, skill in all_skills.items()
            if skill.metadata and tag in skill.metadata.tags
        }

    def validate_skill_directory(self, skill_dir: Path) -> tuple[bool, list[str]]:
        """
        验证技能目录结构

        检查技能目录是否符合规范：
        - 必须包含 SKILL.md 文件
        - SKILL.md 必须有 YAML frontmatter
        - 必需的 metadata 字段

        Args:
            skill_dir: 技能目录路径

        Returns:
            tuple: (is_valid, error_messages)

        示例：
        ```python
        loader = SkillLoader(Path("skills"))
        is_valid, errors = loader.validate_skill_directory(Path("skills/weather"))
        if not is_valid:
            for error in errors:
                print(f"Error: {error}")
        ```
        """
        errors = []

        # 检查目录是否存在
        if not skill_dir.exists():
            errors.append(f"Directory not found: {skill_dir}")
            return False, errors

        # 检查 SKILL.md 是否存在
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            errors.append(f"SKILL.md not found in {skill_dir}")
            return False, errors

        # 尝试解析 SKILL.md
        try:
            import frontmatter

            with open(skill_md, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)

            metadata = post.metadata

            # 检查必需字段
            required_fields = ["name", "display_name", "description"]
            for field in required_fields:
                if field not in metadata:
                    errors.append(f"Missing required field: {field}")

        except Exception as e:
            errors.append(f"Failed to parse SKILL.md: {e}")

        return len(errors) == 0, errors

    def get_skill_info(self, skill_name: str) -> Optional[dict]:
        """
        获取技能信息（不加载完整技能）

        仅返回 metadata，不执行 Discovery 阶段。

        Args:
            skill_name: 技能名称

        Returns:
            dict: 技能信息，如果不存在返回 None

        示例：
        ```python
        loader = SkillLoader(Path("skills"))
        info = loader.get_skill_info("weather")
        if info:
            print(f"Name: {info['display_name']}")
            print(f"Description: {info['description']}")
        ```
        """
        skill = self.discover_by_name(skill_name)
        if skill and skill.metadata:
            return skill.metadata.to_dict()
        return None

    def list_categories(self) -> list[str]:
        """
        列出所有技能类别

        Returns:
            list[str]: 类别列表

        示例：
        ```python
        loader = SkillLoader(Path("skills"))
        categories = loader.list_categories()
        print(f"Available categories: {categories}")
        ```
        """
        all_skills = self.discover_all()
        categories = set()

        for skill in all_skills.values():
            if skill.metadata:
                categories.add(skill.metadata.category)

        return sorted(list(categories))

    def list_tags(self) -> list[str]:
        """
        列出所有标签

        Returns:
            list[str]: 标签列表

        示例：
        ```python
        loader = SkillLoader(Path("skills"))
        tags = loader.list_tags()
        print(f"Available tags: {tags}")
        ```
        """
        all_skills = self.discover_all()
        tags = set()

        for skill in all_skills.values():
            if skill.metadata:
                tags.update(skill.metadata.tags)

        return sorted(list(tags))
