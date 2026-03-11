"""
Agent Skills - 基础类和数据模型

基于 Anthropic Agent Skills 开放标准：
https://agentskills.io/what-are-skills

核心概念：
- Progressive Disclosure: 渐进式加载（发现 → 激活 → 执行）
- SKILL.md: 技能定义文件（YAML frontmatter + Markdown 内容）
- Modular: 模块化设计，易于扩展

使用示例：
```python
from src.voice_assistant.skills.base_skill import AgentSkill

skill = AgentSkill(Path("skills/weather"))
print(f"Skill: {skill.name}")
print(f"Description: {skill.description}")

# 激活技能（加载完整指令）
await skill.activate()
print(f"Instructions: {skill.instructions}")

# 执行技能
result = await skill.execute(user_input="查询北京天气")
print(result)
```
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class SkillState(Enum):
    """技能状态枚举

    Progressive Disclosure 状态机：
    DISCOVERED → ACTIVATED → EXECUTING
    """
    DISCOVERED = "discovered"     # 已发现（仅加载 metadata）
    ACTIVATED = "activated"       # 已激活（加载完整 instructions）
    EXECUTING = "executing"       # 执行中


@dataclass
class SkillMetadata:
    """技能元数据

    从 SKILL.md 的 YAML frontmatter 解析得到。

    YAML 格式（符合 Claude Code 标准）：
    ```yaml
    ---
    name: weather
    display_name: 天气查询
    description: This skill should be used when the user asks to "查询天气", "今天天气怎么样", "天气如何"
    version: 1.0.0
    author: AI Assistant
    tags: [weather, api, utility]
    category: utility
    requires_tools: []
    ---
    ```

    Attributes:
        name: 技能唯一标识符（如 weather, calendar）
        display_name: 技能显示名称（中文）
        description: 技能描述（必须使用第三人称："This skill should be used when..."）
        version: 技能版本
        author: 作者
        tags: 标签列表（用于分类和搜索）
        category: 技能类别（utility, productivity, entertainment 等）
        requires_tools: 需要的工具列表（用于 MCP 集成）
        references_files: references/ 目录中的文件列表（详细文档）
        examples_files: examples/ 目录中的文件列表（可运行示例）
        scripts_files: scripts/ 目录中的文件列表（可执行工具）
    """
    name: str
    display_name: str
    description: str
    version: str = "1.0.0"
    author: str = "AI Assistant"
    tags: list[str] = field(default_factory=list)
    category: str = "utility"
    requires_tools: list[str] = field(default_factory=list)
    # 渐进式披露资源（Claude Code 标准）
    references_files: list[str] = field(default_factory=list)
    examples_files: list[str] = field(default_factory=list)
    scripts_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "category": self.category,
            "requires_tools": self.requires_tools,
            "references_files": self.references_files,
            "examples_files": self.examples_files,
            "scripts_files": self.scripts_files,
        }

    @property
    def has_progressive_resources(self) -> bool:
        """检查是否有渐进式披露资源"""
        return bool(self.references_files or self.examples_files or self.scripts_files)


@dataclass
class SkillExecutionContext:
    """技能执行上下文

    提供技能执行时需要的上下文信息。

    Attributes:
        user_input: 用户输入
        conversation_history: 对话历史
        tools_manager: 工具管理器（MCP）
        llm_service: LLM 服务
        additional_context: 额外上下文信息
    """
    user_input: str
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    tools_manager: Optional[Any] = None
    llm_service: Optional[Any] = None
    additional_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """技能执行结果

    Attributes:
        success: 是否成功
        content: 结果内容
        error: 错误信息（如果失败）
        metadata: 额外元数据（如使用的工具、执行时间等）
    """
    success: bool
    content: str
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "metadata": self.metadata,
        }


class AgentSkill:
    """
    Agent Skill 基类

    实现 Progressive Disclosure 机制：
    1. Discovery: 初始化时仅加载 metadata
    2. Activation: 调用 activate() 加载完整 instructions
    3. Execution: 调用 execute() 执行技能逻辑

    属性:
        skill_path: 技能目录路径
        metadata: 技能元数据（初始化时加载）
        state: 当前状态
        instructions: 技能指令（activate 时加载）
        skill_md_path: SKILL.md 文件路径

    使用示例：
    ```python
    # 1. 发现技能（仅加载 metadata）
    skill = AgentSkill(Path("skills/weather"))
    print(f"发现技能: {skill.metadata.display_name}")

    # 2. 激活技能（加载完整指令）
    await skill.activate()
    print(f"指令长度: {len(skill.instructions)}")

    # 3. 执行技能
    context = SkillExecutionContext(user_input="查询北京天气")
    result = await skill.execute(context)
    print(result.content)
    ```
    """

    def __init__(self, skill_path: Path):
        """
        初始化技能（Discovery 阶段）

        Args:
            skill_path: 技能目录路径（必须包含 SKILL.md 文件）

        Raises:
            FileNotFoundError: SKILL.md 文件不存在
            ValueError: SKILL.md 格式错误
        """
        self.skill_path = skill_path
        self.skill_md_path = skill_path / "SKILL.md"

        if not self.skill_md_path.exists():
            raise FileNotFoundError(f"SKILL.md not found in {skill_path}")

        # 初始状态：已发现（仅加载 metadata）
        self.state = SkillState.DISCOVERED
        self.metadata: Optional[SkillMetadata] = None
        self.instructions: Optional[str] = None

        # 加载 metadata（不加载完整 instructions）
        self._load_metadata()

    def _load_metadata(self) -> None:
        """
        加载技能元数据（Discovery 阶段）

        使用 python-frontmatter 解析 SKILL.md 文件，
        仅读取 YAML frontmatter，不加载完整内容。

        同时扫描渐进式披露资源目录：
        - references/ - 详细文档
        - examples/ - 可运行示例
        - scripts/ - 可执行工具
        """
        try:
            import frontmatter

            with open(self.skill_md_path, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)

            # 解析 YAML frontmatter
            metadata_dict = post.metadata

            # 扫描渐进式披露资源目录
            references_files = self._scan_resource_dir("references")
            examples_files = self._scan_resource_dir("examples")
            scripts_files = self._scan_resource_dir("scripts")

            self.metadata = SkillMetadata(
                name=metadata_dict.get("name", self.skill_path.name),
                display_name=metadata_dict.get("display_name", "未命名技能"),
                description=metadata_dict.get("description", ""),
                version=metadata_dict.get("version", "1.0.0"),
                author=metadata_dict.get("author", "AI Assistant"),
                tags=metadata_dict.get("tags", []),
                category=metadata_dict.get("category", "utility"),
                requires_tools=metadata_dict.get("requires_tools", []),
                references_files=references_files,
                examples_files=examples_files,
                scripts_files=scripts_files,
            )

        except Exception as e:
            raise ValueError(f"Failed to parse SKILL.md: {e}")

    def _scan_resource_dir(self, dir_name: str) -> list[str]:
        """
        扫描资源目录，返回文件列表

        Args:
            dir_name: 目录名称（references/examples/scripts）

        Returns:
            list[str]: 相对路径列表（如 ["references/patterns.md"]）
        """
        resource_dir = self.skill_path / dir_name
        if not resource_dir.exists() or not resource_dir.is_dir():
            return []

        files = []
        for file_path in resource_dir.iterdir():
            if file_path.is_file():
                # 返回相对路径：skill_name/references/file.md
                relative_path = f"{self.metadata.name if self.metadata else self.skill_path.name}/{dir_name}/{file_path.name}"
                files.append(relative_path)

        return sorted(files)

    def get_resource_path(self, resource_type: str, filename: str) -> Path:
        """
        获取资源文件的完整路径

        Args:
            resource_type: 资源类型（references/examples/scripts）
            filename: 文件名

        Returns:
            Path: 文件完整路径

        Raises:
            FileNotFoundError: 文件不存在
        """
        resource_dir = self.skill_path / resource_type
        file_path = resource_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Resource not found: {resource_type}/{filename}")

        return file_path

    async def load_reference(self, filename: str) -> str:
        """
        加载 references/ 目录中的文档

        Args:
            filename: 文件名（如 "patterns.md"）

        Returns:
            str: 文件内容

        Raises:
            FileNotFoundError: 文件不存在
        """
        file_path = self.get_resource_path("references", filename)

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    async def load_example(self, filename: str) -> str:
        """
        加载 examples/ 目录中的示例代码

        Args:
            filename: 文件名（如 "example.sh"）

        Returns:
            str: 文件内容
        """
        file_path = self.get_resource_path("examples", filename)

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_script_path(self, filename: str) -> Path:
        """
        获取 scripts/ 目录中脚本的路径

        Args:
            filename: 脚本文件名

        Returns:
            Path: 脚本完整路径
        """
        return self.get_resource_path("scripts", filename)

    async def activate(self) -> None:
        """
        激活技能（Activation 阶段）

        加载完整的技能指令内容。

        Raises:
            RuntimeError: 技能未处于 DISCOVERED 状态
        """
        if self.state != SkillState.DISCOVERED:
            raise RuntimeError(f"Cannot activate skill in state: {self.state}")

        try:
            import frontmatter

            with open(self.skill_md_path, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)

            # 加载完整内容（不包含 frontmatter）
            self.instructions = post.content
            self.state = SkillState.ACTIVATED

        except Exception as e:
            raise RuntimeError(f"Failed to activate skill: {e}")

    async def execute(self, context: SkillExecutionContext) -> SkillResult:
        """
        执行技能（Execution 阶段）

        默认实现：返回指令内容供 LLM 使用。

        子类可以重写此方法实现自定义逻辑：
        - 调用 MCP 工具
        - 调用外部 API
        - 执行复杂的多步骤操作

        Args:
            context: 执行上下文

        Returns:
            SkillResult: 执行结果

        Raises:
            RuntimeError: 技能未激活
        """
        if self.state != SkillState.ACTIVATED:
            raise RuntimeError(f"Cannot execute skill in state: {self.state}")

        self.state = SkillState.EXECUTING

        try:
            # 默认实现：返回指令供 LLM 使用
            # 子类可以重写此方法实现具体逻辑
            if not self.instructions:
                raise RuntimeError("Skill instructions not loaded")

            content = f"技能指令：\n{self.instructions}\n\n用户输入：\n{context.user_input}"

            return SkillResult(
                success=True,
                content=content,
                metadata={"skill_name": self.metadata.name if self.metadata else "unknown"}
            )

        except Exception as e:
            return SkillResult(
                success=False,
                content="",
                error=str(e),
                metadata={"skill_name": self.metadata.name if self.metadata else "unknown"}
            )

        finally:
            # 执行完成后恢复为 ACTIVATED 状态
            if self.state == SkillState.EXECUTING:
                self.state = SkillState.ACTIVATED

    def __repr__(self) -> str:
        """字符串表示"""
        if self.metadata:
            return f"AgentSkill(name={self.metadata.name}, state={self.state.value})"
        return f"AgentSkill(state={self.state.value})"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式（用于序列化）"""
        return {
            "state": self.state.value,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "has_instructions": self.instructions is not None,
        }
