"""
Agent Skills - 技能管理器（Claude Code 设计）

核心原理：
- 所有技能的 description 注入到 LLM system prompt
- LLM 自己判断使用哪个技能
- 统一的 skill_execute 函数

使用示例：
```python
from src.voice_assistant.skills.skill_manager import SkillManager

# 创建管理器
manager = SkillManager(Path("skills"))
await manager.initialize()

# 获取所有技能的 description（注入到 LLM）
skills_prompt = manager.get_skills_prompt()

# 执行技能
result = await manager.execute_skill("weather", user_input="查询北京天气")
```
"""
from pathlib import Path
from typing import Any, Optional

from pipecat.services.llm_service import FunctionCallParams

from .base_skill import (
    AgentSkill,
    SkillExecutionContext,
    SkillResult,
)
from .skill_loader import SkillLoader


class SkillManager:
    """
    技能管理器（Claude Code 设计）

    核心原理：
    1. 启动时加载所有技能的 metadata（仅 description）
    2. 把所有技能的 description 添加到 LLM system prompt
    3. 注册统一的 skill_execute 函数
    4. LLM 自己判断调用哪个技能

    Attributes:
        skills_dir: 技能根目录
        skills: 所有技能 {name: AgentSkill}
        llm_service: LLM 服务

    使用示例：
    ```python
    manager = SkillManager(Path("skills"))
    await manager.initialize()

    # 获取技能提示文本
    skills_prompt = manager.get_skills_prompt()

    # 设置 LLM 服务
    manager.set_llm_service(llm)
    ```
    """

    def __init__(self, skills_dir: Path):
        """
        初始化技能管理器

        Args:
            skills_dir: 技能根目录
        """
        self.skills_dir = skills_dir
        self.loader = SkillLoader(skills_dir)
        self.skills: dict[str, AgentSkill] = {}
        self.llm_service: Optional[Any] = None

    async def initialize(self) -> None:
        """
        初始化技能管理器

        加载所有技能的 metadata（仅 description，不加载完整指令）。

        示例：
        ```python
        manager = SkillManager(Path("skills"))
        await manager.initialize()
        print(f"Loaded {len(manager.skills)} skills")
        ```
        """
        print("🔍 Loading skills...")
        self.skills = self.loader.discover_all()
        print(f"✓ Loaded {len(self.skills)} skills")

        # 打印技能列表
        for skill_name, skill in self.skills.items():
            if skill.metadata:
                print(f"  - {skill_name}: {skill.metadata.display_name}")

    def get_skills_prompt(self) -> str:
        """
        获取所有技能的提示文本

        生成用于注入到 LLM system prompt 的文本。

        Returns:
            str: 技能提示文本

        示例：
        ```python
        skills_prompt = manager.get_skills_prompt()
        system_message = f\"\"\"
        你是一个智能助手。当用户需要时，你可以使用以下技能：

        {skills_prompt}

        当需要使用技能时，调用 skill_execute 函数，传入技能名称和用户输入。
        \"\"\"
        ```
        """
        if not self.skills:
            return ""

        lines = ["## 可用技能\n"]

        for skill_name, skill in self.skills.items():
            if skill.metadata:
                lines.append(f"### {skill.metadata.display_name} ({skill_name})")
                lines.append(f"{skill.metadata.description}\n")

        return "\n".join(lines)

    async def execute_skill(
        self,
        skill_name: str,
        user_input: str,
        **kwargs
    ) -> SkillResult:
        """
        执行技能

        Args:
            skill_name: 技能名称
            user_input: 用户输入
            **kwargs: 额外参数

        Returns:
            SkillResult: 执行结果

        示例：
        ```python
        result = await manager.execute_skill(
            "weather",
            user_input="查询北京天气"
        )
        if result.success:
            print(result.content)
        ```
        """
        if skill_name not in self.skills:
            return SkillResult(
                success=False,
                content="",
                error=f"Skill not found: {skill_name}"
            )

        skill = self.skills[skill_name]

        # 激活技能（加载完整指令）
        if skill.state.value != "activated":
            try:
                await skill.activate()
            except Exception as e:
                return SkillResult(
                    success=False,
                    content="",
                    error=f"Failed to activate skill: {e}"
                )

        # 构建执行上下文
        context = SkillExecutionContext(
            user_input=user_input,
            llm_service=self.llm_service,
            additional_context=kwargs
        )

        # 执行技能
        return await skill.execute(context)

    def set_llm_service(self, llm_service: Any) -> None:
        """
        设置 LLM 服务并注册 skill_execute 函数

        Args:
            llm_service: LLM 服务实例

        示例：
        ```python
        from pipecat.services.openai import OpenAILLMService

        llm = OpenAILLMService(api_key="...")
        manager.set_llm_service(llm)
        ```
        """
        self.llm_service = llm_service

        # 注册统一的 skill_execute 函数
        self._register_skill_execute_function()

        print(f"✓ LLM service set for SkillManager")
        print(f"✓ Registered skill_execute function")

    def _register_skill_execute_function(self) -> None:
        """
        注册统一的 skill_execute 函数

        LLM 可以调用这个函数来执行任何技能。
        """
        if not self.llm_service:
            return

        async def skill_execute_handler(params: FunctionCallParams):
            """统一的技能执行处理器"""
            try:
                # 检查是否是 skill_execute 函数调用
                if params.function_name != "skill_execute":
                    # 不是我们的函数，不处理
                    return

                skill_name = params.arguments.get("skill_name")
                user_input = params.arguments.get("user_input", "")

                if not skill_name:
                    await params.result_callback("Error: skill_name is required")
                    return

                # 执行技能
                result = await self.execute_skill(skill_name, user_input)

                if result.success:
                    await params.result_callback(result.content)
                else:
                    await params.result_callback(f"Error: {result.error}")

            except Exception as e:
                await params.result_callback(f"Exception: {e}")

        # 注册函数（只处理 skill_execute）
        self.llm_service.register_function(
            "skill_execute",
            skill_execute_handler
        )

        print(f"  ✓ Registered function: skill_execute(skill_name, user_input)")

    def get_skill_names(self) -> list[str]:
        """
        获取所有技能名称

        Returns:
            list[str]: 技能名称列表

        示例：
        ```python
        names = manager.get_skill_names()
        print(f"Available skills: {', '.join(names)}")
        ```
        """
        return list(self.skills.keys())

    def get_skill(self, skill_name: str) -> Optional[AgentSkill]:
        """
        获取技能实例

        Args:
            skill_name: 技能名称

        Returns:
            AgentSkill: 技能实例，如果不存在返回 None

        示例：
        ```python
        skill = manager.get_skill("weather")
        if skill and skill.metadata:
            print(f"Description: {skill.metadata.description}")
        ```
        """
        return self.skills.get(skill_name)

    async def get_status(self) -> dict[str, Any]:
        """
        获取管理器状态

        Returns:
            dict: 状态信息

        示例：
        ```python
        status = await manager.get_status()
        print(f"Total skills: {status['total_skills']}")
        ```
        """
        return {
            "total_skills": len(self.skills),
            "skill_names": self.get_skill_names(),
            "has_llm_service": self.llm_service is not None,
        }

    def __repr__(self) -> str:
        """字符串表示"""
        return f"SkillManager(skills={len(self.skills)})"
