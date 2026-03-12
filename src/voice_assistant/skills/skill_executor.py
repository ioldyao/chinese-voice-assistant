"""
Agent Skills - 技能执行器

处理技能的执行逻辑，包括：
- 直接执行（返回指令给 LLM）
- 工具调用（调用 MCP 工具）
- API 调用（调用外部 API）
- 多步骤执行（复杂工作流）

使用示例：
```python
from src.voice_assistant.skills.skill_executor import SkillExecutor

executor = SkillExecutor()

# 执行技能（会自动选择执行方式）
result = await executor.execute(skill, context)
```
"""
from typing import Any

from .base_skill import (
    AgentSkill,
    SkillExecutionContext,
    SkillResult,
)


class SkillExecutor:
    """
    技能执行器

    负责执行技能逻辑，支持多种执行模式。

    Attributes:
        mcp_manager: MCP 工具管理器
        llm_service: LLM 服务

    使用示例：
    ```python
    executor = SkillExecutor(mcp_manager=mcp, llm_service=llm)
    result = await executor.execute(skill, context)
    ```
    """

    def __init__(self, mcp_manager: Any = None, llm_service: Any = None):
        """
        初始化技能执行器

        Args:
            mcp_manager: MCP 工具管理器（用于工具调用）
            llm_service: LLM 服务（用于生成式执行）
        """
        self.mcp_manager = mcp_manager
        self.llm_service = llm_service

    async def execute(
        self,
        skill: AgentSkill,
        context: SkillExecutionContext
    ) -> SkillResult:
        """
        执行技能

        根据技能配置选择执行方式：
        1. 如果 skill 定义了 execute 方法，直接调用
        2. 如果需要 MCP 工具，执行工具调用
        3. 否则返回指令给 LLM（默认）

        Args:
            skill: 技能实例
            context: 执行上下文

        Returns:
            SkillResult: 执行结果

        示例：
        ```python
        executor = SkillExecutor()
        result = await executor.execute(skill, context)
        if result.success:
            print(result.content)
        ```
        """
        try:
            # 检查技能是否有自定义 execute 方法
            # （子类可以重写 execute 方法）
            if hasattr(skill, 'execute') and skill.__class__.execute != AgentSkill.execute:
                # 调用技能的自定义 execute 方法
                return await skill.execute(context)

            # 检查是否需要 MCP 工具
            if skill.metadata and skill.metadata.requires_tools:
                return await self._execute_with_tools(skill, context)

            # 默认：返回指令给 LLM
            return await self._execute_default(skill, context)

        except Exception as e:
            return SkillResult(
                success=False,
                content="",
                error=str(e),
                metadata={"skill_name": skill.metadata.name if skill.metadata else "unknown"}
            )

    async def _execute_default(
        self,
        skill: AgentSkill,
        context: SkillExecutionContext
    ) -> SkillResult:
        """
        默认执行方式

        将技能指令返回给 LLM 处理。

        Args:
            skill: 技能实例
            context: 执行上下文

        Returns:
            SkillResult: 执行结果
        """
        if not skill.instructions:
            return SkillResult(
                success=False,
                content="",
                error="Skill instructions not loaded",
                metadata={"skill_name": skill.metadata.name if skill.metadata else "unknown"}
            )

        content = f"""# {skill.metadata.display_name if skill.metadata else 'Skill'}

{skill.instructions}

用户输入：{context.user_input}
"""

        return SkillResult(
            success=True,
            content=content,
            metadata={
                "skill_name": skill.metadata.name if skill.metadata else "unknown",
                "execution_mode": "default",
            }
        )

    async def _execute_with_tools(
        self,
        skill: AgentSkill,
        context: SkillExecutionContext
    ) -> SkillResult:
        """
        使用工具执行

        调用 MCP 工具执行技能逻辑。

        Args:
            skill: 技能实例
            context: 执行上下文

        Returns:
            SkillResult: 执行结果
        """
        if not self.mcp_manager:
            return SkillResult(
                success=False,
                content="",
                error="MCP manager not configured",
                metadata={"skill_name": skill.metadata.name if skill.metadata else "unknown"}
            )

        if not skill.metadata or not skill.metadata.requires_tools:
            return SkillResult(
                success=False,
                content="",
                error="No tools required",
                metadata={"skill_name": skill.metadata.name if skill.metadata else "unknown"}
            )

        # 调用 MCP 工具
        tool_name = skill.metadata.requires_tools[0]  # 使用第一个工具

        try:
            result = await self.mcp_manager.call_tool_async(
                tool_name,
                {"user_input": context.user_input}
            )

            if result.success:
                return SkillResult(
                    success=True,
                    content=str(result.content) if result.content else "操作成功",
                    metadata={
                        "skill_name": skill.metadata.name,
                        "execution_mode": "tool",
                        "tool_name": tool_name,
                    }
                )
            else:
                return SkillResult(
                    success=False,
                    content="",
                    error=result.error,
                    metadata={
                        "skill_name": skill.metadata.name,
                        "execution_mode": "tool",
                        "tool_name": tool_name,
                    }
                )

        except Exception as e:
            return SkillResult(
                success=False,
                content="",
                error=str(e),
                metadata={
                    "skill_name": skill.metadata.name if skill.metadata else "unknown",
                    "execution_mode": "tool",
                    "tool_name": tool_name,
                }
            )
