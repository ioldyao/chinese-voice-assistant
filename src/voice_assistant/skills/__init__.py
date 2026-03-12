"""
Agent Skills - Claude Code 设计

基于 Claude Code 的技能系统设计。

核心原理：
- 所有技能的 description 注入到 LLM system prompt
- LLM 自己判断使用哪个技能
- 统一的 skill_execute 函数

核心组件：
- AgentSkill: 技能基类
- SkillLoader: 技能加载器
- SkillManager: 技能管理器（简化版）

使用示例：
```python
from src.voice_assistant.skills import SkillManager

# 创建管理器
manager = SkillManager(Path("skills"))
await manager.initialize()

# 获取技能提示文本（注入到 LLM）
skills_prompt = manager.get_skills_prompt()

# 设置 LLM 服务
manager.set_llm_service(llm)

# LLM 会自动调用 skill_execute 函数
```
"""
# ===== 基础组件 =====
from .base_skill import (
    AgentSkill,
    SkillExecutionContext,
    SkillResult,
    SkillState,
    SkillMetadata,
)
from .skill_loader import SkillLoader
from .skill_manager import SkillManager
from .skill_executor import SkillExecutor

__all__ = [
    # 基础组件
    "AgentSkill",
    "SkillExecutionContext",
    "SkillResult",
    "SkillState",
    "SkillMetadata",
    "SkillLoader",
    "SkillManager",
    "SkillExecutor",
]
