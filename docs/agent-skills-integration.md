# Agent Skills 集成指南

本文档说明如何将 Agent Skills 集成到你的语音助手项目中。

## 目录

- [概述](#概述)
- [架构说明](#架构说明)
- [快速开始](#快速开始)
- [集成到主程序](#集成到主程序)
- [配置选项](#配置选项)
- [故障排除](#故障排除)

## 概述

Agent Skills 是基于 Anthropic 开放标准的技能系统，提供：

- ✅ **Progressive Disclosure**: 渐进式加载（发现 → 激活 → 执行）
- ✅ **模块化设计**: 每个技能是独立的 SKILL.md 文件
- ✅ **Pipecat 集成**: 无缝集成到 Pipecat Pipeline
- ✅ **MCP 支持**: 与 MCP 工具集成
- ✅ **动态注册**: 运行时动态注册技能函数

## 架构说明

### 组件关系

```
skills/                    # 技能目录
├── weather/SKILL.md      # 天气查询技能
├── calendar/SKILL.md     # 日历管理技能
└── browser/SKILL.md      # 浏览器控制技能

src/voice_assistant/skills/
├── base_skill.py         # 技能基类
├── skill_loader.py       # 技能加载器
├── skill_manager.py      # 技能管理器
├── skill_executor.py     # 技能执行器
└── skill_processor.py    # Pipecat 集成
```

### Progressive Disclosure 机制

```
Discovery (发现)
  ↓ 仅加载 metadata
Activation (激活)
  ↓ 加载完整 instructions
Execution (执行)
  ↓ 执行技能逻辑
Result (返回)
```

## 快速开始

### 1. 创建技能

在 `skills/` 目录下创建新技能：

```bash
mkdir skills/my_skill
touch skills/my_skill/SKILL.md
```

编辑 `SKILL.md`：

```markdown
---
name: my_skill
display_name: 我的技能
description: 技能描述
version: 1.0.0
author: AI Assistant
tags: [demo, utility]
category: utility
requires_tools: []
---

# 我的技能

你是一个专业的助手，可以...

## 使用方法

当用户...时...
```

### 2. 使用 SkillManager

```python
from pathlib import Path
from src.voice_assistant.skills import SkillManager

# 创建管理器
manager = SkillManager(Path("skills"))

# 发现所有技能
await manager.discover_all()

# 激活技能
await manager.activate_skill("my_skill")

# 执行技能
result = await manager.execute_skill(
    "my_skill",
    user_input="用户输入"
)

if result.success:
    print(result.content)
else:
    print(f"Error: {result.error}")
```

## 集成到主程序

### 方式 1: 使用 SkillProcessor（推荐）

```python
from src.voice_assistant.skills import SkillManager, SkillProcessor
from pipecat.pipeline.pipeline import Pipeline

# 1. 创建 SkillManager
manager = SkillManager(Path("skills"))
await manager.discover_all()

# 2. 创建 SkillProcessor
processor = SkillProcessor(
    manager=manager,
    llm_service=llm,
    mcp_manager=mcp,
    auto_activate=True  # 自动激活所有技能
)

# 3. 添加到 Pipeline
pipeline = Pipeline([
    transport.input(),
    kws_proc,
    asr_proc,
    user_aggregator,
    processor,  # ← SkillProcessor
    llm,
    tts_proc,
    assistant_aggregator,
    transport.output(),
])

# 4. 创建 PipelineTask
task = PipelineTask(pipeline, params=PipelineParams(...))

# 5. 设置 task（用于注入系统指令）
await processor.set_task(task)
```

### 方式 2: 手动集成

```python
# 在 create_pipecat_pipeline() 中

# 1. 创建并配置 SkillManager
manager = SkillManager(Path("skills"))
await manager.discover_all()

# 2. 设置 LLM 服务
manager.set_llm_service(llm)

# 3. 激活需要的技能
await manager.activate_skill("weather")
await manager.activate_skill("calendar")

# 4. 创建 PipelineTask 后
manager.set_task(task)

# 5. 注册技能函数处理器
async def skill_function_handler(params: FunctionCallParams):
    function_name = params.function_name
    user_input = params.arguments.get("user_input", "")

    # 提取技能名称（去掉 "skill_" 前缀）
    skill_name = function_name.replace("skill_", "")

    # 执行技能
    result = await manager.execute_skill(skill_name, user_input)

    # 返回结果
    await params.result_callback(result.content)

# 注册为 catch-all 处理器
llm.register_function(None, skill_function_handler)
```

## 配置选项

### SkillProcessor 参数

```python
processor = SkillProcessor(
    manager=skill_manager,      # 必需：技能管理器
    llm_service=llm,            # 可选：LLM 服务
    mcp_manager=mcp,            # 可选：MCP 工具管理器
    auto_activate=True          # 可选：自动激活所有技能（默认 True）
)
```

### 环境变量

```bash
# .env
SKILLS_DIR=skills              # 技能目录
SKILLS_AUTO_ACTIVATE=true      # 是否自动激活
```

## 故障排除

### 问题 1: 技能未发现

**症状**: `discover_all()` 返回空字典

**解决方案**:
1. 检查 `skills/` 目录是否存在
2. 确保每个技能子目录包含 `SKILL.md` 文件
3. 验证 `SKILL.md` 格式正确（YAML frontmatter + content）

```bash
# 检查目录结构
ls -la skills/weather/SKILL.md
```

### 问题 2: 技能激活失败

**症状**: `activate_skill()` 返回 False

**解决方案**:
1. 检查 `SKILL.md` 的 YAML frontmatter 格式
2. 确保所有必需字段存在（name, display_name, description）
3. 查看错误日志获取详细错误信息

```python
# 验证 SKILL.md
from src.voice_assistant.skills import SkillLoader

loader = SkillLoader(Path("skills"))
is_valid, errors = loader.validate_skill_directory(Path("skills/weather"))
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

### 问题 3: 技能函数未注册

**症状**: LLM 无法调用技能函数

**解决方案**:
1. 确保设置了 `llm_service`
2. 检查 `llm.register_function()` 是否被调用
3. 验证函数名称格式：`skill_{skill_name}`

```python
# 检查已注册的函数
print(f"Active skills: {manager.list_active_skills()}")
```

### 问题 4: 系统指令未注入

**症状**: LLM 不知道如何使用技能

**解决方案**:
1. 确保设置了 `task`
2. 调用 `activate_skill()` 时 `task` 已设置
3. 检查系统消息是否正确添加到 context

```python
# 手动注入指令
await processor.inject_system_instructions(
    "你可以使用天气查询技能..."
)
```

## 进阶使用

### 自定义技能执行逻辑

继承 `AgentSkill` 并重写 `execute()` 方法：

```python
from src.voice_assistant.skills import AgentSkill, SkillResult

class WeatherSkill(AgentSkill):
    async def execute(self, context):
        # 调用天气 API
        api_result = await call_weather_api(context.user_input)

        return SkillResult(
            success=True,
            content=api_result,
            metadata={"source": "weather_api"}
        )
```

### 与 MCP 工具集成

在 `SKILL.md` 中声明需要的工具：

```yaml
---
requires_tools: ["browser_navigate", "browser_click"]
---
```

技能执行时会自动调用这些工具。

### 动态重载技能

开发时可以重载技能而无需重启：

```python
await manager.reload_skill("weather")
```

## 最佳实践

1. **技能命名**: 使用简洁的英文名称（如 weather, calendar）
2. **分类**: 使用合适的 category（utility, productivity, entertainment）
3. **标签**: 添加相关标签便于搜索和过滤
4. **文档**: 在 SKILL.md 中提供详细的使用说明
5. **错误处理**: 在自定义 execute 方法中处理异常

## 相关文档

- [技能编写指南](./agent-skills-authoring.md)
- [AgentSkills.io 官方文档](https://agentskills.io/)
- [Pipecat 官方文档](https://docs.pipecat.ai/)
