# Agent Skills v3.0 集成指南

## 概述

本指南详细说明如何将 Agent Skills v3.0 集成到你的 Pipecat 应用中。

## 前置要求

- Python 3.10+
- Pipecat 最新版本
- 已配置 LLM 服务（Qwen/DeepSeek/OpenAI）
- 已配置 MCP 服务（可选）

## 快速开始

### 1. 创建技能目录结构

```
project/
├── skills/
│   ├── weather/
│   │   └── SKILL.md
│   ├── calculator/
│   │   └── SKILL.md
│   └── calendar/
│       └── SKILL.md
├── src/
│   └── voice_assistant/
│       └── main.py
└── pyproject.toml
```

### 2. 定义技能（SKILL.md）

**skills/weather/SKILL.md**:
```markdown
---
name: weather
display_name: 天气查询
description: 查询全球各地的天气信息
version: 1.0.0
author: AI Assistant
tags: [weather, api, utility]
category: utility
requires_tools: []
---

# 天气查询技能

## 功能

- 查询实时天气
- 查询天气预报
- 查询天气预警

## 使用方法

当用户询问天气相关信息时，使用此技能。

## 示例

- "今天北京天气怎么样"
- "明天会下雨吗"
- "查询上海天气"
```

### 3. 集成到主程序

```python
import asyncio
from pathlib import Path
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams

# ✅ 导入 v3.0 组件
from src.voice_assistant.skills import (
    SkillRegistry,
    SkillMatcher,
    SkillManager,
    SkillIntegrationLayer,
)

async def main():
    # 1. 创建技能系统组件
    project_root = Path(__file__).parent.parent.parent
    skills_dir = project_root / "skills"

    registry = SkillRegistry(skills_dir)
    matcher = SkillMatcher(registry)
    skill_manager = SkillManager(skills_dir)

    # 2. 创建集成层
    skill_integration_layer = SkillIntegrationLayer(
        registry=registry,
        matcher=matcher,
        skill_manager=skill_manager,
        llm_service=llm,  # 你的 LLM 服务
        mcp_manager=mcp,  # 你的 MCP 管理（可选）
        match_threshold=0.5,
        max_active_skills=3
    )

    # 3. 初始化
    await skill_integration_layer.initialize()

    # 4. 创建 Pipeline（不含 SkillProcessor）
    pipeline = Pipeline([
        transport.input(),
        kws_proc,
        asr_proc,
        user_aggregator,
        vision_proc,
        llm,
        tts_proc,
        assistant_aggregator,
        transport.output(),
    ])

    # 5. 添加 TranscriptionListener
    from pipecat.frames.frames import TranscriptionFrame, Frame
    from pipecat.processors.frame_processor import FrameProcessor

    class TranscriptionListener(FrameProcessor):
        def __init__(self, skill_layer: SkillIntegrationLayer):
            super().__init__()
            self.skill_layer = skill_layer

        async def process_frame(self, frame: Frame, direction: int):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

            if isinstance(frame, TranscriptionFrame):
                await self.skill_layer.process_user_input(frame.text)

    transcription_listener = TranscriptionListener(skill_integration_layer)
    pipeline._processors.insert(
        pipeline._processors.index(vision_proc),
        transcription_listener
    )

    # 6. 创建并运行任务
    task = PipelineTask(pipeline, params=PipelineParams(...))
    skill_integration_layer.set_task(task)

    await task.queue_frames([StartFrame()])

    runner = PipelineRunner()
    await runner.run(task)
```

## 高级配置

### 自定义匹配权重

```python
# 调整匹配策略权重
skill_integration_layer.matcher.set_weights({
    "exact_name_match": 1.0,      # 精确名称匹配
    "exact_display_match": 0.9,   # 精确显示名称匹配
    "keyword_in_name": 0.8,       # 关键词在名称中
    "keyword_in_display": 0.7,    # 关键词在显示名称中
    "keyword_in_desc": 0.5,       # 关键词在描述中
    "tag_match": 0.6,             # 标签匹配
    "category_match": 0.4,        # 类别匹配
})
```

### 禁用自动激活

```python
# 禁用自动激活，手动控制
skill_integration_layer.set_auto_activate(False)

# 手动激活技能
await skill_integration_layer.manually_activate_skill("weather")
```

### 性能优化

```python
# 启用缓存（默认开启）
matcher = SkillMatcher(registry, enable_cache=True)

# 清空缓存
matcher.clear_cache()

# 查看缓存大小
cache_size = matcher.get_cache_size()
```

## 监控和调试

### 查看系统状态

```python
# 获取完整状态
status = await skill_integration_layer.get_status()

print(f"Total skills: {status['registry']['total_skills']}")
print(f"Active skills: {status['registry']['active_skills']}")
print(f"Cache size: {status['matcher']['cache_size']}")
```

### 查看匹配结果

```python
# 查看匹配详情
results = await skill_integration_layer.matcher.match_skills("今天天气怎么样")
for result in results:
    print(f"Skill: {result.skill_name}")
    print(f"Score: {result.score:.2f}")
    print(f"Reason: {result.match_reason}")
```

### 查看激活的技能

```python
# 列出所有激活的技能
active_skills = skill_integration_layer.registry.get_active_skills()
for skill in active_skills:
    print(f"- {skill.metadata.display_name}")
```

## 技能定义最佳实践

### 1. 元数据

```yaml
---
name: weather              # 简短、唯一的英文名称
display_name: 天气查询     # 清晰的中文显示名称
description: 查询全球各地的天气信息，包括温度、湿度、风力等  # 详细的描述
version: 1.0.0
author: AI Assistant
tags: [weather, api, utility]  # 相关标签，提高匹配准确度
category: utility           # 合理的类别
requires_tools: []         # 需要的工具（如需要 MCP）
---
```

### 2. 关键词优化

```markdown
# 天气查询技能

## 功能

- **实时天气**：查询当前天气状况
- **天气预报**：查询未来几天天气
- **天气预警**：查询天气预警信息

## 关键词

天气、气温、温度、下雨、晴天、阴天、下雪、风力、湿度、预报

## 使用场景

当用户询问：
- "今天天气怎么样"
- "明天会下雨吗"
- "查询北京天气"
```

### 3. 技能执行

如果需要自定义执行逻辑：

```python
from src.voice_assistant.skills.base_skill import AgentSkill, SkillExecutionContext, SkillResult

class WeatherSkill(AgentSkill):
    async def execute(self, context: SkillExecutionContext) -> SkillResult:
        # 调用 API
        weather_data = await self._call_weather_api(context.user_input)

        # 返回结果
        return SkillResult(
            success=True,
            content=f"今天{weather_data['city']}的天气是{weather_data['condition']}",
            metadata={"api_call": "weather_api"}
        )
```

## 测试

### 单元测试

```python
import pytest
from src.voice_assistant.skills import SkillRegistry, SkillMatcher

@pytest.mark.asyncio
async def test_skill_matching():
    registry = SkillRegistry(Path("skills"))
    await registry.discover_all()

    matcher = SkillMatcher(registry)
    results = await matcher.match_skills("今天天气怎么样")

    assert len(results) > 0
    assert results[0].skill_name == "weather"
```

### 集成测试

```python
@pytest.mark.asyncio
async def test_skill_activation():
    # ... 初始化代码 ...

    # 模拟用户输入
    await skill_integration_layer.process_user_input("今天天气怎么样")

    # 验证技能已激活
    assert skill_integration_layer.registry.is_active("weather")
```

### 手动测试

```python
# 测试脚本
async def test_skills():
    # 初始化
    await skill_integration_layer.initialize()

    # 测试用例
    test_cases = [
        ("今天天气怎么样", "weather"),
        ("计算 1+1", "calculator"),
        ("添加日程", "calendar"),
    ]

    for user_input, expected_skill in test_cases:
        print(f"\n测试: {user_input}")

        # 匹配
        results = await skill_integration_layer.matcher.match_skills(user_input)
        for result in results:
            print(f"  - {result.skill_name}: {result.score:.2f}")

        # 激活
        await skill_integration_layer.process_user_input(user_input)

        # 验证
        if skill_integration_layer.registry.is_active(expected_skill):
            print(f"  ✓ {expected_skill} 已激活")
        else:
            print(f"  ✗ {expected_skill} 未激活")
```

## 故障排除

### 问题 1：技能不激活

**症状**：用户输入相关内容，但技能未激活

**排查步骤**：
1. 检查匹配阈值是否过高
2. 检查技能元数据是否包含相关关键词
3. 查看匹配结果评分

```python
# 调试匹配
results = await skill_integration_layer.matcher.match_skills("测试")
for result in results:
    print(f"{result.skill_name}: {result.score:.2f} - {result.match_reason}")
```

### 问题 2：误激活

**症状**：不相关的技能被激活

**解决方法**：
1. 提高匹配阈值
2. 优化技能元数据，减少通用关键词
3. 降低相关权重

```python
# 提高阈值
skill_integration_layer.set_match_threshold(0.7)

# 调整权重
skill_integration_layer.matcher.set_weights({
    "keyword_in_desc": 0.3,  # 降低描述匹配权重
})
```

### 问题 3：性能问题

**症状**：响应变慢，内存占用增加

**优化方法**：
1. 启用匹配缓存
2. 限制最大激活技能数
3. 提高匹配阈值，减少激活次数

```python
# 启用缓存
matcher = SkillMatcher(registry, enable_cache=True)

# 限制激活数
skill_integration_layer.set_max_active_skills(2)

# 提高阈值
skill_integration_layer.set_match_threshold(0.7)
```

## 迁移检查清单

从 v2.7 迁移到 v3.0 的检查清单：

- [ ] 更新导入语句
- [ ] 移除 SkillProcessor 从 Pipeline
- [ ] 创建 SkillIntegrationLayer
- [ ] 添加 TranscriptionListener
- [ ] 更新 main 函数
- [ ] 测试技能发现
- [ ] 测试意图匹配
- [ ] 测试技能激活
- [ ] 测试完整流程
- [ ] 性能测试
- [ ] 更新文档

## 参考资料

- [Agent Skills v3.0 架构文档](./agent-skills-v3-architecture.md)
- [Agent Skills v3.0 迁移指南](./agent-skills-migration-v3.md)
- [Pipecat 官方文档](https://github.com/pipecat-ai/pipecat)
- [Agent Skills 开放标准](https://agentskills.io/)

## 支持

如有问题或建议，请：

1. 查看架构文档和迁移指南
2. 检查测试用例
3. 查看 Pipecat 官方文档
4. 提交 Issue 或 Pull Request
