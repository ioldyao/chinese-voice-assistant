# Agent Skills v3.0 迁移指南

## 概述

本指南帮助你从 Agent Skills v2.7 迁移到 v3.0。

## 版本对比

| 特性 | v2.7 (旧) | v3.0 (新) |
|------|-----------|-----------|
| Pipeline 集成 | FrameProcessor | 独立运行（TranscriptionListener） |
| 技能激活 | 批量激活（auto_activate=True） | 智能意图匹配 |
| 帧流影响 | 可能干扰音频流 | 零干扰 |
| 激活时机 | 启动时激活所有 | 用户输入时动态激活 |
| 组件数量 | 2 个（SkillManager, SkillProcessor） | 4 个（+Registry, +Matcher, +IntegrationLayer） |

## 迁移步骤

### 步骤 1：更新导入

**v2.7（旧）**：
```python
from src.voice_assistant.skills import SkillManager, SkillProcessor
```

**v3.0（新）**：
```python
from src.voice_assistant.skills import (
    SkillManager,
    SkillIntegrationLayer,
    SkillRegistry,
    SkillMatcher,
)
```

### 步骤 2：移除 SkillProcessor 从 Pipeline

**v2.7（旧）**：
```python
# 创建 SkillProcessor
skill_proc = SkillProcessor(
    skill_manager=skill_manager,
    llm_service=llm,
    mcp_manager=mcp,
    auto_activate=True  # 自动激活所有技能
)

# 添加到 Pipeline
pipeline = Pipeline([
    transport.input(),
    kws_proc,
    asr_proc,
    user_aggregator,
    vision_proc,
    skill_proc,  # ← 移除这行
    llm,
    tts_proc,
    assistant_aggregator,
    transport.output(),
])
```

**v3.0（新）**：
```python
# ✅ 移除 skill_proc
pipeline = Pipeline([
    transport.input(),
    kws_proc,
    asr_proc,
    user_aggregator,
    vision_proc,
    # ❌ skill_proc 已移除
    llm,
    tts_proc,
    assistant_aggregator,
    transport.output(),
])
```

### 步骤 3：创建 SkillIntegrationLayer

**v2.7（旧）**：
```python
# 创建 SkillManager
skill_manager = SkillManager(skills_dir)

# 创建 SkillProcessor
skill_proc = SkillProcessor(
    skill_manager=skill_manager,
    llm_service=llm,
    mcp_manager=mcp,
    auto_activate=True
)

# 手动触发启动
await skill_proc.start()
```

**v3.0（新）**：
```python
# 创建技能系统组件
registry = SkillRegistry(skills_dir)
matcher = SkillMatcher(registry)
skill_manager = SkillManager(skills_dir)

# 创建集成层
skill_integration_layer = SkillIntegrationLayer(
    registry=registry,
    matcher=matcher,
    skill_manager=skill_manager,
    llm_service=llm,
    mcp_manager=mcp,
    match_threshold=0.5,  # 匹配阈值
    max_active_skills=3   # 最多同时激活 3 个技能
)

# 初始化（发现技能、注册函数）
await skill_integration_layer.initialize()
```

### 步骤 4：添加 TranscriptionListener

**新增代码**：
```python
from pipecat.frames.frames import TranscriptionFrame, Frame
from pipecat.processors.frame_processor import FrameProcessor

class TranscriptionListener(FrameProcessor):
    """监听 ASR 识别结果，触发技能匹配"""

    def __init__(self, skill_layer: SkillIntegrationLayer):
        super().__init__()
        self.skill_layer = skill_layer

    async def process_frame(self, frame: Frame, direction: int):
        # 转发所有帧
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        # 监听 TranscriptionFrame
        if isinstance(frame, TranscriptionFrame):
            # ASR 识别文本后，触发技能匹配
            await self.skill_layer.process_user_input(frame.text)

# 创建监听器
transcription_listener = TranscriptionListener(skill_integration_layer)

# 动态插入到 Pipeline（在 user_aggregator 之后，vision_proc 之前）
pipeline._processors.insert(
    pipeline._processors.index(vision_proc),
    transcription_listener
)
```

### 步骤 5：更新 main 函数

**v2.7（旧）**：
```python
async def main():
    pipeline, transport, wake_system, mcp, skill_manager, skill_proc = await create_pipecat_pipeline()

    task = PipelineTask(pipeline, params=PipelineParams(...))

    # 设置 SkillProcessor 的 task
    await skill_proc.set_task(task)

    await task.queue_frames([StartFrame()])

    runner = PipelineRunner()
    runner_task = asyncio.create_task(runner.run(task))
    await runner_task
```

**v3.0（新）**：
```python
async def main():
    pipeline, transport, wake_system, mcp, skill_integration_layer = await create_pipecat_pipeline()

    task = PipelineTask(pipeline, params=PipelineParams(...))

    # 设置 SkillIntegrationLayer 的 task
    skill_integration_layer.set_task(task)

    # ... TranscriptionListener 代码 ...

    await task.queue_frames([StartFrame()])

    runner = PipelineRunner()
    runner_task = asyncio.create_task(runner.run(task))
    await runner_task
```

## 配置选项

### 匹配阈值

控制技能激活的灵敏度：

```python
# 高阈值：只激活高匹配度技能
skill_integration_layer.set_match_threshold(0.8)

# 低阈值：激活更多可能相关的技能
skill_integration_layer.set_match_threshold(0.3)
```

### 最大激活技能数

限制同时激活的技能数量：

```python
# 最多激活 5 个技能
skill_integration_layer.set_max_active_skills(5)

# 只激活最相关的 1 个技能
skill_integration_layer.set_max_active_skills(1)
```

### 自动激活开关

控制是否自动激活技能：

```python
# 禁用自动激活（手动控制）
skill_integration_layer.set_auto_activate(False)

# 启用自动激活（默认）
skill_integration_layer.set_auto_activate(True)
```

### 自定义匹配权重

调整匹配策略的权重：

```python
# 获取当前权重
weights = skill_integration_layer.matcher.get_weights()

# 设置新权重
skill_integration_layer.matcher.set_weights({
    "exact_name_match": 1.0,
    "keyword_in_name": 0.9,
    "keyword_in_desc": 0.6,  # 提高描述匹配权重
    "tag_match": 0.7,        # 提高标签匹配权重
})
```

## 测试迁移

### 1. 测试技能发现

```python
# 验证技能已发现
status = await skill_integration_layer.get_status()
print(f"Total skills: {status['registry']['total_skills']}")
print(f"Active skills: {status['registry']['active_skills']}")
```

### 2. 测试意图匹配

```python
# 测试匹配
results = await skill_integration_layer.matcher.match_skills("今天天气怎么样")
for result in results:
    print(f"{result.skill_name}: {result.score:.2f}")
```

### 3. 测试技能激活

```python
# 手动激活
success = await skill_integration_layer.manually_activate_skill("weather")
print(f"Activated: {success}")

# 检查状态
if skill_integration_layer.registry.is_active("weather"):
    print("Weather skill is active")
```

### 4. 测试完整流程

```python
# 模拟用户输入
await skill_integration_layer.process_user_input("今天北京天气怎么样")

# 验证技能已激活
assert skill_integration_layer.registry.is_active("weather")
```

## 常见问题

### Q1: 迁移后技能不激活？

**A**: 检查以下几点：
1. 匹配阈值是否过高（降低 `match_threshold`）
2. 技能元数据是否包含相关关键词
3. `auto_activate` 是否启用
4. `TranscriptionListener` 是否正确插入 Pipeline

### Q2: Pipeline 启动失败？

**A**: 检查以下几点：
1. 所有导入是否正确
2. `skill_integration_layer.initialize()` 是否调用
3. `TranscriptionListener` 是否正确插入
4. 查看错误堆栈信息

### Q3: 技能匹配不准确？

**A**: 优化方法：
1. 调整匹配权重
2. 完善 SKILL.md 中的关键词和描述
3. 调整匹配阈值
4. 增加技能标签

### Q4: 性能下降？

**A**: 优化方法：
1. 启用匹配缓存（默认开启）
2. 限制最大激活技能数
3. 提高匹配阈值，减少激活次数

### Q5: 如何调试？

**A**: 启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看匹配结果
results = await skill_integration_layer.matcher.match_skills("测试")
for result in results:
    print(f"Match: {result.skill_name} (score: {result.score:.2f}, reason: {result.match_reason})")

# 查看激活状态
status = await skill_integration_layer.get_status()
print(status)
```

## 回滚方案

如果迁移失败，可以回滚到 v2.7：

```python
# 恢复 v2.7 导入
from src.voice_assistant.skills import SkillManager, SkillProcessor

# 恢复 SkillProcessor
skill_proc = SkillProcessor(
    skill_manager=skill_manager,
    llm_service=llm,
    mcp_manager=mcp,
    auto_activate=True
)

# 恢复 Pipeline
pipeline = Pipeline([
    transport.input(),
    kws_proc,
    asr_proc,
    user_aggregator,
    vision_proc,
    skill_proc,  # 恢复
    llm,
    tts_proc,
    assistant_aggregator,
    transport.output(),
])
```

## 总结

迁移到 v3.0 的主要好处：

1. ✅ **零干扰**：不影响 Pipeline 帧流
2. ✅ **智能激活**：基于意图动态激活
3. ✅ **更高准确性**：减少误触发
4. ✅ **更好性能**：按需加载，节省资源

迁移成本：

- 代码改动：~50 行
- 测试时间：1-2 小时
- 学习曲线：中等

建议：

1. 先在开发环境测试
2. 逐步迁移，保留 v2.7 作为备份
3. 充分测试后再部署到生产环境

## 参考资料

- [Agent Skills v3.0 架构文档](./agent-skills-v3-architecture.md)
- [Agent Skills 集成指南](./agent-skills-integration-v3.md)
- [Pipecat 官方文档](https://github.com/pipecat-ai/pipecat)
