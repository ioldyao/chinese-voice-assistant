# Agent Skills v3.0 - 智能激活架构

## 概述

Agent Skills v3.0 是对技能系统的重大架构升级，解决了 v2.7 版本中的核心问题：

### v2.7 的问题

1. **FrameProcessor 冲突**：`SkillProcessor` 作为 Pipecat FrameProcessor 在 Pipeline 中导致帧流混乱
2. **批量激活问题**：`auto_activate=True` 导致所有技能同时激活，系统指令批量注入 LLM Context
3. **唤醒词误触发**：过多的技能指令影响 LLM 判断，导致非预期的系统行为
4. **缺乏智能筛选**：没有基于用户意图的动态技能选择机制

### v3.0 的改进

1. ✅ **移除 FrameProcessor 集成**：技能系统独立于 Pipeline 运行
2. ✅ **智能激活机制**：基于用户意图动态激活相关技能
3. ✅ **零干扰原则**：不影响现有 Pipeline 流程和唤醒词检测
4. ✅ **渐进式加载**：保留 Progressive Disclosure 机制（metadata → activation → execution）
5. ✅ **平滑迁移**：最小化代码改动，复用现有组件

## 核心架构

### 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│              用户输入处理（ASR 识别后）                         │
│                     用户文本                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│         SkillIntegrationLayer（新增）                        │
│  - 协调所有组件                                               │
│  - process_user_input() ← 核心入口                           │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼──────┐ ┌──▼──────┐  ┌──▼──────────┐
│ SkillMatcher │ │Registry│  │SkillManager │
│  （新增）    │ │（新增） │  │  （保留）   │
│ 意图匹配引擎  │ │元数据   │  │生命周期管理 │
└──────────────┘ └─────────┘  └─────────────┘
        │            │            │
        └────────────┼────────────┘
                     │
        ┌────────────▼────────────┐
        │   LLM Service 层         │
        │  register_function()     │
        │  （现有机制）            │
        └──────────────────────────┘
```

### 核心工作流程

```
1. 用户说话
   ↓
2. ASR 识别文本
   ↓
3. user_aggregator 添加到 context
   ↓
4. TranscriptionListener 监听 TranscriptionFrame
   ↓
5. SkillIntegrationLayer.process_user_input(text)
   ├─ SkillMatcher.match_skills() → 匹配相关技能
   ├─ SkillManager.activate_skill() → 激活高评分技能
   └─ 注入技能指令到 context
   ↓
6. LLM 生成回复
   ├─ 可能调用 skill_xxx 函数
   └─ SkillManager.execute_skill() → 执行技能
   ↓
7. 返回结果给用户
```

## 组件详解

### 1. SkillRegistry（技能注册表）

**职责**：管理所有技能的元数据和查询接口

**核心功能**：
- 发现和加载技能元数据（Discovery 阶段）
- 提供技能查询接口
- 管理激活状态
- 提供技能统计信息

**与 SkillManager 的区别**：
- `SkillRegistry`: 只负责元数据管理，不涉及执行
- `SkillManager`: 负责完整的生命周期管理（发现 → 激活 → 执行）

**API 示例**：
```python
registry = SkillRegistry(Path("skills"))
await registry.discover_all()

# 查询技能
skill = registry.get_skill("weather")

# 检查激活状态
if registry.is_active("weather"):
    print("Weather skill is active")

# 获取激活的技能
active_skills = registry.get_active_skills()
```

### 2. SkillMatcher（意图匹配引擎）

**职责**：基于用户输入智能匹配技能

**匹配策略**：
1. **精确匹配**：技能名称完全匹配（权重: 1.0）
2. **关键词匹配**：在显示名称、描述、标签中查找关键词（权重: 0.5 ~ 0.8）
3. **标签匹配**：匹配技能标签（权重: 0.6）
4. **类别匹配**：匹配技能类别（权重: 0.4）

**评分机制**：
- 每种匹配策略有独立的权重
- 综合计算最高评分作为最终匹配度
- 支持自定义权重配置

**API 示例**：
```python
matcher = SkillMatcher(registry)

# 匹配用户输入
results = await matcher.match_skills("今天北京天气怎么样")
for result in results:
    print(f"{result.skill_name}: {result.score:.2f} - {result.match_reason}")

# 自配置
results = await matcher.match_skills(
    "查询天气",
    top_k=5,          # 最多返回 5 个结果
    threshold=0.3     # 最低评分 0.3
)
```

### 3. SkillIntegrationLayer（统一集成层）

**职责**：协调所有组件，提供统一的接口

**核心方法**：
- `initialize()`: 初始化（发现技能、注册函数）
- `process_user_input()`: 处理用户输入（意图匹配、技能激活、指令注入）
- `manually_activate_skill()`: 手动激活技能
- `deactivate_skill()`: 停用技能

**配置选项**：
- `auto_activate`: 是否自动激活技能（默认 True）
- `match_threshold`: 匹配阈值（默认 0.5）
- `max_active_skills`: 最大同时激活技能数（默认 3）
- `enable_skill_instructions`: 是否注入技能指令（默认 True）

**API 示例**：
```python
layer = SkillIntegrationLayer(
    registry=registry,
    matcher=matcher,
    skill_manager=skill_manager,
    llm_service=llm,
    mcp_manager=mcp,
    match_threshold=0.5,
    max_active_skills=3
)

# 初始化
await layer.initialize()

# 处理用户输入（自动匹配和激活技能）
await layer.process_user_input("今天北京天气怎么样")

# 手动激活
await layer.manually_activate_skill("weather")

# 停用技能
await layer.deactivate_skill("weather")
```

## Pipeline 集成

### 关键改动

**v2.7（旧）**：
```python
# ❌ 旧方式：SkillProcessor 在 Pipeline 中
pipeline = Pipeline([
    transport.input(),
    kws_proc,
    asr_proc,
    user_aggregator,
    vision_proc,
    skill_proc,  # ← 问题所在
    llm,
    tts_proc,
    assistant_aggregator,
    transport.output(),
])
```

**v3.0（新）**：
```python
# ✅ 新方式：移除 SkillProcessor
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

# ✅ 使用 TranscriptionListener 监听 ASR 结果
class TranscriptionListener(FrameProcessor):
    def __init__(self, skill_layer: SkillIntegrationLayer):
        super().__init__()
        self.skill_layer = skill_layer

    async def process_frame(self, frame: Frame, direction: int):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # ASR 识别文本后，触发技能匹配
            await self.skill_layer.process_user_input(frame.text)

# 动态插入监听器到 Pipeline
transcription_listener = TranscriptionListener(skill_integration_layer)
pipeline._processors.insert(
    pipeline._processors.index(vision_proc),
    transcription_listener
)
```

### 事件流程

```
Audio Frame
  ↓
KWS Processor (唤醒词检测)
  ↓
ASR Processor (语音识别)
  ↓
User Aggregator (添加到 context)
  ↓
TranscriptionFrame (ASR 结果) ← TranscriptionListener 监听
  ├─ trigger: process_user_input(text)
  ├─ SkillMatcher.match_skills()
  ├─ SkillManager.activate_skill()
  └─ LLMMessagesAppendFrame (注入技能指令)
  ↓
Vision Processor (视觉理解)
  ↓
LLM Service (生成回复)
  ├─ 可能调用 skill_xxx 函数
  └─ SkillManager.execute_skill()
  ↓
TTS Processor (语音合成)
  ↓
Assistant Aggregator (保存回复)
  ↓
Audio Output
```

## 性能优化

### 技能匹配缓存

`SkillMatcher` 支持结果缓存，避免重复计算：

```python
matcher = SkillMatcher(registry, enable_cache=True)

# 第一次匹配：计算并缓存
results1 = await matcher.match_skills("天气")

# 第二次匹配：从缓存读取
results2 = await matcher.match_skills("天气")

# 清空缓存
matcher.clear_cache()
```

### 渐进式加载

保留 Progressive Disclosure 机制：

1. **Discovery**（启动时）: 仅加载 metadata，不加载完整指令
2. **Activation**（匹配时）: 加载完整指令，注册为 LLM 函数
3. **Execution**（调用时）: 执行技能逻辑

### 激活策略

- **智能激活**：仅激活评分 > 阈值的技能
- **限制数量**：最多同时激活 N 个技能（默认 3）
- **避免重复**：已激活的技能不会重复激活

## 验证标准

### 功能验证

- ✅ 技能发现和加载正常
- ✅ 意图匹配准确率 > 80%
- ✅ 技能动态激活/停用
- ✅ LLM 可以调用技能函数
- ✅ 不影响唤醒词检测

### 性能验证

- ✅ 技能匹配响应 < 100ms
- ✅ 内存增量 < 50MB
- ✅ Pipeline 延迟无增加

### 质量验证

- ✅ 单元测试覆盖率 > 80%
- ✅ 所有测试通过
- ✅ 无严重 Bug

## 向后兼容

### v2.7 迁移

**保留 `SkillProcessor`**（标记废弃）：

```python
from src.voice_assistant.skills import SkillProcessor

# ⚠️ 已废弃，请使用 SkillIntegrationLayer
processor = SkillProcessor(manager, llm_service, mcp_manager)
```

**迁移步骤**：

1. 移除 `SkillProcessor` 从 Pipeline
2. 创建 `SkillIntegrationLayer`
3. 添加 `TranscriptionListener` 监听 ASR 结果
4. 测试技能激活和执行

详见：[Agent Skills v3.0 迁移指南](./agent-skills-migration-v3.md)

## 参考资料

### 现有代码
- `src/voice_assistant/skills/skill_manager.py` - Progressive Discovery 机制
- `src/voice_assistant/skills/base_skill.py` - AgentSkill 基类
- `src/voice_assistant/qwen_llm_service.py` - LLM 函数注册机制

### 外部参考
- Agent Skills 开放标准：https://agentskills.io/
- Pipecat 官方文档：https://github.com/pipecat-ai/pipecat

### 测试文件
- `tests/test_skill_registry.py` - SkillRegistry 测试
- `tests/test_skill_matcher.py` - SkillMatcher 测试
- `tests/test_skill_integration.py` - SkillIntegrationLayer 测试
