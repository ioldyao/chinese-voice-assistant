# Agent Skills 编写指南

本文档说明如何编写高质量的 Agent Skills。

## 目录

- [概述](#概述)
- [SKILL.md 文件格式](#skillmd-文件格式)
- [YAML Frontmatter](#yaml-frontmatter)
- [Markdown 内容](#markdown-内容)
- [最佳实践](#最佳实践)
- [示例模板](#示例模板)
- [高级功能](#高级功能)

## 概述

Agent Skills 使用简单的 Markdown 文件定义，包含：

1. **YAML Frontmatter**: 技能元数据
2. **Markdown 内容**: 技能指令和说明

这种设计使得技能易于编写、版本控制和共享。

## SKILL.md 文件格式

### 基本结构

```markdown
---
name: skill_name
display_name: 技能显示名称
description: 技能描述
version: 1.0.0
author: 作者名称
tags: [tag1, tag2]
category: utility
requires_tools: []
---

# 技能标题

你是一个专业的助手，可以...

## 能力

- 能力1
- 能力2

## 使用方法

当用户...时...
```

### 文件位置

```
skills/
├── weather/
│   └── SKILL.md          ← 天气技能
├── calendar/
│   └── SKILL.md          ← 日历技能
└── my_skill/
    ├── SKILL.md          ← 主技能文件
    ├── scripts/          ← 可选：脚本文件
    └── references/       ← 可选：参考资料
```

## YAML Frontmatter

### 必需字段

```yaml
---
name: weather              # 技能唯一标识符（英文，小写）
display_name: 天气查询      # 显示名称（中文）
description: 查询全球各地的天气信息  # 简短描述
---
```

### 可选字段

```yaml
---
version: 1.0.0             # 版本号（语义化版本）
author: AI Assistant       # 作者
tags: [weather, api, utility]  # 标签（用于分类和搜索）
category: utility          # 类别（utility/productivity/entertainment）
requires_tools: []         # 需要的 MCP 工具列表
---
```

### 字段说明

#### name
- **类型**: 字符串
- **必需**: 是
- **格式**: 英文小写，可以用下划线
- **示例**: `weather`, `calendar_manager`, `email_sender`

#### display_name
- **类型**: 字符串
- **必需**: 是
- **格式**: 中文或英文
- **示例**: `天气查询`, `日历管理`, `邮件发送`

#### description
- **类型**: 字符串
- **必需**: 是
- **格式**: 简短描述（一句话）
- **示例**: `查询全球各地的天气信息，包括当前天气、温度、湿度等`

#### version
- **类型**: 字符串
- **必需**: 否
- **格式**: 语义化版本（major.minor.patch）
- **示例**: `1.0.0`, `2.1.3`

#### author
- **类型**: 字符串
- **必需**: 否
- **示例**: `AI Assistant`, `Your Name`

#### tags
- **类型**: 列表
- **必需**: 否
- **用途**: 分类和搜索
- **常用标签**:
  - `api`: 使用外部 API
  - `browser`: 浏览器操作
  - `mcp`: 使用 MCP 工具
  - `utility`: 实用工具
  - `productivity`: 生产力
  - `entertainment`: 娱乐

#### category
- **类型**: 字符串
- **必需**: 否
- **常用类别**:
  - `utility`: 实用工具
  - `productivity`: 生产力
  - `entertainment`: 娱乐
  - `information`: 信息查询
  - `automation`: 自动化

#### requires_tools
- **类型**: 列表
- **必需**: 否
- **用途**: 声明需要的 MCP 工具
- **示例**: `["browser_navigate", "browser_click"]`

## Markdown 内容

### 推荐结构

```markdown
# 技能标题

简短描述技能的功能。

## 能力

列出技能的主要能力：
- 能力1
- 能力2
- 能力3

## 使用方法

详细说明如何使用这个技能。

### 步骤1：...

### 步骤2：...

## 示例对话

提供一些示例对话。

**用户**: ...
**助手**: ...

## 注意事项

- 注意事项1
- 注意事项2

## 限制

说明技能的局限性。
```

### 写作技巧

1. **明确角色**: 用"你是..."定义角色
2. **具体说明**: 提供详细的使用方法
3. **示例对话**: 展示典型的交互场景
4. **注意事项**: 说明重要的注意事项
5. **简洁清晰**: 避免冗余，保持简洁

### 格式建议

- 使用 `##` 标记主要章节
- 使用 `###` 标记子章节
- 使用列表（`-`）列出多个项目
- 使用 `**粗体**` 强调重要内容
- 使用代码块引用示例

## 最佳实践

### 1. 技能命名

✅ **好的命名**:
- `weather` - 天气查询
- `calendar_manager` - 日历管理
- `email_sender` - 邮件发送

❌ **不好的命名**:
- `Weather` - 使用了大写
- `weather-skill` - 使用了连字符
- `myWeatherSkill` - 使用了驼峰命名

### 2. 描述撰写

✅ **好的描述**:
- "查询全球各地的天气信息，包括当前天气、温度、湿度等"

❌ **不好的描述**:
- "天气" - 太简短
- "这是一个天气查询技能，可以查询天气，还能查询温度" - 冗余

### 3. 标签使用

✅ **合理的标签**:
```yaml
tags: [weather, api, utility]
```

❌ **不合理的标签**:
```yaml
tags: [a, b, c, d, e, f, g]  # 太多
tags: [天气, 查询, 助手]      # 使用中文
```

### 4. 内容组织

✅ **好的组织**:
```markdown
# 天气查询技能

你是一个专业的天气助手。

## 能力
- 查询当前天气
- 提供穿衣建议

## 使用方法
当用户询问天气时...
```

❌ **不好的组织**:
```markdown
天气

可以查天气，还有温度，也可以查湿度，然后还能...
```

## 示例模板

### 模板 1: API 调用技能

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

你是一个专业的天气助手，可以查询全球各地的天气信息。

## 能力

- 查询当前天气
- 查询温度、湿度、风速
- 提供穿衣建议
- 预报未来天气

## 使用方法

当用户询问天气时：
1. 识别用户询问的城市名称
2. 确认需要查询的具体信息
3. 用自然的中文回复用户

## 示例对话

**用户**: 今天北京天气怎么样？
**助手**: 让我为您查询一下北京的天气...

**用户**: 上海现在多少度？
**助手**: 上海当前温度为 25°C，体感舒适。

## 注意事项

- 如果用户没有指定城市，礼貌地询问
- 使用简洁、友好的语言
- 提供实用的建议
```

### 模板 2: 浏览器控制技能

```markdown
---
name: browser_control
display_name: 浏览器控制
description: 使用浏览器进行网页导航、点击、输入等操作
version: 1.0.0
author: AI Assistant
tags: [browser, playwright, mcp, automation]
category: utility
requires_tools: ["browser_navigate", "browser_click", "browser_snapshot"]
---

# 浏览器控制技能

你是一个专业的浏览器控制助手，可以使用工具操作浏览器。

## 可用工具

- **browser_navigate**: 导航到指定 URL
- **browser_snapshot**: 获取页面快照
- **browser_click**: 点击页面元素

## 工作流程

1. 调用 browser_navigate 打开网页
2. 调用 browser_snapshot 查看页面
3. 使用 browser_click 进行操作

## 示例对话

**用户**: 打开百度
**助手**: 好的，我来为您打开百度首页。

**用户**: 点击搜索框
**助手**: 已点击搜索框。

## 注意事项

- 点击元素前必须先调用 browser_snapshot
- 使用快照中的 ref 进行点击
- 如果 ref 失效，重新获取快照
```

### 模板 3: 生产力技能

```markdown
---
name: calendar_manager
display_name: 日历管理
description: 管理日历和日程安排
version: 1.0.0
author: AI Assistant
tags: [calendar, productivity, scheduling]
category: productivity
requires_tools: []
---

# 日历管理技能

你是一个专业的日历助手，帮助用户管理日程安排。

## 能力

- 创建日程事件
- 查询已有日程
- 删除或修改事件
- 提醒即将到来的事项

## 使用方法

当用户需要管理日程时：
1. 理解用户的意图
2. 提取关键信息（时间、事件名称）
3. 确认操作并执行

## 示例对话

**用户**: 帮我安排明天下午3点的会议
**助手**: 好的，我来帮您创建日程。请告诉我会议的主题？

**用户**: 查询这周的日程
**助手**: 让我为您查看本周的日程安排...

## 注意事项

- 时间要明确（日期、具体时间）
- 事件名称要清晰
- 重要事项建议用户添加提醒
```

## 高级功能

### 与 MCP 工具集成

在 YAML frontmatter 中声明需要的工具：

```yaml
---
requires_tools: ["browser_navigate", "browser_click", "browser_fill"]
---
```

技能执行时会自动调用这些工具。

### 自定义执行逻辑

如果需要更复杂的逻辑，可以创建 Python 类：

```python
from src.voice_assistant.skills import AgentSkill, SkillResult

class CustomSkill(AgentSkill):
    async def execute(self, context):
        # 自定义逻辑
        result = await some_api_call(context.user_input)

        return SkillResult(
            success=True,
            content=result,
            metadata={"source": "custom_api"}
        )
```

### 多语言支持

SKILL.md 支持中文和英文：

```markdown
## 能力 / Capabilities

- 查询天气 / Query weather
- 提供建议 / Provide suggestions
```

## 测试技能

### 验证格式

```python
from src.voice_assistant.skills import SkillLoader

loader = SkillLoader(Path("skills"))
is_valid, errors = loader.validate_skill_directory(Path("skills/my_skill"))

if not is_valid:
    for error in errors:
        print(f"❌ Error: {error}")
else:
    print("✅ Skill format is valid")
```

### 测试执行

```python
from src.voice_assistant.skills import SkillManager

manager = SkillManager(Path("skills"))
await manager.discover_all()
await manager.activate_skill("my_skill")

result = await manager.execute_skill(
    "my_skill",
    user_input="测试输入"
)

if result.success:
    print(result.content)
else:
    print(f"❌ Error: {result.error}")
```

## 常见问题

### Q: SKILL.md 支持哪些 Markdown 语法？

A: 支持标准 Markdown 语法，包括：
- 标题（#, ##, ###）
- 列表（-, *, 1.）
- 粗体（**text**）
- 代码块（```）
- 引用（>）

### Q: 可以在 SKILL.md 中使用图片吗？

A: 可以，但建议使用外部链接：
```markdown
![示例图片](https://example.com/image.png)
```

### Q: 技能可以调用其他技能吗？

A: 目前不支持技能间的直接调用。每个技能是独立的。

### Q: 如何更新技能？

A: 直接编辑 SKILL.md 文件，然后重新加载：
```python
await manager.reload_skill("my_skill")
```

## 相关文档

- [集成指南](./agent-skills-integration.md)
- [AgentSkills.io 官方文档](https://agentskills.io/)
