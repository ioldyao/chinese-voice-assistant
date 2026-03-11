# Pipecat Flows 集成指南 (v2.3.0)

## 概述

本项目现已集成 **Pipecat Flows**，提供强大的对话流程管理能力。集成采用**混合架构**设计，在保留所有现有功能的基础上，为复杂对话场景提供更好的支持。

## 架构特点

### 1. 混合架构

```
简单指令（快速路径）         复杂对话（流程路径）
     ↓                           ↓
SimpleCommandProcessor      FlowManager
     ↓                           ↓
直接执行 MCP 工具          多节点对话流程
     ↓                           ↓
  即时响应                   引导式对话
```

### 2. 核心优势

✅ **保留所有现有功能**
- 唤醒词检测（Sherpa-ONNX KWS）
- 语音识别（Sherpa-ONNX ASR）
- 视觉分析（Moondream/Qwen-VL）
- 语音合成（Piper TTS）
- MCP 工具集成

✅ **新增对话流程能力**
- 多步骤对话管理
- 状态持久化
- 动态节点转换
- 函数自动编排

✅ **双模式支持**
- 简单指令：快速执行（绕过流程，< 1秒响应）
- 复杂对话：引导式流程（多步收集信息）

## 安装

### 1. 安装 pipecat-ai-flows

```bash
pip install pipecat-ai-flows
```

或使用 uv：

```bash
uv pip install pipecat-ai-flows
```

### 2. 更新依赖

```bash
uv sync
```

## 使用方法

### 1. 启动 Flows 集成版

```bash
# 使用 uv 运行
uv run python src/voice_assistant/pipecat_flows_main.py

# 或直接运行
python src/voice_assistant/pipecat_flows_main.py
```

### 2. 对话示例

#### 场景 1：简单指令（快速路径）

```
用户: "小智"                   [唤醒]
用户: "打开百度"                [简单指令，直接执行]
系统: ✅ 快速导航成功: https://www.baidu.com
```

**特点**：
- 不经过 FlowManager
- 不调用 LLM
- 直接执行 MCP 工具
- 响应速度 < 1秒

#### 场景 2：复杂对话（流程路径）

```
用户: "小智"                   [唤醒]
系统: "你好！我是小智，请问你想浏览哪个网站？"
用户: "我想看看我们公司的官网"
系统: "请告诉我网址，或者说出网站名称"
用户: "example.com"
系统: 调用 navigate_to_url → 转到 analysis 节点
系统: "已经打开了网页，需要我讲解页面内容吗？"
用户: "是的"
系统: 调用 analyze_page_content → 截图 → Vision 分析 → TTS 讲解
系统: "这是一个企业官网，主要展示了..."
系统: "还需要浏览其他网站吗？"
```

**特点**：
- 经过 FlowManager 多节点流程
- 状态持久化（URL、截图路径等）
- 引导式对话
- 自动节点转换

## 对话流程详解

### 节点结构

```
initial (初始节点)
    ↓ navigate_to_url()
analysis (分析节点)
    ↓ analyze_page_content()
completion (完成节点)
    ↓ end_conversation() → 回到 initial
```

### 节点定义

#### 1. initial（初始节点）

**任务**：询问用户想要浏览什么网站

**可用函数**：
- `navigate_to_url(url)` - 导航到指定 URL
- `end_conversation()` - 结束对话

**转换条件**：
- 调用 `navigate_to_url` → 转到 `analysis` 节点

#### 2. analysis（分析节点）

**任务**：已打开网页，询问是否需要讲解

**可用函数**：
- `analyze_page_content()` - 分析并讲解页面内容
- `navigate_to_url(url)` - 导航到其他 URL
- `end_conversation()` - 结束对话

**转换条件**：
- 调用 `analyze_page_content` → 转到 `completion` 节点
- 调用 `navigate_to_url` → 停留在 `analysis` 节点

#### 3. completion（完成节点）

**任务**：讲解完成，询问是否还需要其他帮助

**可用函数**：
- `navigate_to_url(url)` - 导航到新 URL
- `analyze_page_content()` - 再次分析当前页面
- `end_conversation()` - 结束对话

**转换条件**：
- 调用 `navigate_to_url` → 转到 `analysis` 节点
- 调用 `analyze_page_content` → 停留在 `completion` 节点
- 调用 `end_conversation` → 回到 `initial` 节点

### 状态管理

FlowManager 在节点间维护以下状态：

```python
flow_manager.state = {
    "mcp_manager": MCPManager实例,
    "current_url": "https://example.com",
    "screenshot_path": "/path/to/screenshot.png",
    # 可以添加更多自定义状态
}
```

### 函数处理器示例

```python
async def navigate_to_url(
    args: FlowArgs,
    flow_manager: FlowManager
) -> ConsolidatedFunctionResult:
    """导航到 URL 并转换到分析节点"""

    url = args.get("url", "")

    # 1. 获取 MCP Manager
    mcp_manager = flow_manager.state.get("mcp_manager")

    # 2. 调用 MCP 工具
    result = await mcp_manager.call_tool("playwright_navigate", {"url": url})

    # 3. 保存状态
    if result.get("status") == "success":
        flow_manager.state["current_url"] = url

        # 4. 返回结果和下一个节点
        result_data = URLInputResult(url=url, status="success")
        next_node = create_analysis_node()
        return result_data, next_node
    else:
        # 失败时不转换节点
        result_data = URLInputResult(url=url, status="error")
        return result_data, None
```

## 简单指令处理器

### 支持的简单指令

| 指令          | 功能              | MCP 工具              |
| ------------- | ----------------- | --------------------- |
| "打开百度"    | 快速导航          | playwright_navigate   |
| "截图"        | 快速截图          | playwright_screenshot |
| "返回"        | 浏览器后退        | playwright_back       |

### 扩展简单指令

编辑 `SimpleCommandProcessor._handle_simple_command()` 方法：

```python
async def _handle_simple_command(self, text: str) -> bool:
    # 新增模式：刷新页面
    if "刷新" in text or "重新加载" in text:
        try:
            result = await self.mcp_manager.call_tool("playwright_reload", {})
            if result.get("status") == "success":
                logger.info(f"✅ 快速刷新成功")
                return True
        except Exception as e:
            logger.error(f"❌ 快速刷新失败: {e}")

    return False
```

## 自定义对话流程

### 1. 添加新节点

```python
def create_search_node() -> NodeConfig:
    """创建搜索节点：在当前页面进行搜索"""
    return {
        "name": "search",
        "task_messages": [
            {
                "role": "system",
                "content": "询问用户想搜索什么关键词"
            }
        ],
        "functions": [search_schema, navigate_schema, end_conversation_schema],
    }
```

### 2. 添加新函数

```python
# 定义函数处理器
async def search_keyword(
    args: FlowArgs,
    flow_manager: FlowManager
) -> ConsolidatedFunctionResult:
    """在页面中搜索关键词"""
    keyword = args.get("keyword", "")

    mcp_manager = flow_manager.state.get("mcp_manager")

    # 调用搜索工具
    result = await mcp_manager.call_tool("playwright_search", {"keyword": keyword})

    if result.get("status") == "success":
        result_data = {"status": "success", "keyword": keyword}
        next_node = create_analysis_node()
        return result_data, next_node
    else:
        result_data = {"status": "error"}
        return result_data, None

# 定义函数 Schema
search_schema = FlowsFunctionSchema(
    name="search_keyword",
    description="在当前页面搜索关键词",
    properties={
        "keyword": {
            "type": "string",
            "description": "要搜索的关键词"
        }
    },
    required=["keyword"],
    handler=search_keyword,
)
```

### 3. 修改节点转换逻辑

在函数处理器中，返回不同的 `next_node` 来控制流程走向：

```python
async def conditional_handler(
    args: FlowArgs,
    flow_manager: FlowManager
) -> ConsolidatedFunctionResult:
    """根据条件选择不同节点"""

    user_input = args.get("input", "")

    if "搜索" in user_input:
        next_node = create_search_node()
    elif "分析" in user_input:
        next_node = create_analysis_node()
    else:
        next_node = create_initial_node()

    result = {"status": "success"}
    return result, next_node
```

## 与现有版本对比

| 特性              | v2.2.1（现有版本）    | v2.3.0（Flows 版本） |
| ----------------- | --------------------- | -------------------- |
| 唤醒词检测        | ✅                     | ✅                    |
| 语音识别          | ✅                     | ✅                    |
| Vision 分析       | ✅                     | ✅                    |
| TTS 语音合成      | ✅                     | ✅                    |
| MCP 工具集成      | ✅                     | ✅                    |
| 简单指令          | ✅（通过 LLM）         | ✅（快速路径）        |
| 多步骤对话        | ❌                     | ✅                    |
| 状态持久化        | ❌                     | ✅                    |
| 对话流程管理      | ❌                     | ✅                    |
| 节点转换          | ❌                     | ✅                    |

## 技术架构

### Pipeline 结构

```
PyAudio Input
    ↓
KWS Processor (唤醒词检测)
    ↓
ASR Processor (语音识别)
    ↓
SimpleCommandProcessor (简单指令快速处理) ← 新增
    ↓
User Aggregator (添加到 context)
    ↓
Vision Processor (视觉分析)
    ↓
LLM (Qwen + MCP Tools)
    ↓
TTS Processor (Piper 语音合成)
    ↓
Assistant Aggregator (保存响应)
    ↓
PyAudio Output
```

### FlowManager 集成点

```
FlowManager → LLM (通过 Frame)
    ↓
LLMMessagesAppendFrame  (添加消息到 context)
LLMSetToolsFrame        (设置可用函数)
LLMRunFrame             (触发 LLM 推理)
```

### 数据流

```
用户语音
    ↓ VAD + ASR
转录文本 (TranscriptionFrame)
    ↓ SimpleCommandProcessor 检查
简单指令? → 是 → 直接执行 MCP → 返回
    ↓ 否
传递给 User Aggregator
    ↓
添加到 LLMContext
    ↓
Vision Processor (如果需要)
    ↓
LLM 推理 (调用 FlowManager 注册的函数)
    ↓
函数返回 (result, next_node)
    ↓
FlowManager 转换到 next_node
    ↓
新节点的 messages 和 functions 注入 LLM
    ↓
LLM 生成响应
    ↓
TTS 合成语音
    ↓
播放给用户
```

## 故障排查

### 问题 1：pipecat-flows 导入失败

**错误**：
```
ImportError: No module named 'pipecat_flows'
```

**解决方案**：
```bash
pip install pipecat-ai-flows
```

### 问题 2：函数未被调用

**可能原因**：
- LLM 未正确识别函数参数格式
- 函数 Schema 定义不清晰

**解决方案**：
1. 检查 `FlowsFunctionSchema` 的 `description` 是否清晰
2. 检查 `properties` 是否包含足够的类型信息
3. 查看 LLM 日志确认函数调用请求

### 问题 3：节点未转换

**可能原因**：
- 函数处理器返回了 `None` 作为 `next_node`
- 函数处理器抛出异常

**解决方案**：
1. 检查函数处理器的返回值
2. 添加异常处理和日志
3. 确认 `next_node` 是有效的 `NodeConfig`

### 问题 4：简单指令被 FlowManager 处理

**可能原因**：
- `SimpleCommandProcessor` 未正确拦截
- 模式匹配失败

**解决方案**：
1. 检查 `_handle_simple_command()` 的模式匹配逻辑
2. 添加日志确认是否进入简单指令分支
3. 确保 `SimpleCommandProcessor` 在 Pipeline 中的位置正确

## 性能优化建议

### 1. 简单指令优先

对于高频简单操作（如"打开百度"），优先使用 `SimpleCommandProcessor` 快速路径，避免 LLM 调用开销。

### 2. 状态缓存

在 `flow_manager.state` 中缓存常用数据，减少重复计算：

```python
# 缓存截图路径
flow_manager.state["screenshot_cache"] = {
    "url1": "/path/to/screenshot1.png",
    "url2": "/path/to/screenshot2.png",
}
```

### 3. 异步并发

在函数处理器中使用 `asyncio.gather` 并发调用多个 MCP 工具：

```python
async def batch_operations(args, flow_manager):
    mcp_manager = flow_manager.state.get("mcp_manager")

    # 并发执行多个操作
    results = await asyncio.gather(
        mcp_manager.call_tool("tool1", {}),
        mcp_manager.call_tool("tool2", {}),
        mcp_manager.call_tool("tool3", {}),
    )

    return {"status": "success"}, next_node
```

## 下一步计划

- [ ] 添加更多预定义节点（搜索、表单填写、数据提取）
- [ ] 支持分支流程（if-else 逻辑）
- [ ] 实现流程可视化工具
- [ ] 添加流程回放和调试功能
- [ ] 支持多用户并发对话流程

## 参考资料

- [Pipecat Flows 官方文档](https://github.com/pipecat-ai/pipecat-flows)
- [Pipecat 官方文档](https://github.com/pipecat-ai/pipecat)
- [当前项目 README](../README.md)
