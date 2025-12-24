# Qwen LLM Service - Pipecat 官方框架集成指南

## 概述

`QwenLLMService` 是基于 Pipecat 官方框架的 Qwen-Plus 集成，完全兼容 OpenAI API 格式，享受 Pipecat 官方生态的所有优势。

## 核心优势

✅ **符合官方最佳实践** - 继承 `OpenAILLMService`
✅ **自动管理对话历史** - 使用 `LLMContextAggregatorPair`
✅ **支持 Function Calling** - MCP 工具无缝集成
✅ **完全异步** - 纯异步架构，无线程开销
✅ **保持 Qwen 生态** - 中文效果好，使用现有 API Key

---

## 快速开始

### 1. 基础使用

```python
from src.voice_assistant import (
    QwenLLMService,
    QwenLLMContext,
    create_tools_schema_from_mcp,
    register_mcp_functions
)
from pipecat.processors.aggregators.llm_response import LLMContextAggregatorPair

# 初始化 Qwen LLM Service
llm = QwenLLMService(
    model="qwen-plus",  # 或 qwen-max, qwen-turbo
)

# 创建对话上下文
messages = [
    {"role": "system", "content": "你是一个智能助手"}
]
context = QwenLLMContext(messages)

# 创建 Context Aggregator（官方）
context_aggregator = LLMContextAggregatorPair(context)
```

### 2. 集成 MCP 工具

```python
from src.voice_assistant import MCPManager

# 初始化 MCP Manager
mcp = MCPManager()

# 启动 MCP Servers（异步）
await mcp.add_server_async(
    "playwright",
    "npx",
    ["@playwright/mcp@latest"],
    timeout=120
)

# 获取 MCP 工具列表
mcp_tools = await mcp.list_all_tools_async()

# 转换为 Pipecat FunctionSchema
tools_schema = create_tools_schema_from_mcp(mcp_tools)

# 转换为 OpenAI API 格式（用于 LLMContext）
tools = mcp_tools_to_openai_format(mcp_tools)

# 添加到上下文
context = QwenLLMContext(messages, tools=tools)

# 注册 MCP 函数处理器（统一处理所有 MCP 工具调用）
await register_mcp_functions(llm, mcp)
```

### 3. 构建 Pipeline（混合架构）

```python
from pipecat.pipeline.pipeline import Pipeline
from src.voice_assistant.pipecat_adapters import (
    SherpaKWSProcessor,      # 自定义：KWS 唤醒词
    SherpaASRProcessor,      # 自定义：ASR 本地识别
    ScreenshotProcessor,     # 自定义：截图
    QwenVisionProcessor,     # 自定义：Qwen Vision API
    PiperTTSProcessor,       # 自定义：Piper 本地 TTS
)

# 混合 Pipeline：自定义 + 官方
pipeline = Pipeline([
    # 自定义部分（官方不支持）
    kws_proc,                       # KWS 唤醒词检测
    asr_proc,                       # ASR 本地识别

    # 官方部分（完全采用 Pipecat 框架）
    context_aggregator.user(),      # 官方：添加用户消息到上下文

    # Vision 处理（自定义，保留 Qwen-VL-Max）
    screenshot_proc,                # 截图 → UserImageRawFrame
    qwen_vision_proc,               # Vision API → TextFrame

    llm,                            # 官方：Qwen LLM Service（已注册 MCP 函数）

    context_aggregator.assistant(), # 官方：保存助手响应到上下文

    # 自定义部分（本地、免费）
    piper_tts_proc,                 # Piper TTS（本地）

    transport.output(),
])
```

---

## 完整示例代码

```python
import asyncio
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner

from src.voice_assistant import (
    QwenLLMService,
    QwenLLMContext,
    MCPManager,
    create_tools_schema_from_mcp,
    register_mcp_functions,
    SmartWakeWordSystem,
)
from src.voice_assistant.pipecat_adapters import (
    SherpaKWSProcessor,
    SherpaASRProcessor,
    ScreenshotProcessor,
    QwenVisionProcessor,
    PiperTTSProcessor,
)
from pipecat.processors.aggregators.llm_response import LLMContextAggregatorPair


async def main():
    # 1. 初始化模型加载器
    wake_system = SmartWakeWordSystem(enable_voice=False, enable_mcp=False)

    # 2. 初始化 MCP
    mcp = MCPManager()
    await mcp.add_server_async("playwright", "npx", ["@playwright/mcp@latest"], 120)
    mcp_tools = await mcp.list_all_tools_async()
    tools = mcp_tools_to_openai_format(mcp_tools)

    # 3. 初始化 Qwen LLM Service（官方）
    llm = QwenLLMService(model="qwen-plus")

    # 4. 注册 MCP 函数处理器
    await register_mcp_functions(llm, mcp)

    # 5. 创建对话上下文（官方）
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以使用浏览器工具"}
    ]
    context = QwenLLMContext(messages, tools=tools)
    context_aggregator = LLMContextAggregatorPair(context)

    # 6. 创建 Processors
    kws_proc = SherpaKWSProcessor(wake_system.kws_model)
    asr_proc = SherpaASRProcessor(wake_system.asr_model)
    screenshot_proc = ScreenshotProcessor()
    qwen_vision_proc = QwenVisionProcessor(
        api_url=wake_system.agent.api_url,
        api_key=wake_system.agent.api_key
    )
    tts_proc = PiperTTSProcessor(wake_system.agent.tts, transport)

    # 7. 构建 Pipeline（混合架构）
    pipeline = Pipeline([
        kws_proc,                       # 自定义：KWS
        asr_proc,                       # 自定义：ASR
        context_aggregator.user(),      # 官方：用户消息
        screenshot_proc,                # 自定义：截图
        qwen_vision_proc,               # 自定义：Vision
        llm,                            # 官方：LLM + Function Calling
        context_aggregator.assistant(), # 官方：助手响应
        tts_proc,                       # 自定义：Piper TTS
        transport.output(),
    ])

    # 8. 创建 Task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,  # 启用中断
            enable_metrics=True,
        )
    )

    # 9. 运行 Pipeline
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 架构说明

### 保留自定义（官方不支持）
- ✅ **KWS 唤醒词检测** - Sherpa-ONNX（本地、免费）
- ✅ **ASR 语音识别** - Sherpa-ONNX（本地、免费）
- ✅ **Piper TTS** - 本地语音合成（超低延迟）
- ✅ **Qwen Vision** - Qwen-VL-Max（保持现有 API）

### 改用官方（享受生态）
- ✨ **LLM Service** - QwenLLMService（继承 OpenAILLMService）
- ✨ **Context Aggregator** - LLMContextAggregatorPair（自动管理历史）
- ✨ **Function Calling** - MCP 工具无缝集成
- ✨ **异步架构** - 完全异步，无线程

---

## API 参考

### QwenLLMService

```python
QwenLLMService(
    *,
    api_key: str | None = None,       # DashScope API Key
    base_url: str | None = None,      # DashScope API URL
    model: str = "qwen-plus",         # qwen-plus, qwen-max, qwen-turbo
    **kwargs                          # 传递给 OpenAILLMService 的其他参数
)
```

### QwenLLMContext

```python
QwenLLMContext(
    messages: list[dict] | None = None,  # 对话历史（OpenAI 格式）
    tools: list[dict] | None = None       # Function Calling 工具（OpenAI 格式）
)
```

### mcp_tools_to_function_schemas

```python
mcp_tools_to_function_schemas(
    mcp_tools: list[dict]  # MCP 工具列表
) -> list[FunctionSchema]   # Pipecat FunctionSchema 列表
```

### create_tools_schema_from_mcp

```python
create_tools_schema_from_mcp(
    mcp_tools: list[dict]  # MCP 工具列表
) -> ToolsSchema            # Pipecat ToolsSchema 对象
```

### mcp_tools_to_openai_format (推荐用于 LLMContext)

```python
mcp_tools_to_openai_format(
    mcp_tools: list[dict]  # MCP 工具列表
) -> list[dict]             # OpenAI API 格式的工具列表
```

**用途**: 将 MCP 工具转换为 OpenAI API 格式，用于 `QwenLLMContext` 的 `tools` 参数。

### register_mcp_functions

```python
await register_mcp_functions(
    llm_service: QwenLLMService,  # Qwen LLM Service 实例
    mcp_manager: MCPManager       # MCPManager 实例
)
```

---

## 下一步

- [ ] 修改 `pipecat_main.py` 使用新架构
- [ ] 移除 `ReactAgent` 的手动历史管理
- [ ] 测试完整功能（KWS → ASR → Vision → LLM + Function Calling → TTS）
- [ ] 性能对比（官方 vs 自定义）
- [ ] 文档完善

---

## 常见问题

### Q: 为什么不完全使用官方组件？

A:
- **KWS + ASR** - 官方不支持唤醒词检测和本地识别
- **Piper TTS** - 官方支持 Cartesia/ElevenLabs，但 Piper 本地免费
- **Qwen Vision** - 保持现有 API Key，避免迁移成本

### Q: 如何切换到 GPT-4o Vision？

A:
1. 移除 `ScreenshotProcessor` 和 `QwenVisionProcessor`
2. 改用 `OpenAILLMService(model="gpt-4o")`
3. LLM 原生支持图片，无需单独 Vision Processor

### Q: Function Calling 如何工作？

A:
1. MCP 工具转换为 FunctionSchema
2. 添加到 LLMContext
3. 注册统一的 MCP 函数处理器
4. LLM 自动调用工具 → 返回结果 → 继续对话

---

**版本**: v2.1.0
**作者**: Claude Sonnet 4.5
**日期**: 2025-12-24
