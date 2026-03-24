# Anthropic LLM 服务集成指南

本文档说明如何在中文语音助手中使用 Anthropic Messages API 兼容的 LLM 服务。

## 支持的服务

1. **官方 Claude API**（Anthropic）
2. **智谱 GLM API**（兼容 Anthropic Messages API）
3. 任何兼容 Anthropic Messages API 规范的服务

## 架构说明

使用 **Pipecat 官方 AnthropicLLMService**，无需自定义代码：

- ✅ 使用官方 `pipecat.services.anthropic.llm.AnthropicLLMService`
- ✅ 使用统一的 `LLMContext` 和 `LLMContextAggregatorPair`
- ✅ 所有 LLM 服务（OpenAI、Anthropic）使用相同的架构
- ✅ 支持 Function Calling（官方实现）

## 配置方法

### 1. 官方 Claude API

在 `.env` 文件中配置：

```bash
# 使用 Anthropic 服务
LLM_SERVICE=anthropic

# Claude API 配置
ANTHROPIC_API_KEY=sk-ant-xxxxx
ANTHROPIC_API_URL=https://api.anthropic.com
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
```

可选模型：
- `claude-opus-4-6`（最强推理，支持自适应思考）
- `claude-sonnet-4-5-20250929`（推荐，平衡性能）
- `claude-sonnet-4-6`（最新）
- `claude-3-5-haiku`（快速）

### 2. 智谱 GLM API

在 `.env` 文件中配置：

```bash
# 使用 Anthropic 服务（智谱 GLM 兼容）
LLM_SERVICE=anthropic

# 智谱 GLM API 配置
ANTHROPIC_API_KEY=your-zhipu-api-key
ANTHROPIC_API_URL=https://open.bigmodel.cn/api/anthropic
ANTHROPIC_MODEL=glm-4.7
```

可选模型：
- `glm-4.7`（最新）
- `glm-4-plus`
- `glm-4-flash`
- `glm-4-air`

### 3. 其他兼容服务

任何兼容 Anthropic Messages API 规范的服务都可以使用：

```bash
LLM_SERVICE=anthropic
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_API_URL=https://your-provider.com/anthropic-compatible
ANTHROPIC_MODEL=their-model-name
```

## 代码实现

### llm_services.py 工厂方法

```python
elif service_lower == "anthropic":
    from pipecat.services.anthropic.llm import AnthropicLLMService

    # 检查是否为自定义 API（如智谱 GLM）
    if base_url and base_url != "https://api.anthropic.com":
        # 自定义 API（如智谱 GLM）
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        return AnthropicLLMService(
            api_key=api_key,
            model=model or "claude-sonnet-4-5-20250929",
            client=client,
            **kwargs
        )
    else:
        # 官方 Anthropic API
        return AnthropicLLMService(
            api_key=api_key,
            model=model or "claude-sonnet-4-5-20250929",
            **kwargs
        )
```

### pipecat_main_v2.py 统一架构

```python
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

# 创建统一的 LLM Context
context = LLMContext(messages=messages, tools=tools)

# 创建 Aggregators（所有 LLM 服务通用）
aggregator_pair = LLMContextAggregatorPair(context)
user_aggregator = aggregator_pair.user()
assistant_aggregator = aggregator_pair.assistant()
```

## 优势

1. **统一架构**：所有 LLM 服务使用相同的 Context 和 Aggregator
2. **官方支持**：使用 Pipecat 官方实现，稳定可靠
3. **易于扩展**：添加新的 Anthropic 兼容服务非常简单
4. **Function Calling**：官方支持，无需额外配置

## 测试

运行测试脚本验证配置：

```bash
# 设置环境变量
export ANTHROPIC_API_KEY=your-key

# 运行测试
uv run python tests/test_anthropic_llm.py
```

## 参考文档

- [Pipecat 官方文档 - LLM Services](https://docs.pipecat.ai/guides/learn/llm)
- [Pipecat 官方文档 - Context Management](https://docs.pipecat.ai/guides/learn/context-management)
- [Anthropic Messages API 文档](https://docs.anthropic.com/en/api/messages)
- [智谱 GLM API 文档](https://open.bigmodel.cn/dev/api)
