# Anthropic Messages API LLM Service 使用指南

本文档说明如何使用 Anthropic Messages API 兼容层作为 LLM 服务。

## 📋 目录

- [特性](#特性)
- [兼容服务](#兼容服务)
- [安装](#安装)
- [配置](#配置)
- [使用](#使用)
- [可用模型](#可用模型)
- [思考模式](#思考模式)
- [API 格式差异](#api-格式差异)
- [本地部署](#本地部署)
- [注意事项](#注意事项)

---

## ✨ 特性

- ✅ **Anthropic Messages API 标准兼容层** - 不依赖 OpenAI 兼容层
- ✅ **支持官方 Claude** - Anthropic API（api.anthropic.com）
- ✅ **支持本地模型** - Ollama、vLLM、LM Studio 等
- ✅ **支持其他云服务商** - 提供兼容 API 的供应商
- ✅ **流式响应** - 低延迟实时输出
- ✅ **自适应思考** - Claude 可以自主决定何时进行深度推理
- ⚠️ **Function Calling** - 暂未实现（保留）

---

## 🔌 兼容服务

### 1. Anthropic Claude（官方）

| 服务 | API 地址 | 说明 |
|------|----------|------|
| Claude Opus 4.6 | `https://api.anthropic.com` | 最强推理 |
| Claude Sonnet 4.6 | `https://api.anthropic.com` | 平衡性能 |
| Claude Haiku 3.5 | `https://api.anthropic.com` | 快速响应 |

### 2. 本地部署（兼容 Anthropic API）

| 服务 | API 地址 | 说明 |
|------|----------|------|
| Ollama | `http://localhost:11434` | 开源模型运行器 |
| vLLM | `http://localhost:8000` | 高性能推理服务 |
| LM Studio | `http://localhost:1234` | 本地 GUI 工具 |

### 3. 其他云服务商

任何提供 Anthropic Messages API 兼容接口的云服务商。

---

## 📦 安装

### 1. 安装依赖

```bash
# 使用 uv（推荐）
uv pip install anthropic

# 或使用 pip
pip install anthropic
```

### 2. 配置 API Key

编辑 `.env` 文件：

```bash
# 切换到 Anthropic
LLM_SERVICE=anthropic

# Anthropic API 配置
ANTHROPIC_API_KEY=sk-ant-xxxxx
ANTHROPIC_API_URL=https://api.anthropic.com
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# 思考模式（可选）
ANTHROPIC_ENABLE_THINKING=false
ANTHROPIC_THINKING_EFFORT=medium
```

获取 API Key：https://console.anthropic.com/

---

## ⚙️ 配置

### 环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_SERVICE` | 选择 LLM 服务 | `qwen` |
| `ANTHROPIC_API_KEY` | Anthropic API Key | 必填 |
| `ANTHROPIC_API_URL` | API 地址 | `https://api.anthropic.com` |
| `ANTHROPIC_MODEL` | 模型名称 | `claude-sonnet-4-5-20250929` |
| `ANTHROPIC_ENABLE_THINKING` | 是否启用思考模式 | `false` |
| `ANTHROPIC_THINKING_EFFORT` | 思考强度 | `medium` |

### 配置示例

```bash
# .env 文件

# 使用 Claude Sonnet 4.5（推荐）
LLM_SERVICE=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxx
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# 使用 Claude Opus 4.6（最强推理）
# LLM_SERVICE=anthropic
# ANTHROPIC_API_KEY=sk-ant-xxxxx
# ANTHROPIC_MODEL=claude-opus-4-6
# ANTHROPIC_ENABLE_THINKING=true
# ANTHROPIC_THINKING_EFFORT=high
```

---

## 🚀 使用

### 1. 代码中使用

```python
from voice_assistant.llm_services import create_llm_service

# 创建 Anthropic LLM Service
llm = create_llm_service(
    service="anthropic",
    api_key="sk-ant-xxxxx",
    model="claude-sonnet-4-5-20250929"
)

# 使用与 OpenAI 服务相同的接口
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

async for text_chunk in llm.stream_chat_completion(messages):
    print(text_chunk, end="")
```

### 2. 启动语音助手

```bash
# 方式 1：使用主入口
python main.py

# 方式 2：直接运行 Pipecat 主程序
uv run python -m src.voice_assistant.pipecat_main_v2
```

### 3. 测试

```bash
# 运行测试
pytest tests/test_anthropic_llm.py -v

# 或直接运行
python tests/test_anthropic_llm.py
```

---

## 🤖 可用模型

### Claude Opus 4.6（最强推理）

```bash
ANTHROPIC_MODEL=claude-opus-4-6
ANTHROPIC_ENABLE_THINKING=true
ANTHROPIC_THINKING_EFFORT=high
```

- ✅ 最强推理能力
- ✅ 自适应思考模式
- ✅ 适合复杂任务
- ⚠️ 成本较高

### Claude Sonnet 4.5/4.6（推荐）

```bash
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
# 或
ANTHROPIC_MODEL=claude-sonnet-4-6
```

- ✅ 平衡性能和成本
- ✅ 支持思考模式
- ✅ 适合大多数场景

### Claude 3.5 Haiku（快速）

```bash
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
```

- ✅ 响应速度快
- ✅ 成本低
- ✅ 适合简单任务

---

## 💻 本地部署

### 使用 Ollama

Ollama 支持 Anthropic Messages API 格式。

#### 1. 安装 Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# 下载安装包：https://ollama.com/download
```

#### 2. 运行兼容模型

```bash
# 拉取 Claude 兼容模型
ollama pull qwen2.5-coder:latest

# 启动 Ollama 服务（默认端口 11434）
ollama serve
```

#### 3. 配置语音助手

```bash
# .env 文件
LLM_SERVICE=anthropic
ANTHROPIC_API_URL=http://localhost:11434
ANTHROPIC_API_KEY=ollama  # Ollama 不需要真实 key
ANTHROPIC_MODEL=qwen2.5-coder:latest
```

### 使用 vLLM

vLLM 是高性能的 LLM 推理服务，支持 Anthropic Messages API。

#### 1. 安装 vLLM

```bash
pip install vllm
```

#### 2. 启动 vLLM 服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8000 \
    --chat-template anthropic
```

#### 3. 配置语音助手

```bash
# .env 文件
LLM_SERVICE=anthropic
ANTHROPIC_API_URL=http://localhost:8000
ANTHROPIC_API_KEY=empty
ANTHROPIC_MODEL=qwen2.5-coder
```

### 使用 LM Studio

LM Studio 提供友好的 GUI，支持 Anthropic API。

#### 1. 下载并安装 LM Studio

https://lmstudio.ai/

#### 2. 加载兼容模型

- 在 LM Studio 中搜索兼容 Anthropic 的模型
- 或加载本地 GGUF 模型
- 启用 Anthropic API 兼容模式

#### 3. 配置语音助手

```bash
# .env 文件
LLM_SERVICE=anthropic
ANTHROPIC_API_URL=http://localhost:1234/v1
ANTHROPIC_API_KEY=lm-studio
ANTHROPIC_MODEL=loaded-model-name
```

### 本地部署注意事项

⚠️ **重要**：
1. 确保本地服务完全兼容 Anthropic Messages API 格式
2. 检查模型是否支持 `system` 参数和 `tools`
3. 本地模型可能需要不同的 `chat_template`
4. 性能取决于硬件（GPU 推荐）

---

## 🧠 思考模式

Claude Opus 4.6 和 Sonnet 4.6 支持自适应思考模式。

### 启用思考模式

```bash
ANTHROPIC_ENABLE_THINKING=true
ANTHROPIC_THINKING_EFFORT=medium  # high | medium | low
```

### 思考强度说明

| 强度 | 说明 | 适用场景 |
|------|------|----------|
| `high` | 深度思考 | 复杂推理、多步骤任务 |
| `medium` | 平衡思考 | 大多数场景（推荐） |
| `low` | 快速思考 | 简单任务、快速响应 |

### 注意事项

- 思考模式会增加响应延迟
- 思考模式会增加 token 消耗
- 建议仅在复杂任务时启用

---

## 📊 API 格式差异

### OpenAI vs Anthropic

| 特性 | OpenAI API | Anthropic Messages API |
|------|------------|----------------------|
| **端点** | `/v1/chat/completions` | `/v1/messages` |
| **系统提示** | `messages[0]` | `system` 参数 |
| **最大 tokens** | `max_tokens` | `max_tokens` |
| **工具定义** | `tools[].function` | `tools[]` |
| **工具 schema** | `parameters` | `input_schema` |
| **响应格式** | `choices[0].message` | `content[]` 数组 |
| **流式事件** | `delta.content` | `delta.text` |

### 示例对比

**OpenAI 格式：**
```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
]
```

**Anthropic 格式：**
```python
system = "You are helpful."
messages = [
    {"role": "user", "content": "Hello!"}
]
```

---

## ⚠️ 注意事项

### 1. Function Calling

⚠️ **当前未实现** - Anthropic LLM Service 暂不支持 Function Calling。

如果需要使用工具调用，请使用：
- Qwen（推荐，中文优化）
- DeepSeek（强推理）
- OpenAI（官方）

### 2. 成本

Anthropic API 价格相对较高：

| 模型 | 输入（$/M tokens） | 输出（$/M tokens） |
|------|-------------------|-------------------|
| Opus 4.6 | $15.00 | $75.00 |
| Sonnet 4.5 | $3.00 | $15.00 |
| Haiku 3.5 | $0.80 | $1.00 |

建议：
- 简单任务使用 Haiku
- 大多数任务使用 Sonnet
- 复杂任务使用 Opus

### 3. 限流

Anthropic API 有速率限制：

- 免费账号：5 RPM（requests per minute）
- 付费账号：根据套餐不同

建议：
- 实现请求队列
- 添加重试逻辑
- 监控使用量

### 4. 延迟

- 流式响应延迟：~200-500ms
- 思考模式延迟：+2-5 秒
- 首次请求延迟：+500ms（冷启动）

---

## 🔍 故障排查

### 问题 1：API Key 无效

```
anthropic.BadRequestError: Invalid API key
```

**解决方法：**
1. 检查 API Key 是否正确
2. 确认 API Key 已激活
3. 检查环境变量是否正确设置

### 问题 2：模型名称错误

```
anthropic.BadRequestError: Invalid model name
```

**解决方法：**
1. 检查模型名称拼写
2. 确认模型可用
3. 使用推荐的模型名称

### 问题 3：超时

```
asyncio.TimeoutError
```

**解决方法：**
1. 检查网络连接
2. 增加超时时间
3. 联系 API 提供商

---

## 📚 参考文档

- [Anthropic API 文档](https://docs.anthropic.com/en/api/messages)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Claude 模型介绍](https://docs.anthropic.com/en/docs/about-claude/models)

---

## 📝 更新日志

### v2.8.0（2025-01-XX）

- ✅ 添加 Anthropic LLM Service
- ✅ 支持原生 Messages API
- ✅ 支持流式响应
- ✅ 支持思考模式
- ⚠️ Function Calling 暂未实现

---

如有问题或建议，请提交 Issue 或 Pull Request。
