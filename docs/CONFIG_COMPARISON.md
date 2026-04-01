# 配置方案对比

## 🎯 选择合适的配置方案

根据你的需求选择最合适的配置：

---

## 方案 1：开发环境（免费）💰

### 配置

```bash
# .env
LLM_SERVICE=qwen
QWEN_API_KEY=your-qwen-api-key-here
QWEN_MODEL=qwen-plus

TTS_SERVICE=piper

VISION_SERVICE=moondream
MOONDREAM_USE_CPU=false
```

### 性能指标

| 指标 | 数值 |
|------|------|
| **LLM 延迟** | ~800ms |
| **TTS 延迟** | <100ms ⚡ |
| **首包延迟** | ~900ms |
| **音质** | ⭐⭐⭐ |
| **费用** | 免费（Qwen 免费额度）|

### 优点

✅ 完全免费（Qwen 有免费额度）
✅ 延迟最低（Piper 本地 TTS <100ms）
✅ 离线运行（Piper TTS + Moondream Vision）
✅ 快速迭代

### 缺点

⚠️ 音质一般（Piper TTS）
⚠️ 需要下载模型（~500MB）

### 适用场景

- 本地开发和测试
- 离线环境
- 预算有限的项目

---

## 方案 2：生产环境（推荐）⭐

### 配置

```bash
# .env
LLM_SERVICE=qwen
QWEN_API_KEY=your-qwen-api-key-here
QWEN_MODEL=qwen-plus

TTS_SERVICE=dashscope_realtime
DASHSCOPE_REALTIME_MODEL=qwen3-tts-flash-realtime
DASHSCOPE_REALTIME_VOICE=Cherry
DASHSCOPE_REALTIME_MODE=server_commit

VISION_SERVICE=qwen-vl-plus
```

### 性能指标

| 指标 | 数值 |
|------|------|
| **LLM 延迟** | ~800ms |
| **TTS 延迟** | ~100-200ms ⚡ |
| **首包延迟** | ~900-1000ms |
| **音质** | ⭐⭐⭐⭐⭐ |
| **费用** | 低（按量计费）|

### 优点

✅ 低延迟（WebSocket ~100-200ms）
✅ 高音质（Qwen TTS）
✅ 稳定可靠（阿里云服务）
✅ 实时体验好
✅ 费用低（按量计费）

### 缺点

💰 需要 API 费用（但很低）
🌐 需要网络连接

### 适用场景

- 生产环境部署
- 实时对话场景
- 需要高质量语音
- 推荐大多数用户使用 ⭐

---

## 方案 3：最低延迟 ⚡

### 配置

```bash
# .env
LLM_SERVICE=qwen
QWEN_API_KEY=your-qwen-api-key-here
QWEN_MODEL=qwen-turbo

TTS_SERVICE=dashscope_realtime
DASHSCOPE_REALTIME_MODEL=qwen3-tts-flash-realtime
DASHSCOPE_REALTIME_VOICE=Cherry
DASHSCOPE_REALTIME_MODE=commit

VISION_SERVICE=qwen-vl-plus
```

### 性能指标

| 指标 | 数值 |
|------|------|
| **LLM 延迟** | ~600ms ⚡ |
| **TTS 延迟** | ~100ms ⚡ |
| **首包延迟** | ~700ms ⚡ |
| **音质** | ⭐⭐⭐⭐⭐ |
| **费用** | 低（按量计费）|

### 优点

⚡ 最低延迟（~700ms 首包）
✅ 高音质
✅ 精细控制

### 缺点

⚠️ 需要手动控制合成时机
⚠️ 实现复杂度较高
💰 需要 API 费用

### 适用场景

- 对延迟极度敏感的场景
- 需要精细控制 TTS 合成时机
- 高级用户

---

## 方案 4：高质量（付费）💎

### 配置

```bash
# .env
LLM_SERVICE=anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

TTS_SERVICE=azure
AZURE_TTS_API_KEY=your-azure-api-key-here
AZURE_TTS_REGION=eastasia
AZURE_TTS_VOICE=zh-CN-XiaoxiaoNeural

VISION_SERVICE=qwen-vl-max
```

### 性能指标

| 指标 | 数值 |
|------|------|
| **LLM 延迟** | ~1000ms |
| **TTS 延迟** | ~400ms |
| **首包延迟** | ~1400ms |
| **音质** | ⭐⭐⭐⭐⭐ |
| **费用** | 高 |

### 优点

✅ 最高质量（Claude + Azure TTS）
✅ 最强的理解能力
✅ 最自然的语音

### 缺点

💰 费用较高
⏱️ 延迟较高

### 适用场景

- 对质量要求极高的场景
- 预算充足的项目
- 演示和展示

---

## 快速对比表

| 方案 | 首包延迟 | 音质 | 费用 | 推荐度 |
|------|---------|------|------|--------|
| **方案 1：开发环境** | ~900ms | ⭐⭐⭐ | 免费 | ⭐⭐⭐ |
| **方案 2：生产环境** | ~900-1000ms | ⭐⭐⭐⭐⭐ | 低 | ⭐⭐⭐⭐⭐ |
| **方案 3：最低延迟** | ~700ms | ⭐⭐⭐⭐⭐ | 低 | ⭐⭐⭐⭐ |
| **方案 4：高质量** | ~1400ms | ⭐⭐⭐⭐⭐ | 高 | ⭐⭐⭐ |

---

## 延迟组成分析

```
总延迟 = LLM 延迟 + TTS 延迟 + 网络延迟

方案 1（开发环境）：
  900ms ≈ 800ms (LLM) + <100ms (Piper TTS) + 0ms (本地)

方案 2（生产环境）：
  950ms ≈ 800ms (LLM) + 150ms (WebSocket TTS 平均)

方案 3（最低延迟）：
  700ms ≈ 600ms (Qwen Turbo) + 100ms (WebSocket TTS 优化)
```

---

## 费用估算

### Qwen（阿里云 DashScope）

| 模型 | 价格 | 免费额度 |
|------|------|---------|
| qwen-turbo | ¥0.0008/千tokens | 100万tokens/月 |
| qwen-plus | ¥0.004/千tokens | 100万tokens/月 |
| qwen-max | ¥0.04/千tokens | 无 |

### TTS（DashScope Realtime）

| 模型 | 价格 |
|------|------|
| qwen3-tts-flash-realtime | ¥0.016/千字符 |
| qwen3-tts-instruct-flash-realtime | ¥0.024/千字符 |

**月费用估算**（中等使用）：
- LLM: ~¥10-50/月
- TTS: ~¥5-20/月
- **总计**: ~¥15-70/月

---

## 推荐选择

### 🎯 首次使用？

选择 **方案 1（开发环境）**：
- 免费试用
- 快速上手
- 了解功能

### 🚀 生产部署？

选择 **方案 2（生产环境）** ⭐：
- 最佳性价比
- 低延迟 + 高音质
- 稳定可靠

### ⚡ 追求极限性能？

选择 **方案 3（最低延迟）**：
- 最低延迟
- 需要技术能力
- 高级用户

### 💎 预算充足？

选择 **方案 4（高质量）**：
- 最高质量
- 最强能力
- 不差钱

---

## 切换配置

### 从方案 1 切换到方案 2

```bash
# 修改 .env
# 从这个
TTS_SERVICE=piper

# 改为这个
TTS_SERVICE=dashscope_realtime
DASHSCOPE_REALTIME_MODEL=qwen3-tts-flash-realtime
DASHSCOPE_REALTIME_VOICE=Cherry
DASHSCOPE_REALTIME_MODE=server_commit
```

### 从方案 2 切换到方案 3

```bash
# 修改 .env
# 从这个
QWEN_MODEL=qwen-plus
DASHSCOPE_REALTIME_MODE=server_commit

# 改为这个
QWEN_MODEL=qwen-turbo
DASHSCOPE_REALTIME_MODE=commit
```

---

## 总结

**大多数用户推荐：方案 2（生产环境）⭐**

- ✅ 最佳平衡
- ✅ 低延迟
- ✅ 高音质
- ✅ 费用低
- ✅ 稳定可靠

**快速开始**：
1. 复制 `.env.example` 为 `.env`
2. 配置 Qwen API Key
3. 设置 `TTS_SERVICE=dashscope_realtime`
4. 运行 `uv run python main.py`

**详细文档**：
- [完整配置指南](.env.example)
- [快速开始](QUICKSTART.md)
- [WebSocket TTS 详细说明](docs/QWEN_TTS_REALTIME.md)
