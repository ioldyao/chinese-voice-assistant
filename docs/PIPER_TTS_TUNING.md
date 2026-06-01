# Piper TTS 参数调优指南

## 🎛️ 可调参数

Piper TTS 支持通过 `SynthesisConfig` 调节音色和语速，以下是所有可用参数：

### 1️⃣ **length_scale** - 语速

```bash
PIPER_LENGTH_SCALE=1.0
```

**范围**: `0.5 - 2.0`

| 值 | 效果 | 适用场景 |
|----|------|---------|
| `0.5` | 2倍速（非常快） | 快速阅读、信息密集 |
| `0.8` | 1.25倍速（快） | 活泼对话、快速响应 |
| **`1.0`** | **正常速度** | **默认，推荐** ⭐ |
| `1.2` | 0.83倍速（慢） | 沉稳讲解、重要信息 |
| `2.0` | 0.5倍速（非常慢） | 慢速教学、逐字说明 |

**示例**：
```bash
# 快速模式（适合实时对话）
PIPER_LENGTH_SCALE=0.8

# 慢速模式（适合教学、讲解）
PIPER_LENGTH_SCALE=1.2
```

---

### 2️⃣ **volume** - 音量

```bash
PIPER_VOLUME=1.0
```

**范围**: `0.1 - 2.0`

| 值 | 效果 |
|----|------|
| `0.1` | 非常小声 |
| `0.5` | 一半音量 |
| **`1.0`** | **正常音量** ⭐ |
| `1.5` | 1.5倍音量 |
| `2.0` | 2倍音量（最大） |

**注意**: 建议保持 `1.0`，通过系统音量调节。

---

### 3️⃣ **noise_scale** - 音频变化（表现力）

```bash
PIPER_NOISE_SCALE=0.667
```

**范围**: `0.1 - 1.0`

| 值 | 效果 | 适用场景 |
|----|------|---------|
| `0.3` | 非常平淡（机械化） | 不推荐 |
| `0.5` | 平淡（缺乏变化） | 不推荐 |
| **`0.667`** | **自然** | **默认，推荐** ⭐ |
| `0.8` | 有表现力（情感丰富） | 活泼对话 |
| `1.0` | 非常有表现力（夸张） | 故事讲述 |

**说明**:
- 控制音频合成的随机性
- 值越高，语音越有抑扬顿挫
- 过高可能导致语音不稳定

**示例**：
```bash
# 自然清晰（推荐）
PIPER_NOISE_SCALE=0.667

# 活泼有表现力
PIPER_NOISE_SCALE=0.8
```

---

### 4️⃣ **noise_w_scale** - 说话变化（抑扬顿挫）

```bash
PIPER_NOISE_W_SCALE=0.8
```

**范围**: `0.1 - 1.0`

| 值 | 效果 | 适用场景 |
|----|------|---------|
| `0.3` | 单调（无变化） | 不推荐 |
| `0.5` | 较单调 | 不推荐 |
| **`0.8`** | **自然** | **默认，推荐** ⭐ |
| `0.9` | 有抑扬顿挫 | 活泼对话 |
| `1.0` | 非常抑扬顿挫 | 故事讲述 |

**说明**:
- 控制语音的音高变化
- 值越高，语音越有抑扬顿挫
- 与 `noise_scale` 配合使用

---

### 5️⃣ **normalize_audio** - 音频标准化

```bash
PIPER_NORMALIZE_AUDIO=true
```

**值**: `true` | `false`

| 值 | 效果 |
|----|------|
| **`true`** | **标准化音频**（推荐）⭐ |
| `false` | 使用原始音频 |

**说明**:
- 标准化可以让音量更均匀
- 建议保持 `true`

---

## 📋 推荐配置方案

### 方案 1：自然清晰（默认）⭐

```bash
PIPER_LENGTH_SCALE=1.0
PIPER_VOLUME=1.0
PIPER_NOISE_SCALE=0.667
PIPER_NOISE_W_SCALE=0.8
PIPER_NORMALIZE_AUDIO=true
```

**特点**:
- ✅ 自然清晰
- ✅ 适合大多数场景
- ✅ 不容易疲劳

---

### 方案 2：快速活泼（实时对话）

```bash
PIPER_LENGTH_SCALE=0.8
PIPER_VOLUME=1.0
PIPER_NOISE_SCALE=0.8
PIPER_NOISE_W_SCALE=0.9
PIPER_NORMALIZE_AUDIO=true
```

**特点**:
- ⚡ 语速快（1.25倍速）
- 🎭 表现力丰富
- 🗣️ 适合活泼对话

---

### 方案 3：慢速沉稳（教学、讲解）

```bash
PIPER_LENGTH_SCALE=1.2
PIPER_VOLUME=1.0
PIPER_NOISE_SCALE=0.6
PIPER_NOISE_W_SCALE=0.7
PIPER_NORMALIZE_AUDIO=true
```

**特点**:
- 🐢 语速慢（0.83倍速）
- 📚 沉稳清晰
- 🎓 适合教学、重要信息

---

### 方案 4：快速高效（信息密集）

```bash
PIPER_LENGTH_SCALE=0.7
PIPER_VOLUME=1.0
PIPER_NOISE_SCALE=0.667
PIPER_NOISE_W_SCALE=0.8
PIPER_NORMALIZE_AUDIO=true
```

**特点**:
- ⚡ 语速非常快（1.43倍速）
- 💾 节省时间
- 📰 适合新闻、信息播报

---

### 方案 5：故事讲述（情感丰富）

```bash
PIPER_LENGTH_SCALE=0.9
PIPER_VOLUME=1.1
PIPER_NOISE_SCALE=0.9
PIPER_NOISE_W_SCALE=1.0
PIPER_NORMALIZE_AUDIO=true
```

**特点**:
- 🎭 情感非常丰富
- 📖 适合讲故事
- 🎬 表现力强

---

## 🔧 如何修改配置

### 方法 1：修改 .env 文件（推荐）

```bash
# 编辑 .env 文件
nano .env

# 添加或修改以下配置
PIPER_LENGTH_SCALE=0.8
PIPER_NOISE_SCALE=0.8
PIPER_NOISE_W_SCALE=0.9

# 保存后重启程序
uv run python main.py
```

---

### 方法 2：设置环境变量（临时）

```bash
# Linux/macOS
export PIPER_LENGTH_SCALE=0.8
export PIPER_NOISE_SCALE=0.8
uv run python main.py

# Windows (PowerShell)
$env:PIPER_LENGTH_SCALE=0.8
$env:PIPER_NOISE_SCALE=0.8
uv run python main.py
```

---

## 🧪 测试建议

### 测试文本

```python
test_texts = [
    "你好，这是一个测试。",
    "我们来检查一下语音效果。",
    "请注意听语速和音色的变化。",
    "这段话可以帮助你评估音频质量。"
]
```

### 测试步骤

1. **启动程序**
   ```bash
   uv run python main.py
   ```

2. **观察输出**
   ```
   ✓ 使用 Piper TTS（本地，超快）- 模型: zh_CN-huayan-medium.onnx
     - 语速: 0.8x (<1 = 快, >1 = 慢)
     - 音量: 1.0x
     - 音频变化: 0.8 (0.667 = 自然)
     - 说话变化: 0.9 (0.8 = 自然)
   ```

3. **评估效果**
   - ✅ 语速是否合适？
   - ✅ 音色是否自然？
   - ✅ 是否有抑扬顿挫？
   - ✅ 是否容易听清？

---

## 🔍 参数调优技巧

### 1. 调节语速

**目标**: 找到适合你的语速

**方法**:
1. 从 `1.0`（正常）开始
2. 觉得慢 → 减小到 `0.8`
3. 觉得快 → 增大到 `1.2`
4. 微调：`0.1` 为单位

---

### 2. 调节表现力

**目标**: 在自然和表现力之间找到平衡

**方法**:
1. 先设置 `noise_scale` 和 `noise_w_scale` 为 `0.667`（自然）
2. 觉得平淡 → 同时增加 `0.1`
3. 觉得夸张 → 同时减少 `0.1`
4. 保持两个参数相近

---

### 3. 配合调节

**推荐组合**:
- **快速 + 表现力**: `length_scale=0.8`, `noise_scale=0.8`
- **慢速 + 自然**: `length_scale=1.2`, `noise_scale=0.6`
- **正常 + 清晰**: `length_scale=1.0`, `noise_scale=0.667`

---

## ⚠️ 常见问题

### Q1: 语速太快听不清？

**解决方案**：
```bash
PIPER_LENGTH_SCALE=1.2  # 增大语速参数
```

---

### Q2: 语音太平淡？

**解决方案**：
```bash
PIPER_NOISE_SCALE=0.8
PIPER_NOISE_W_SCALE=0.9
```

---

### Q3: 语音太夸张不稳定？

**解决方案**：
```bash
PIPER_NOISE_SCALE=0.667
PIPER_NOISE_W_SCALE=0.7
```

---

### Q4: 音量太小？

**解决方案**：
```bash
# 方法 1：增加 Piper 音量
PIPER_VOLUME=1.2

# 方法 2：调节系统音量（推荐）
```

---

## 📊 性能对比

| 配置 | 语速 | 表现力 | 适用场景 |
|------|------|--------|---------|
| **默认** | 正常 | 自然 | 通用 ⭐ |
| **快速** | 1.25x | 丰富 | 实时对话 |
| **慢速** | 0.83x | 平稳 | 教学 |
| **故事** | 1.1x | 夸张 | 讲故事 |

---

## 🎯 总结

### 推荐起步配置（默认）

```bash
PIPER_LENGTH_SCALE=1.0
PIPER_VOLUME=1.0
PIPER_NOISE_SCALE=0.667
PIPER_NOISE_W_SCALE=0.8
PIPER_NORMALIZE_AUDIO=true
```

### 进阶调优

1. **先调语速** (`length_scale`)
2. **再调表现力** (`noise_scale`, `noise_w_scale`)
3. **配合调节**
4. **多次测试**

---

**祝调优愉快！** 🎉

如有问题，请参考：
- [Piper TTS 官方文档](https://github.com/rhasspy/piper)
- [快速开始指南](../QUICKSTART.md)
- [配置方案对比](CONFIG_COMPARISON.md)
