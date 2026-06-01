# WebSocket Realtime TTS 配置指南

## ✅ 已切换到 WebSocket Realtime TTS

你的配置已经从 **HTTP 流式 TTS** 切换到 **WebSocket Realtime TTS**，这将大幅改善音频播放质量。

---

## 📊 配置对比

| 特性 | HTTP 流式 TTS | WebSocket Realtime TTS |
|------|--------------|----------------------|
| **延迟** | ~300ms | **~100-200ms** ⭐ |
| **连接方式** | 每次 HTTP 请求 | **持久 WebSocket 连接** |
| **音频流畅度** | 可能有电音/卡顿 | **抖动缓冲，更流畅** ⭐ |
| **首包延迟** | ~300ms | **~100-200ms** ⭐ |
| **适用场景** | 稳定可靠 | **实时对话** ⭐ |

---

## 🔧 当前配置

### `.env` 配置

```bash
# ✅ 已启用 WebSocket Realtime TTS
TTS_SERVICE=dashscope_realtime

# WebSocket Realtime TTS 配置
DASHSCOPE_REALTIME_MODEL=qwen3-tts-flash-realtime
DASHSCOPE_REALTIME_VOICE=Cherry
DASHSCOPE_REALTIME_MODE=server_commit
```

### 参数说明

- **model**: `qwen3-tts-flash-realtime`
  - WebSocket 实时模型，首包延迟 ~100-200ms

- **voice**: `Cherry`
  - 音色选择：Cherry | Ethan | Sunny | Dylan

- **mode**: `server_commit`
  - `server_commit`（自动提交）：推荐，延迟更低
  - `commit`（手动提交）：需要手动控制提交时机

---

## 🧪 测试音频播放质量

### 方法 1：运行独立测试脚本（推荐）

```bash
cd C:\Users\iEZELL\chinese-voice-assistant
uv run python test_websocket_tts.py
```

这个脚本会：
1. 建立 WebSocket 连接
2. 播放多个测试句子
3. 检查音频块之间的过渡是否流畅

**预期效果：**
- ✅ 无电音（咔哒声）
- ✅ 句子之间过渡平滑
- ✅ 无"一段一段"的感觉

### 方法 2：运行完整程序

```bash
uv run python main.py
```

启动后应该看到：
```
⏳ 尝试初始化 DashScope Realtime TTS（WebSocket）...
✓ DashScope Realtime TTS 配置成功
  注意：WebSocket TTS 将在首次使用时初始化连接
```

首次说话时会看到：
```
✓ Qwen-TTS-Realtime 已连接
  - 模型: qwen3-tts-flash-realtime
  - 音色: Cherry
  - 模式: server_commit
  - 采样率: 24000Hz
✓ WebSocket 连接已建立
✓ 首包延迟: 150ms
```

---

## 🔍 评估音频质量

播放测试音频后，请评估：

### ✅ 好的音频质量
- [ ] 无电音（咔哒声）
- [ ] 句子之间过渡平滑
- [ ] 无明显的音频块边界感
- [ ] 听起来像连续的语音

### ❌ 需要改进
- [ ] 仍有电音或爆音
- [ ] 听起来一段一段的
- [ ] 有明显的停顿或间隙

---

## 🛠️ 如果仍有问题

### 问题 1：仍有电音

**可能原因：**
- 网络不稳定导致音频块到达不均匀
- 音频缓冲区太小

**解决方案：**
1. 检查网络连接质量
2. 增加音频缓冲（修改 `tts_realtime.py`）：
   ```python
   # 增加缓冲区大小
   frames_per_buffer=2048  # 从 1024 增加到 2048
   ```

### 问题 2：仍有卡顿

**可能原因：**
- WebSocket 连接不稳定
- 音频播放队列不足

**解决方案：**
1. 检查 WebSocket 连接状态
2. 增加 Pipecat 的音频缓冲（在 `pipecat_main_v2.py` 中）：
   ```python
   # 增加 audio_out_buffer_size
   audio_out_buffer_size=20  # 增加缓冲
   ```

### 问题 3：想要更极致的流畅度

**终极解决方案：添加音频平滑处理器**

如果 WebSocket Realtime TTS 仍有轻微问题，可以添加我之前创建的 `AudioSmoother`：
- 淡入淡出（Fade In/Out）
- 交叉淡化（Cross-fade）
- 完全消除音频块边界

---

## 📈 性能监控

WebSocket Realtime TTS 会输出性能指标：

```
✓ 首包延迟: 150ms  # 从发送请求到收到第一个音频块的时间
✓ 音频生成完成    # 整个句子合成完成
✓ 响应完成: success  # 响应状态
```

**理想值：**
- 首包延迟：100-200ms
- 连接建立：< 500ms
- 音频流畅：无卡顿

---

## 🔙 切换回其他 TTS 引擎

如果 WebSocket Realtime TTS 不适合你，可以切换回其他引擎：

### 切换到 Piper TTS（本地，最快）

```bash
# 修改 .env
TTS_SERVICE=piper
```

### 切换到 HTTP 流式 TTS（稳定）

```bash
# 修改 .env
TTS_SERVICE=dashscope
DASHSCOPE_TTS_MODEL=qwen3-tts-flash
DASHSCOPE_TTS_VOICE=Cherry
```

### 切换到 Edge TTS（免费）

```bash
# 修改 .env
TTS_SERVICE=edge
EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
```

---

## 📚 相关文档

- [WebSocket TTS 详细说明](QWEN_TTS_REALTIME.md)
- [修复导入错误](FIX_WEBSOCKET_TTS_IMPORTS.md)
- [快速开始指南](../QUICKSTART.md)
- [配置方案对比](CONFIG_COMPARISON.md)

---

## ✅ 下一步

1. **运行测试脚本**
   ```bash
   uv run python test_websocket_tts.py
   ```

2. **评估音频质量**
   - 检查是否还有电音
   - 检查过渡是否流畅

3. **如仍有问题**
   - 告诉我具体的问题
   - 我会添加音频平滑处理器

---

**祝使用愉快！** 🎉
