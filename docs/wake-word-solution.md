# 唤醒词检测解决方案

## 问题

当前自定义 KWS Processor 架构存在问题：
- VAD 和 KWS 同时处理音频，互相干扰
- InterruptionFrame 来源不明确
- 日志混乱，难以调试

## 官方推荐方案：WakeCheckFilter

### 优点
- ✅ 官方支持，稳定可靠
- ✅ 简单清晰，易于维护
- ✅ 不需要自定义 Frame 处理
- ✅ 自动处理唤醒词检测

### 实现示例

```python
from pipecat.processors.filters import WakeCheckFilter

# 创建 WakeCheckFilter
wake_filter = WakeCheckFilter(
    wake_phrases=["小智", "你好助手", "智能助手"],
    keepalive_timeout=5.0  # 检测到唤醒词后，5秒内保持唤醒状态
)

# 添加到 Pipeline
pipeline = Pipeline([
    transport.input(),
    asr_service,           # STT 服务
    wake_filter,           # ← 唤醒词检测
    user_aggregator,
    llm,
    tts,
    transport.output()
])
```

### 工作流程

1. 用户说话 → STT 识别
2. `WakeCheckFilter` 检查识别结果
3. 如果包含唤醒词 → 传递给 LLM
4. 如果不包含 → 拦截（不传给 LLM）
5. `keepalive_timeout` 期间，所有识别结果都会传递

## 当前自定义方案的问题

### 架构
```
transport.input() → kws_proc → asr_proc → ...
```

### 问题
1. **重复处理**：VAD 和 KWS 都在分析音频
2. **Frame 冲突**：InterruptionFrame 来自多个源头
3. **状态同步**：KWS 和 ASR 的 `is_awake` 状态需要手动同步
4. **复杂度高**：需要处理多种 Frame 类型

### 日志分析
```
🔔 KWS 检测到: '小智'        ← KWS 检测到
✅ 确认唤醒词: 小智
🔔 收到唤醒信号，立即激活录音...  ← ASR 收到 InterruptionFrame
🔔 收到唤醒信号，立即激活录音...  ← 又一个 InterruptionFrame（来自哪里？）
```

第二个"收到唤醒信号"说明有其他地方也在发送 `InterruptionFrame`。

## 建议

**推荐使用官方 WakeCheckFilter**，原因：
1. 简单：不需要自定义 Processor
2. 稳定：官方维护，经过充分测试
3. 清晰：基于文本，逻辑直观
4. 可维护：减少自定义代码

如果必须使用自定义 KWS（如需要更低延迟），则需要：
1. 确保 VAD 不会发送干扰 Frame
2. 统一 Frame 类型管理
3. 清晰的状态机设计
