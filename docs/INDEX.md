# 📚 文档索引

智能语音助手项目的完整文档索引。

---

## 🚀 新手必读

1. **[快速开始指南](../QUICKSTART.md)** ⭐
   - 5 分钟快速上手
   - 推荐配置方案
   - 常见问题解答

2. **[配置方案对比](CONFIG_COMPARISON.md)** ⭐
   - 4 种配置方案详细对比
   - 性能指标和费用估算
   - 帮助你选择最合适的配置

3. **[完整配置文档](../.env.example)**
   - 所有配置参数说明
   - 按使用场景组织
   - 注释清晰，易于理解

---

## 🔧 核心功能文档

### LLM 服务

- **[多 LLM 服务配置](../.env.example)** (LLM 部分)
  - Qwen / DeepSeek / OpenAI / Anthropic Claude
  - 工厂模式，一键切换

### TTS 语音合成

- **[WebSocket TTS 完整指南](QWEN_TTS_REALTIME.md)** 🆕
  - WebSocket 实时语音合成
  - 首包延迟 ~100-200ms
  - 事件驱动架构

- **[TTS 迁移指南](TTS_MIGRATION_GUIDE.md)**
  - 从 HTTP 流式切换到 WebSocket
  - 3 步快速迁移
  - 配置参数对照表

- **[TTS 配置对比](CONFIG_COMPARISON.md)** (TTS 部分)
  - Piper / DashScope Realtime / HTTP / Edge / Azure
  - 性能和费用对比

### Vision 视觉理解

- **[Vision 服务配置](../.env.example)** (Vision 部分)
  - Moondream 本地模型
  - Qwen-VL API
  - 多模型支持

### MCP 工具

- **[MCP 配置说明](../.env.example)** (MCP 部分)
  - Playwright 浏览器控制
  - Windows 系统操作
  - Filesystem 文件系统
  - GitHub 操作

---

## 📖 开发文档

### 架构设计

- **[Pipeline 架构](../README.md)** (架构部分)
  - 基于 Pipecat 官方框架
  - 模块化设计
  - 完全异步架构

### Agent Skills

- **[Agent Skills 系统](../README.md)** (特性部分)
  - Claude Code 设计
  - LLM 自主判断技能使用
  - 零停用词、零关键词匹配

### 音频处理

- **[音频降噪](../README.md)** (特性部分)
  - RNNoise 深度学习降噪
  - soxr 高质量重采样
  - VAD + Smart Turn Detection

---

## 🛠️ 示例代码

### 测试脚本

- `test_agent_skills.py` - Agent Skills 测试
- `test_vision_models.py` - Vision 模型测试
- `list_audio_devices.py` - 音频设备列表

### 示例项目

- `Example/pipecat/` - Pipecat 官方示例
- `Example/pipecat-client-web/` - Web 客户端
- `Example/voice-ui-kit/` - 语音 UI 组件

---

## 🔍 问题排查

### 常见问题

1. **[快速开始 - 常见问题](../QUICKSTART.md)** (常见问题部分)
   - 没有声音？
   - 唤醒词检测不到？
   - 如何切换服务？

2. **[TTS 迁移指南 - 常见问题](TTS_MIGRATION_GUIDE.md)** (常见问题部分)
   - WebSocket 连接失败？
   - 音质有变化吗？
   - 可以回退到 HTTP 吗？

---

## 📝 更新日志

- **[v2.9.0 - WebSocket 实时语音合成](../README.md)** (更新日志部分)
  - WebSocket Realtime TTS
  - 首包延迟优化
  - 配置和文档更新

---

## 🎯 按场景查找文档

### 场景 1：首次使用

1. 阅读 [快速开始指南](../QUICKSTART.md)
2. 参考 [配置方案对比](CONFIG_COMPARISON.md) 选择方案
3. 复制 [完整配置文档](../.env.example) 中的配置

### 场景 2：生产部署

1. 阅读 [配置方案对比](CONFIG_COMPARISON.md) (方案 2：生产环境)
2. 配置 WebSocket Realtime TTS
3. 参考 [TTS 完整指南](QWEN_TTS_REALTIME.md)

### 场景 3：从 HTTP 迁移到 WebSocket

1. 阅读 [TTS 迁移指南](TTS_MIGRATION_GUIDE.md)
2. 按照 3 步快速迁移
3. 运行测试验证

### 场景 4：性能优化

1. 阅读 [配置方案对比](CONFIG_COMPARISON.md) (性能优化部分)
2. 选择合适的 TTS 引擎
3. 调整 LLM 模型

### 场景 5：自定义开发

1. 阅读 [Pipeline 架构](../README.md)
2. 查看示例代码
3. 参考官方文档

---

## 🔗 外部资源

### 官方文档

- [Pipecat 官方文档](https://docs.pipecat.ai/)
- [DashScope 文档](https://help.aliyun.com/zh/dashscope/)
- [Qwen-TTS-Realtime 文档](https://help.aliyun.com/zh/model-studio/qwen-tts-realtime)
- [Anthropic Claude 文档](https://docs.anthropic.com/)

### 社区资源

- [GitHub Issues](https://github.com/yourusername/chinese-voice-assistant/issues)
- [GitHub Discussions](https://github.com/yourusername/chinese-voice-assistant/discussions)

---

## 📧 获取帮助

如果文档没有解决你的问题：

1. 查看 [常见问题](../QUICKSTART.md)
2. 搜索 [GitHub Issues](https://github.com/yourusername/chinese-voice-assistant/issues)
3. 提交新的 [Issue](https://github.com/yourusername/chinese-voice-assistant/issues/new)

---

**最后更新**: 2025-04-01
