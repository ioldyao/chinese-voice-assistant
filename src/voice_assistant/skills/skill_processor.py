"""
Agent Skills - Pipecat 集成处理器

将 Agent Skills 集成到 Pipecat Pipeline 中。

功能：
- 作为 Pipecat FrameProcessor 工作
- 动态注册技能为 LLM 函数
- 自动注入技能系统指令
- 处理技能调用和结果返回

架构：
```
transport.input() → KWS → ASR → user_agg → SkillProcessor → LLM → TTS → transport.output()
                                                       ↓
                                                 SkillManager
                                                       ↓
                                                 MCP Tools
```

使用示例：
```python
from src.voice_assistant.skills.skill_processor import SkillProcessor

# 创建 SkillManager
manager = SkillManager(Path("skills"))
await manager.discover_all()

# 创建 SkillProcessor
processor = SkillProcessor(manager, llm_service, mcp_manager)

# 添加到 Pipeline
pipeline = Pipeline([
    transport.input(),
    kws_proc,
    asr_proc,
    user_aggregator,
    processor,  # ← SkillProcessor
    llm,
    tts_proc,
    assistant_aggregator,
    transport.output(),
])
```
"""
from typing import Any

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame
from pipecat.services.llm_service import FunctionCallParams

from .skill_manager import SkillManager


class SkillProcessor(FrameProcessor):
    """
    技能处理器 - Pipecat 集成

    将 Agent Skills 集成到 Pipecat Pipeline 中。

    功能：
    1. 自动发现并激活技能
    2. 动态注册技能为 LLM 函数
    3. 注入技能系统指令
    4. 处理技能调用

    Attributes:
        skill_manager: 技能管理器
        llm_service: LLM 服务
        mcp_manager: MCP 工具管理器
        auto_activate: 是否自动激活所有技能

    使用示例：
    ```python
    # 创建管理器
    manager = SkillManager(Path("skills"))
    await manager.discover_all()

    # 创建处理器
    processor = SkillProcessor(
        manager=manager,
        llm_service=llm,
        mcp_manager=mcp,
        auto_activate=True  # 自动激活所有技能
    )

    # 添加到 Pipeline
    pipeline = Pipeline([..., processor, llm, ...])
    ```
    """

    def __init__(
        self,
        skill_manager: SkillManager,
        llm_service: Any = None,
        mcp_manager: Any = None,
        auto_activate: bool = True
    ):
        """
        初始化技能处理器

        Args:
            skill_manager: 技能管理器
            llm_service: LLM 服务（用于注册函数）
            mcp_manager: MCP 工具管理器（用于工具调用）
            auto_activate: 是否自动激活所有技能
        """
        super().__init__()

        self.skill_manager = skill_manager
        self.llm_service = llm_service
        self.mcp_manager = mcp_manager
        self.auto_activate = auto_activate

    async def start(self):
        """
        启动处理器

        发现并激活技能。
        """
        print("🔧 SkillProcessor starting...")

        # 1. 发现所有技能
        await self.skill_manager.discover_all()

        # 2. 设置 LLM 服务和 task
        if self.llm_service:
            self.skill_manager.set_llm_service(self.llm_service)

        # 3. 激活技能
        if self.auto_activate:
            await self.skill_manager.activate_all()

        print("✓ SkillProcessor started")

    async def stop(self):
        """
        停止处理器

        停用所有技能。
        """
        print("🔻 SkillProcessor stopping...")
        await self.skill_manager.deactivate_all()
        print("✓ SkillProcessor stopped")

    async def process_frame(self, frame: Frame, direction: int):
        """
        处理 Frame

        转发所有 frames，不做修改。

        重要：必须调用 super().process_frame() 来正确处理 StartFrame、
        EndFrame、StartInterruptionFrame 等特殊帧。
        """
        # ✅ 关键：调用父类方法处理特殊帧（StartFrame、InterruptionFrame 等）
        await super().process_frame(frame, direction)

        # ✅ 转发 frame 到下一个处理器
        await self.push_frame(frame, direction)

    async def set_task(self, task: Any):
        """
        设置 PipelineTask

        用于注入系统指令。

        Args:
            task: PipelineTask 实例
        """
        self.skill_manager.set_task(task)

        # 重新激活技能（注入指令）
        if self.auto_activate:
            active_skills = list(self.skill_manager.active_skills.keys())
            for skill_name in active_skills:
                await self.skill_manager.activate_skill(skill_name)

    def register_skill_function(self, skill_name: str, handler: callable):
        """
        手动注册技能函数

        用于自定义函数处理。

        Args:
            skill_name: 技能名称
            handler: 处理函数

        示例：
        ```python
        async def custom_handler(params: FunctionCallParams):
            # 自定义逻辑
            await params.result_callback("Result")

        processor.register_skill_function("weather", custom_handler)
        ```
        """
        if self.llm_service:
            self.llm_service.register_function(f"skill_{skill_name}", handler)
            print(f"✓ Registered custom handler for skill: {skill_name}")

    async def inject_system_instructions(self, instructions: str):
        """
        注入系统指令

        向 LLM Context 添加系统消息。

        Args:
            instructions: 指令内容

        示例：
        ```python
        await processor.inject_system_instructions(
            "你是一个专业的天气助手。"
        )
        ```
        """
        if not self.skill_manager.task:
            print("⚠️  No task set, cannot inject instructions")
            return

        system_message = {
            "role": "system",
            "content": instructions
        }

        await self.skill_manager.task.queue_frames([
            LLMMessagesAppendFrame([system_message], run_llm=False)
        ])

    async def get_status(self) -> dict:
        """
        获取处理器状态

        Returns:
            dict: 状态信息

        示例：
        ```python
        status = await processor.get_status()
        print(f"Active skills: {status['active_skills']}")
        ```
        """
        manager_status = await self.skill_manager.get_status()

        return {
            **manager_status,
            "auto_activate": self.auto_activate,
            "has_llm_service": self.llm_service is not None,
            "has_mcp_manager": self.mcp_manager is not None,
        }
