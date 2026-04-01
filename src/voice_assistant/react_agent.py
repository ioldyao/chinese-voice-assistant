"""React Agent - 使用 MCP 工具的智能代理（异步版本）"""
import json
import logging
import re
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DASHSCOPE_API_KEY, DASHSCOPE_API_URL,
    TTS_SERVICE,
    PIPER_TTS_MODEL_PATH,
    DASHSCOPE_TTS_MODEL,
    DASHSCOPE_TTS_VOICE,
    EDGE_TTS_VOICE,
    AZURE_TTS_API_KEY, AZURE_TTS_REGION, AZURE_TTS_VOICE,
    DASHSCOPE_REALTIME_MODEL,
    DASHSCOPE_REALTIME_VOICE,
)
from .mcp_client import MCPManager, MCPResponse
from .tts import TTSManager
from .vision import VisionUnderstanding


@dataclass
class ReActStep:
    """React 单步执行结果"""
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    success: bool
    timestamp: float  # 时间戳（秒）


class ReActParser:
    """
    ReAct 响应解析器

    解析 LLM 返回的 Thought/Action/Action Input 格式
    """

    THOUGHT_PATTERN = r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer)|\Z)"
    ACTION_PATTERN = r"Action:\s*(.*?)(?=\nAction Input|\n|$)"
    ACTION_INPUT_PATTERN = r"Action Input:\s*(\{.*?\})"
    FINAL_ANSWER_PATTERN = r"Final Answer:\s*(.*)"

    @staticmethod
    def parse(response: str) -> Optional[Dict[str, Any]]:
        """
        解析 LLM 响应

        Returns:
            {
                "thought": "思考内容",
                "action": "工具名称",
                "action_input": {...},
                "done": False/True
            }
        """
        try:
            # 检查是否完成
            final_match = re.search(ReActParser.FINAL_ANSWER_PATTERN, response, re.DOTALL)
            if final_match:
                return {
                    "thought": "任务已完成",
                    "action": None,
                    "action_input": {},
                    "done": True,
                    "final_answer": final_match.group(1).strip()
                }

            # 提取 Thought
            thought_match = re.search(ReActParser.THOUGHT_PATTERN, response, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""

            # 提取 Action
            action_match = re.search(ReActParser.ACTION_PATTERN, response)
            action = action_match.group(1).strip() if action_match else ""

            # 提取 Action Input - 严格 JSON 格式
            action_input = {}
            action_input_index = response.find("Action Input:")
            if action_input_index != -1:
                try:
                    # 找到第一个 {
                    start_idx = response.find("{", action_input_index)
                    if start_idx != -1:
                        # 使用栈来找到匹配的 }
                        brace_count = 0
                        end_idx = start_idx
                        for i in range(start_idx, len(response)):
                            if response[i] == '{':
                                brace_count += 1
                            elif response[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break

                        json_str = response[start_idx:end_idx]

                        # 仅做最小兼容：Python 布尔值转 JSON 格式
                        json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')

                        # 严格解析 JSON（不兼容单引号等其他格式）
                        action_input = json.loads(json_str)

                        print(f"[调试] 解析到的参数: {action_input}")

                except json.JSONDecodeError as e:
                    print(f"❌ Action Input 格式错误: {e}")
                    print(f"   原始内容: {response[start_idx:end_idx]}")
                    print(f"   请使用标准 JSON 格式: {{\"key\": \"value\"}}")
                except Exception as e:
                    print(f"[调试] 提取 JSON 失败: {e}")
            else:
                print(f"[调试] 未找到 Action Input")

            return {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "done": False
            }

        except Exception as e:
            logging.error(f"解析 ReAct 响应失败: {e}")
            import traceback
            traceback.print_exc()
            return None


class ReactAgent:
    """
    React (Reasoning and Acting) Agent

    基于 ReAct 框架的智能代理（异步版本）：
    1. Thought: 分析当前状态，思考下一步
    2. Action: 选择并执行 MCP 工具
    3. Observation: 观察执行结果
    4. 循环直到任务完成

    仅支持异步执行（用于 Pipecat 模式）
    """

    def __init__(self, api_url=None, api_key=None):
        self.api_url = api_url or DASHSCOPE_API_URL
        self.api_key = api_key or DASHSCOPE_API_KEY
        self.logger = logging.getLogger(__name__)

        # MCP Manager（管理多个 MCP Server）- 异步版本
        self.mcp = MCPManager()

        # TTS（从 .env 配置读取引擎类型）
        self.tts = self._create_tts()

        # Vision（视觉理解）
        self.vision = VisionUnderstanding(api_url, api_key)

        # React 历史记录（短期记忆：当前对话）
        self.history: List[ReActStep] = []

        # 可用工具列表
        self.available_tools: List[Dict[str, Any]] = []

        # 最大步数（防止死循环）
        self.max_steps = 15

        # 长期记忆（跨会话持久化）
        self.long_term_memory = {
            "summary": None,  # LLM 自动生成的状态总结
            "last_update": None,  # 最后更新时间
        }

        # 中断标志（用于在执行中检测唤醒词）
        self.interrupt_flag = False

        # 执行状态标志（标识是否正在执行任务）
        self.is_executing = False

    async def execute_command_async(self, user_command: str, enable_voice: bool = False) -> Dict:
        """
        执行用户命令 - 异步版本（用于 Pipecat 模式）

        基于 MCP 官方推荐模式：完全异步执行

        Args:
            user_command: 用户指令
            enable_voice: 是否语音播报（Pipecat 模式下通常为 False，由 TTS Processor 处理）

        Returns:
            执行结果
        """
        self.logger.info(f"🤖 开始执行 (async): {user_command}")

        # 标记为执行中
        self.is_executing = True
        self.interrupt_flag = False

        try:
            # Pipecat 模式：目前只支持 React 模式（浏览器操作）
            # Vision 模式可以后续添加
            self.logger.info("使用 React 模式 (async)")
            return await self._react_mode_async(user_command, enable_voice)
        finally:
            # 执行完成，清除执行标志
            self.is_executing = False
            self.interrupt_flag = False

    async def _react_mode_async(self, user_command: str, enable_voice: bool = False) -> Dict:
        """
        React 操作模式 - 异步版本（用于 Pipecat 模式）

        基于官方推荐模式：完全异步的 React 循环

        Args:
            user_command: 用户指令
            enable_voice: 是否语音播报（Pipecat 模式下通常为 False）

        Returns:
            执行结果
        """
        # 重置历史
        self.history = []

        # React 循环
        for step in range(self.max_steps):
            # 检查中断标志
            if self.interrupt_flag:
                print("\n⚠️ 检测到中断请求，停止执行")
                self.logger.warning("用户中断执行")
                return {
                    "success": False,
                    "message": "用户中断",
                    "steps": step,
                    "interrupted": True
                }

            print(f"\n--- 步骤 {step + 1} ---")
            self.logger.info(f"\n--- Step {step + 1} ---")

            # 1. LLM 思考：下一步做什么
            parsed_action = self._think(user_command)

            if not parsed_action:
                print("❌ 思考失败")
                self.logger.error("❌ 思考失败")
                break

            # 2. 判断是否完成
            if parsed_action.get("done", False):
                print("✅ 任务完成")
                self.logger.info("✅ 任务完成")
                return {
                    "success": True,
                    "message": parsed_action.get("final_answer", "任务完成"),
                    "steps": step + 1
                }

            # 3. 执行动作（异步）
            print(f"🎯 执行: {parsed_action['action']}")
            print(f"   参数: {parsed_action['action_input']}")
            observation = await self._execute_action_async(
                parsed_action["action"],
                parsed_action["action_input"]
            )

            # 执行后再次检查中断
            if self.interrupt_flag:
                print("\n⚠️ 检测到中断请求，停止执行")
                self.logger.warning("用户中断执行")
                return {
                    "success": False,
                    "message": "用户中断",
                    "steps": step + 1,
                    "interrupted": True
                }

            # 4. 显示结果
            if observation and observation.success:
                print(f"✓ 成功: {observation.content[:100] if observation.content else '执行成功'}")
            else:
                error_msg = observation.error if observation else '未知错误'
                print(f"✗ 失败: {error_msg}")

            # 5. 记录历史（带时间戳）
            import time
            self.history.append(ReActStep(
                thought=parsed_action["thought"],
                action=parsed_action["action"],
                action_input=parsed_action["action_input"],
                observation=observation.content if observation else "执行失败",
                success=observation.success if observation else False,
                timestamp=time.time()
            ))

            # 6. 如果失败，继续尝试调整策略
            if not (observation and observation.success):
                self.logger.warning(f"⚠️ 步骤失败: {observation.error if observation else '未知错误'}")

        # 超过最大步数
        self.logger.warning("⚠️ 超过最大步数，任务未完成")
        return {
            "success": False,
            "message": "超过最大步数",
            "steps": self.max_steps
        }

    def _think(self, user_command: str) -> Optional[Dict[str, Any]]:
        """
        思考：根据用户命令和历史记录，决定下一步动作

        Returns:
            {
                "thought": "我的思考...",
                "action": "Click-Tool",
                "action_input": {"x": 100, "y": 200},
                "done": False
            }
        """
        # 构造提示词
        prompt = self._build_react_prompt(user_command)

        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen-plus",
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.1
                },
                timeout=30  # 增加超时时间：15秒 → 30秒
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # 调试：显示 LLM 原始响应
                print(f"\n[调试] LLM 响应:\n{content}\n")

                # 使用 ReActParser 解析
                parsed = ReActParser.parse(content)

                if parsed:
                    # 显示思考内容
                    print(f"💭 思考: {parsed.get('thought', '')[:100]}")
                    self.logger.info(f"💭 思考: {parsed.get('thought', '')}")
                    if not parsed.get("done"):
                        self.logger.info(f"🎯 动作: {parsed.get('action')} {parsed.get('action_input', {})}")
                else:
                    print("[调试] ❌ 解析失败")

                return parsed
            else:
                self.logger.error(f"LLM 请求失败: {response.status_code}")
                print(f"⚠️ LLM 请求失败: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            self.logger.error("LLM 请求超时")
            print("⚠️ LLM 请求超时，请检查网络连接")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"LLM 网络请求错误: {e}")
            print(f"⚠️ LLM 网络请求错误: {e}")
            return None
        except Exception as e:
            self.logger.error(f"思考失败: {e}")
            print(f"⚠️ 思考失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        tool_descriptions = self._format_tool_descriptions()

        # 临时调试：显示工具描述（只在第一次调用时）
        if not hasattr(self, '_prompt_shown'):
            print("\n[调试] 工具参数格式示例:")
            lines = tool_descriptions.split('\n')
            for line in lines[:20]:  # 只显示前20行
                print(f"  {line}")
            if len(lines) > 20:
                print(f"  ... (共 {len(lines)} 行)")
            print()
            self._prompt_shown = True

        return f"""你是一个智能助手，同时使用 Windows-MCP 和 Playwright-MCP 工具完成用户任务。

按照 ReAct (Reasoning and Acting) 框架思考和行动：
1. Thought: 分析当前情况，思考下一步
2. Action: 选择一个工具执行
3. Action Input: 提供工具参数
4. Observation: 观察执行结果（由系统提供）
5. 重复以上步骤直到完成

工具选择策略：
• 浏览器操作（导航、点击网页元素、填写表单等）→ **必须使用 Playwright 工具**（browser_*）
  - 例如：browser_navigate、browser_click、browser_type、browser_snapshot
  - ⚠️ 浏览器操作前必须先调用 browser_snapshot 获取页面元素
  - ⚠️ 只能使用 browser_snapshot 返回的 ref（格式：e38、e77等），不能使用 State-Tool 的 ref（格式：49、50等）
  - Playwright 更快、更可靠、理解网页结构
• 桌面操作（打开应用、系统快捷键、窗口管理等）→ 使用 Windows 工具
  - 例如：App-Tool、Shortcut-Tool、Desktop-Tool
  - State-Tool 只用于了解桌面状态，不用于浏览器内操作

可用工具：
{tool_descriptions}

输出格式：
Thought: [你的思考过程]
Action: [工具名称]
Action Input: {{"param": "value"}}

⚠️ Action Input 必须是有效的 JSON 格式：
✅ 正确：Action Input: {{"ref": "e386", "element": "link"}}
❌ 错误：Action Input:\n  ref: e386\n  element: link
❌ 错误：Action Input: {{'ref': 'e386'}}  (不能用单引号)

如果任务完成，输出：
Thought: 任务已完成
Final Answer: [总结结果]

重要规则：
1. 每次只执行一个动作
2. 浏览器操作必须使用 Playwright 工具（browser_*），先 browser_snapshot 再操作
3. 不要混用 State-Tool 和 browser_* 的 ref 系统
4. 优先使用快捷键和简单操作，避免复杂流程
5. 如果任务不清晰或无法理解，直接返回 Final Answer 说明原因
6. 最多 15 步必须完成，但应尽快完成任务
7. 如果连续失败 3 次，立即停止并返回 Final Answer"""

    def _format_tool_descriptions(self) -> str:
        """格式化工具描述"""
        if not self.available_tools:
            return "暂无可用工具"

        descriptions = []
        # 不再限制数量，显示所有工具
        for tool in self.available_tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            schema = tool.get("input_schema", {})

            # 提取必需参数
            required = schema.get("required", [])
            properties = schema.get("properties", {})

            # 构造参数说明
            params = []
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                is_required = param_name in required
                param_desc = param_info.get("description", "")

                if is_required:
                    params.append(f"    - {param_name} ({param_type}, 必需): {param_desc}")
                else:
                    params.append(f"    - {param_name} ({param_type}, 可选): {param_desc}")

            if params:
                descriptions.append(f"- {name}: {desc}\n  参数:\n" + "\n".join(params))
            else:
                descriptions.append(f"- {name}: {desc}")

        return "\n".join(descriptions)

    def _build_react_prompt(self, user_command: str) -> str:
        """构造 ReAct 提示词"""

        # 长期记忆（跨对话的状态）
        memory_context = ""
        if self.long_term_memory["summary"]:
            memory_context = f"\n上次状态: {self.long_term_memory['summary']}\n"

        # 短期记忆（5分钟时间窗口）
        history_text = ""
        if self.history:
            import time
            current_time = time.time()
            five_minutes_ago = current_time - 300  # 300秒 = 5分钟

            # 筛选5分钟内的步骤
            recent_steps = [
                step for step in self.history
                if step.timestamp >= five_minutes_ago
            ]

            if recent_steps:
                history_text = "\n最近5分钟的操作:\n"
                for i, step in enumerate(recent_steps, 1):
                    history_text += f"\nStep {i}:\n"
                    history_text += f"Thought: {step.thought}\n"
                    history_text += f"Action: {step.action}\n"
                    history_text += f"Action Input: {step.action_input}\n"
                    history_text += f"Observation: {step.observation}\n"

        prompt = f"""用户任务: {user_command}
{memory_context}{history_text}

请分析当前情况，决定下一步动作。

重要提示：
1. 如果上次状态显示已在目标页面，先用 browser_snapshot 确认，避免重复操作
2. "输入XXX" 指的是在输入框/搜索框中输入文字，不是访问网站"""

        return prompt

    async def _execute_action_async(self, action: str, action_input: Dict[str, Any]) -> Optional[MCPResponse]:
        """
        执行动作（异步版本，用于 Pipecat 模式）

        基于 MCP 官方推荐模式：直接使用 await call_tool_async()

        Args:
            action: 工具名称
            action_input: 工具参数

        Returns:
            MCPResponse 对象
        """
        if not action:
            return MCPResponse(success=False, error="未指定工具")

        try:
            # 确保参数不为空
            if not action_input:
                action_input = {}

            self.logger.debug(f"执行工具 (async): {action}, 参数: {action_input}")

            # 使用官方推荐的异步调用方式
            result = await self.mcp.call_tool_async(action, action_input)
            return result

        except Exception as e:
            self.logger.error(f"执行动作失败 (async): {e}")
            import traceback
            traceback.print_exc()
            return MCPResponse(success=False, error=str(e))

    def _update_memory_async(self):
        """更新长期记忆（让 LLM 自动总结当前状态）"""
        try:
            # 如果没有实际操作历史，跳过记忆更新
            if not self.history:
                self.logger.debug("无操作历史，跳过记忆更新")
                return

            # 构造总结提示词
            recent_actions = ""
            if self.history:
                recent_actions = "最近的操作:\n"
                for i, step in enumerate(self.history[-5:], 1):  # 最近5步
                    recent_actions += f"{i}. {step.action}"
                    if step.action_input:
                        recent_actions += f" {step.action_input}"
                    recent_actions += f" → {step.observation[:100]}\n"

            summarize_prompt = f"""请用1句话总结当前浏览器状态，包括：正在哪个网站、做了什么操作。

{recent_actions}

要求：
- 基于以上实际操作生成总结
- 只返回1句话，不要其他内容
- 格式：浏览器当前在[网站]，[最后操作]

总结："""

            # 调用 LLM 生成总结
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen-plus",
                    "messages": [
                        {"role": "user", "content": summarize_prompt}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.1  # 降低温度，减少随机性
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                summary = result["choices"][0]["message"]["content"].strip()

                # 清理可能的前缀
                summary = summary.replace("总结：", "").strip()

                # 更新长期记忆
                self.long_term_memory["summary"] = summary
                import time
                self.long_term_memory["last_update"] = time.time()

                print(f"💾 记忆: {summary}")
                self.logger.info(f"更新记忆: {summary}")
            else:
                self.logger.warning(f"生成记忆总结失败: {response.status_code}")

        except Exception as e:
            self.logger.warning(f"更新记忆失败: {e}")

    def _create_tts(self) -> TTSManager:
        """
        根据 .env 配置创建 TTS 引擎

        支持的引擎：
        - piper: 本地神经网络 TTS（默认，最快）
        - dashscope: 阿里云 TTS（音质好）
        - edge: 微软 Edge TTS（免费）
        - azure: Azure Speech Services（高质量）
        - coqui: Coqui TTS（本地）

        Returns:
            TTSManager: TTS 管理器实例
        """
        tts_config = {
            "piper": {
                "engine_type": "piper",
                "model_path": PIPER_TTS_MODEL_PATH,
            },
            "dashscope": {
                "engine_type": "dashscope",
                "api_key": DASHSCOPE_API_KEY,
                "model": DASHSCOPE_TTS_MODEL,
                "voice": DASHSCOPE_TTS_VOICE,
            },
            "dashscope_realtime": {
                # ReactAgent 暂不支持 WebSocket Realtime
                # 降级使用 DashScope HTTP 流式
                "engine_type": "dashscope",
                "api_key": DASHSCOPE_API_KEY,
                "model": DASHSCOPE_TTS_MODEL,
                "voice": DASHSCOPE_TTS_VOICE,
            },
            "edge": {
                "engine_type": "edge",
                "voice": EDGE_TTS_VOICE,
            },
            "azure": {
                "engine_type": "azure",
                "api_key": AZURE_TTS_API_KEY,
                "voice": AZURE_TTS_VOICE,
            },
            "coqui": {
                "engine_type": "coqui",
            },
        }

        # 获取当前选择的 TTS 配置
        service = TTS_SERVICE.lower()
        if service not in tts_config:
            self.logger.warning(
                f"不支持的 TTS_SERVICE: {TTS_SERVICE}，使用默认的 piper"
            )
            service = "piper"

        # 特殊处理：dashscope_realtime 降级到 dashscope
        if service == "dashscope_realtime":
            print("⚠️  ReactAgent 暂不支持 WebSocket Realtime TTS，降级使用 DashScope HTTP")
            service = "dashscope"

        config = tts_config[service]

        # 过滤掉 None 值
        config = {k: v for k, v in config.items() if v is not None}

        # 创建 TTS 引擎
        return TTSManager(**config)
