@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
uv run python -m src.voice_assistant.pipecat_flows_main
