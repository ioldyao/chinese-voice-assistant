#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 集成测试脚本
测试传统模式和 Pipecat 模式的基本功能
"""

import sys


def test_traditional_mode_import():
    """测试传统模式导入"""
    print("=" * 60)
    print("测试 1: 传统模式导入")
    print("=" * 60)

    try:
        from src.voice_assistant import SmartWakeWordSystem
        print("✓ SmartWakeWordSystem 导入成功")

        from src.voice_assistant import ReactAgent
        print("✓ ReactAgent 导入成功")

        from src.voice_assistant import TTSManager
        print("✓ TTSManager 导入成功")

        print("\n✅ 传统模式导入测试通过\n")
        return True
    except Exception as e:
        print(f"\n❌ 传统模式导入失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_pipecat_mode_import():
    """测试 Pipecat 模式导入"""
    print("=" * 60)
    print("测试 2: Pipecat 模式导入")
    print("=" * 60)

    try:
        from src.voice_assistant import pipecat_main
        print("✓ pipecat_main 模块导入成功")

        from src.voice_assistant.pipecat_adapters import (
            SherpaKWSProcessor,
            SherpaASRProcessor,
            ReactAgentProcessor,
            PiperTTSProcessor,
        )
        print("✓ Pipecat 适配器导入成功")

        print("\n✅ Pipecat 模式导入测试通过\n")
        return True
    except Exception as e:
        print(f"\n❌ Pipecat 模式导入失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_main_entry():
    """测试主程序入口"""
    print("=" * 60)
    print("测试 3: 主程序入口")
    print("=" * 60)

    try:
        import main
        print("✓ main.py 导入成功")
        print("✓ main() 函数存在:", hasattr(main, 'main'))

        print("\n✅ 主程序入口测试通过\n")
        return True
    except Exception as e:
        print(f"\n❌ 主程序入口测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("🧪 Phase 1 集成测试")
    print("=" * 60 + "\n")

    results = []

    # 测试 1: 传统模式导入
    results.append(("传统模式导入", test_traditional_mode_import()))

    # 测试 2: Pipecat 模式导入
    results.append(("Pipecat 模式导入", test_pipecat_mode_import()))

    # 测试 3: 主程序入口
    results.append(("主程序入口", test_main_entry()))

    # 总结
    print("=" * 60)
    print("📊 测试总结")
    print("=" * 60)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status} - {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！Phase 1 集成成功！")
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
