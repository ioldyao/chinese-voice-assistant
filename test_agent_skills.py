#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Agent Skills 集成

验证：
1. 技能加载
2. 技能激活
3. 技能执行
"""
import asyncio
from pathlib import Path


async def main():
    """测试主函数"""
    print("="*60)
    print("🧪 测试 Agent Skills 集成")
    print("="*60)

    # 导入
    from src.voice_assistant.skills import SkillManager

    # 1. 测试技能发现
    print("\n📂 测试 1: 技能发现")
    skills_dir = Path("skills")
    manager = SkillManager(skills_dir)

    await manager.discover_all()
    print(f"✓ 发现 {len(manager.skills)} 个技能")

    # 列出所有技能
    for name, skill in manager.skills.items():
        print(f"  - {name}: {skill.metadata.display_name}")

    # 2. 测试技能激活
    print("\n⚡ 测试 2: 技能激活")
    test_skill_name = "weather"

    if test_skill_name in manager.skills:
        success = await manager.activate_skill(test_skill_name)
        if success:
            print(f"✓ 成功激活技能: {test_skill_name}")

            # 检查状态
            skill = manager.get_skill(test_skill_name)
            print(f"  状态: {skill.state.value}")
            print(f"  指令长度: {len(skill.instructions) if skill.instructions else 0}")
        else:
            print(f"❌ 激活失败: {test_skill_name}")
    else:
        print(f"⚠️  技能不存在: {test_skill_name}")

    # 3. 测试技能执行
    print("\n🚀 测试 3: 技能执行")
    if manager.is_skill_active(test_skill_name):
        result = await manager.execute_skill(
            test_skill_name,
            user_input="今天北京天气怎么样？"
        )

        if result.success:
            print(f"✓ 执行成功")
            print(f"  内容预览: {result.content[:100]}...")
        else:
            print(f"❌ 执行失败: {result.error}")

    # 4. 测试批量激活
    print("\n📦 测试 4: 批量激活所有技能")
    await manager.activate_all()
    print(f"✓ 已激活 {len(manager.active_skills)} 个技能")

    # 5. 获取状态
    print("\n📊 测试 5: 获取状态")
    status = await manager.get_status()
    print(f"总技能数: {status['total_skills']}")
    print(f"已激活: {status['active_skills']}")
    print(f"未激活: {status['inactive_skills']}")

    print("\n" + "="*60)
    print("✅ 所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
