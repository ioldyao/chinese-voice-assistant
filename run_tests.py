#!/usr/bin/env python3
"""
运行 Agent Skills v3.0 测试
"""
import sys
import subprocess

def main():
    """运行测试"""
    print("=" * 60)
    print("🧪 运行 Agent Skills v3.0 测试")
    print("=" * 60)

    # 测试文件列表
    test_files = [
        "tests/test_skill_registry.py",
        "tests/test_skill_matcher.py",
        "tests/test_skill_integration.py",
    ]

    # 运行每个测试文件
    for test_file in test_files:
        print(f"\n📋 运行 {test_file}...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v"],
            capture_output=False
        )

        if result.returncode != 0:
            print(f"❌ {test_file} 测试失败")
        else:
            print(f"✅ {test_file} 测试通过")

    print("\n" + "=" * 60)
    print("✨ 测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
