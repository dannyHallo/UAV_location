#!/usr/bin/env python3
"""
检查项目依赖是否安装
"""
import sys


def check_dependencies():
    """检查所有必需的依赖"""
    print("=== 检查项目依赖 ===")

    dependencies = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("scipy", "SciPy"),
        ("tkinter", "Tkinter"),
    ]

    missing_deps = []

    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {display_name} - 已安装")
        except ImportError:
            print(f"❌ {display_name} - 未安装")
            missing_deps.append(display_name)

    print(f"\nPython 版本: {sys.version}")

    if missing_deps:
        print(f"\n⚠️ 缺少依赖: {', '.join(missing_deps)}")
        print("\n安装命令:")
        if "PyTorch" in missing_deps:
            print("pip install torch torchvision torchaudio")
        if "NumPy" in missing_deps:
            print("pip install numpy")
        if "Matplotlib" in missing_deps:
            print("pip install matplotlib")
        if "SciPy" in missing_deps:
            print("pip install scipy")
        if "Tkinter" in missing_deps:
            print("Tkinter 通常随 Python 一起安装，如果缺失请重新安装 Python")
    else:
        print("\n✅ 所有依赖都已安装！")


if __name__ == "__main__":
    check_dependencies()

