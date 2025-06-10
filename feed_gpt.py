# feed_me_clipboard_final.py
import argparse
import json
import os
import subprocess
import sys


def copy_to_clipboard_windows(text: str):
    """
    Copies the given text (UTF-8) to the Windows clipboard using PowerShell's Set-Clipboard,
    并在 PowerShell 中将 stdin 解码方式设置为 UTF-8，确保中文不会乱码。
    """
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            # 先把 PowerShell stdin 解码设为 UTF8，再把读到的全部文本写入剪贴板
            "[Console]::InputEncoding = [Text.Encoding]::UTF8;"
            "Set-Clipboard -Value ([Console]::In.ReadToEnd())",
        ]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)
        print("\n--- 内容已成功复制到剪贴板 (PowerShell, UTF-8 模式) ---")
    except FileNotFoundError:
        print("\n--- 错误：未找到 powershell.exe，请检查是否安装并在 PATH 中。 ---")
    except subprocess.CalledProcessError as e:
        print(f"\n--- PowerShell 异常退出，返回码：{e.returncode} ---")
    except Exception as e:
        print(f"\n--- 复制到剪贴板时发生意外错误：{e} ---")


def read_and_print_files(config_path: str, separator: str = "\n" + "=" * 80 + "\n"):
    print(f"--- 读取配置：{config_path} ---")
    try:
        with open(config_path, "r", encoding="utf-8-sig") as cfg:
            file_paths = [
                line.strip()
                for line in cfg
                if line.strip() and not line.strip().startswith("#")
            ]
    except FileNotFoundError:
        print(f"错误：配置文件未找到：'{config_path}'")
        return
    except Exception as e:
        print(f"读取配置文件时出错：{e}")
        return

    if not file_paths:
        print("配置文件中没有可用的路径。")
        return

    parts = []
    for i, path in enumerate(file_paths):
        parts.append(f"File: {path}")
        parts.append("-" * (len(path) + 6))

        if not os.path.exists(path):
            parts.append(f"(找不到文件：'{path}')")
        else:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".ipynb":
                # 处理 Jupyter Notebook，只提取 code cells
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        nb = json.load(f)
                    cells = nb.get("cells", [])
                    code_cells = [c for c in cells if c.get("cell_type") == "code"]
                    if not code_cells:
                        parts.append("(此 notebook 无代码单元)")
                    else:
                        for ci, cell in enumerate(code_cells, start=1):
                            parts.append(f"# ----- Code Cell [{ci}] -----")
                            src = cell.get("source", [])
                            # source 可能是 list，也可能是 str
                            if isinstance(src, list):
                                content = "".join(src)
                            else:
                                content = str(src)
                            parts.append(content.rstrip("\n"))
                except json.JSONDecodeError as e:
                    parts.append(f"(无效的 .ipynb 文件，JSON 解码失败: {e})")
                except Exception as e:
                    parts.append(f"(读取 .ipynb 时出错: {e})")
            else:
                # 处理普通文本 / 源码文件
                try:
                    with open(path, "r", encoding="utf-8-sig") as f:
                        parts.append(f.read().rstrip("\n"))
                except UnicodeDecodeError as e:
                    parts.append(f"(非 UTF-8 文件，无法读取：{e})")
                except Exception as e:
                    parts.append(f"(读取时出错：{e})")

        # 文件之间加分隔线
        if i < len(file_paths) - 1:
            parts.append(separator)

    final_output = "\n".join(parts)
    print("\n" + final_output)

    if sys.platform == "win32":
        copy_to_clipboard_windows(final_output)
    else:
        print(
            f"\n--- 当前系统不是 Windows，跳过剪贴板操作 (platform={sys.platform}) ---"
        )

    print(f"\n--- 共处理 {len(file_paths)} 个文件。---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="读取配置中的 Python 文件或 Jupyter Notebook，打印并复制到剪贴板 (仅限 Windows UTF-8)。",
        epilog="示例: python feed_me_clipboard_final.py --config feed_gpt_config.txt",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="feed_gpt_config.txt",
        help="列出要读取文件路径的配置文件，默认 'feed_gpt_config.txt'",
    )
    args = parser.parse_args()
    read_and_print_files(args.config)
