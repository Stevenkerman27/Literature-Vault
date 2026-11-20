#!/usr/bin/env python3
"""Check Obsidian-compatible Markdown math delimiters throughout the vault."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BLOCK = re.compile(r"^\s*(?:>\s*)?\$\$\s*$")
FENCE = re.compile(r"^\s*(?:>\s*)?(\x60{3,}|~{3,})")
BAD_INLINE = re.compile(r"(?<!\\)\\\(")
BAD_DISPLAY = re.compile(r"(?<!\\)\\\[")
BARE_ENV = re.compile(r"\\begin\{(?:equation|align|gather|multline)\*?\}")
CURRENCY = re.compile(r"(?<!\\)\$\d+(?:\.\d+)?(?![\d.])(?=\s|$|[,，。.;；|])")
BACKTICKS = re.compile(r"\x60+[^\x60\n]*\x60+")
DOLLARS = re.compile(r"(?<!\\)\$")


def inspect(path: Path) -> tuple[list[str], int]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    errors: list[str] = []
    in_yaml = bool(lines and lines[0].strip() == "---")
    in_fence: str | None = None
    in_display = False
    display_started = 0
    block_count = 0

    for number, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if in_yaml:
            if number > 1 and stripped == "---":
                in_yaml = False
            continue
        fence = FENCE.match(raw)
        if fence:
            marker = fence.group(1)
            if in_fence is None:
                in_fence = marker[0]
            elif in_fence == marker[0]:
                in_fence = None
            continue
        if in_fence is not None:
            continue

        line = BACKTICKS.sub("", raw)
        if BLOCK.fullmatch(line):
            if in_display:
                in_display = False
                block_count += 1
            else:
                in_display = True
                display_started = number
            continue
        if in_display:
            if "$$" in line:
                errors.append(f"{path}:{number}: 展示公式的 $$ 应单独占行")
            continue

        if "$$" in line:
            errors.append(f"{path}:{number}: 展示公式的 $$ 应单独占行")
        if BAD_INLINE.search(line):
            errors.append(f"{path}:{number}: Obsidian 行内公式请使用 $...$，不要使用 \\(...\\)")
        if BAD_DISPLAY.search(line):
            errors.append(f"{path}:{number}: Obsidian 展示公式请使用 $$...$$，不要使用 \\[...\\]")
        if BARE_ENV.search(line):
            errors.append(f"{path}:{number}: LaTeX 数学环境应放进独立的 $$ 公式块")
        # Do not interpret ordinary dollar amounts (e.g. $1.87) as math delimiters.
        dollar_count = len(DOLLARS.findall(line))
        # A numeric math formula such as $2.5^\\circ$ is NOT a price.
        # Only disregard an unpaired dollar when the entire token is a price.
        is_currency = dollar_count == 1 and bool(CURRENCY.search(line))
        if dollar_count % 2 and not is_currency:
            errors.append(f"{path}:{number}: 行内 $ 分隔符数量不成对")

    if in_display:
        errors.append(f"{path}:{display_started}: 展示公式缺少结束 $$")
    return errors, block_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="vault 根目录（默认当前目录）")
    args = parser.parse_args()
    root = Path(args.root)
    files = sorted(p for p in root.rglob("*.md") if ".git" not in p.parts)
    all_errors: list[str] = []
    blocks = 0
    for path in files:
        errors, count = inspect(path)
        all_errors.extend(errors)
        blocks += count
    for error in all_errors:
        print(error)
    print(f"已检查 {len(files)} 篇 Markdown，{blocks} 个独立展示公式，发现 {len(all_errors)} 个格式问题。")
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
