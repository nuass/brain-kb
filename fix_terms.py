#!/usr/bin/env python3
"""Apply ASR-error term dictionary to transcripts_distilled/*.md.

高置信度 ASR 错字 → 正确术语。重复运行幂等。
"""
from pathlib import Path

TERMS = [
    ("self coalition", "self correlation"),
    ("Multiplayer", "multiplier"),
    ("Alphayeah the rate per quarter", "alpha yield rate per quarter"),
    ("Alphayeah the rate", "alpha yield rate"),
    ("Alphayeah", "alpha yield"),
    ("窝矿", "WorldQuant"),
    ("Brand 平台", "BRAIN 平台"),
    ("Brand平台", "BRAIN平台"),
    ("五封钱利率", "无风险利率"),
    ("五封钱", "无风险"),
    ("比雷和GKT", "beta和GKT"),
    ("比雷为1", "beta为1"),
    ("比雷为 1", "beta 为 1"),
]

ROOT = Path(__file__).parent / "transcripts_distilled"


def apply(text: str) -> tuple[str, int]:
    n = 0
    for src, dst in TERMS:
        if src in text:
            n += text.count(src)
            text = text.replace(src, dst)
    return text, n


def main() -> None:
    total_files, total_subs = 0, 0
    for p in sorted(ROOT.glob("*.distilled.md")):
        raw = p.read_text(encoding="utf-8")
        fixed, n = apply(raw)
        if n:
            p.write_text(fixed, encoding="utf-8")
            print(f"[{n:3d}] {p.name}")
            total_files += 1
            total_subs += n
    print(f"---\nfiles touched: {total_files}, substitutions: {total_subs}")


if __name__ == "__main__":
    main()
