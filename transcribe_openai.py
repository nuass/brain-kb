#!/usr/bin/env python3
"""Transcribe audio with OpenAI Whisper API, output .auc.md format."""
import json
import os
from pathlib import Path

from openai import OpenAI

AUDIO_PATH = Path("/Users/cony.zhangbjgmail.com/dev/wq-brain/knowledge/audio/带你读论文第三课_2025_01_15.mp3")
OUT_PATH = Path("/Users/cony.zhangbjgmail.com/dev/wq-brain/knowledge/transcripts/带你读论文第三课_2025_01_15.auc.md")

def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print(f"Transcribing {AUDIO_PATH.name} ...")
    with AUDIO_PATH.open("rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="zh",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = result.segments or []
    lines = [f"# {AUDIO_PATH.stem}", ""]
    for seg in segments:
        start = seg.start
        end = seg.end
        text = seg.text.strip()
        if text:
            lines.append(f"[{fmt_time(start)} - {fmt_time(end)}] {text}")
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved to {OUT_PATH} ({len(segments)} segments, {len(lines)} lines)")

if __name__ == "__main__":
    main()
