#!/usr/bin/env python3
"""Transcribe audio with faster-whisper, output .auc.md format."""
from pathlib import Path

from faster_whisper import WhisperModel

AUDIO_PATH = Path("/Users/cony.zhangbjgmail.com/dev/wq-brain/knowledge/audio/带你读论文第三课_2025_01_15.mp3")
OUT_PATH = Path("/Users/cony.zhangbjgmail.com/dev/wq-brain/knowledge/transcripts/带你读论文第三课_2025_01_15.auc.md")


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def main():
    print("Loading model...")
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    print(f"Transcribing {AUDIO_PATH.name} ...")
    segments, info = model.transcribe(str(AUDIO_PATH), language="zh", beam_size=5)
    print(f"Detected language: {info.language} (probability {info.language_probability:.2f})")

    lines = [f"# {AUDIO_PATH.stem}", ""]
    count = 0
    for seg in segments:
        text = seg.text.strip()
        if text:
            lines.append(f"[{fmt_time(seg.start)} - {fmt_time(seg.end)}] {text}")
            count += 1
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved to {OUT_PATH} ({count} segments)")


if __name__ == "__main__":
    main()
