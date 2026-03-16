#!/usr/bin/env python3
"""Whisper speech-to-text on Metal GPU.

Usage:
    python whisper_transcribe.py audio.wav
    python whisper_transcribe.py audio.mp3 --model base --language es --task translate
"""
import argparse
import time
from applegpu_runtime.models.whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser(description="Whisper speech-to-text on Metal GPU")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default="tiny", choices=["tiny", "base"])
    parser.add_argument("--language", default=None)
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    args = parser.parse_args()

    print(f"Loading Whisper {args.model}...")
    t0 = time.time()
    model = WhisperModel(args.model)
    print(f"Loaded in {time.time() - t0:.1f}s")

    print(f"\nTranscribing: {args.audio}")
    t0 = time.time()
    text = model.transcribe(args.audio, language=args.language, task=args.task)
    elapsed = time.time() - t0

    print(f"\n{text}")
    print(f"\n--- Transcribed in {elapsed:.1f}s ---")


if __name__ == "__main__":
    main()
