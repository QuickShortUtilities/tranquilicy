"""
Slice a folder of full-length songs into fixed-length training clips.

Usage:
    python 01_chunk_audio.py --input_dir "D:\music\chillout" --output_dir "D:\musicgen_data\chunks" --chunk_seconds 30
"""
import argparse
import csv
from pathlib import Path

import librosa
import soundfile as sf

from progress_util import ProgressWriter

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff"}


def find_audio_files(input_dir: Path):
    return sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith(".")
    )


def chunk_file(path: Path, out_dir: Path, sr: int, chunk_seconds: float, min_seconds: float, rows: list):
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as e:
        print(f"  skip (load failed): {path.name} ({e})")
        return

    y, _ = librosa.effects.trim(y, top_db=40)

    chunk_len = int(chunk_seconds * sr)
    min_len = int(min_seconds * sr)
    n_chunks = max(1, len(y) // chunk_len)

    for i in range(n_chunks):
        start = i * chunk_len
        end = start + chunk_len
        clip = y[start:end]
        if len(clip) < min_len:
            continue

        out_name = f"{path.stem}_{i:03d}.wav"
        out_path = out_dir / out_name
        sf.write(str(out_path), clip, sr)

        rows.append({
            "chunk_path": str(out_path),
            "source_file": str(path),
            "start_sec": round(start / sr, 2),
            "end_sec": round(end / sr, 2),
            "duration_sec": round(len(clip) / sr, 2),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sr", type=int, default=32000, help="MusicGen expects 32kHz")
    ap.add_argument("--chunk_seconds", type=float, default=30.0)
    ap.add_argument("--min_seconds", type=float, default=10.0, help="drop trailing chunks shorter than this")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(input_dir)
    print(f"Found {len(files)} source tracks in {input_dir}")

    progress = ProgressWriter(stage="chunking", total=len(files))

    rows = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        chunk_file(path, output_dir, args.sr, args.chunk_seconds, args.min_seconds, rows)
        progress.update(i, path.name, chunks_written=len(rows))

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_path", "source_file", "start_sec", "end_sec", "duration_sec"])
        writer.writeheader()
        writer.writerows(rows)

    progress.finish(chunks_written=len(rows))
    print(f"\nWrote {len(rows)} chunks from {len(files)} tracks")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
