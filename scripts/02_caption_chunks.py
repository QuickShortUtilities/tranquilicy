"""
Generate a text caption for each audio chunk.

Prefers real descriptive tags pulled from "<title> lyrics.txt" sidecar files
(matched to the source track by title) when available -- these carry genre,
BPM, and instrumentation detail written for the track, which is far richer
than anything auto-derived from the audio. Falls back to tempo/energy
analysis + filename hints for chunks with no matching txt file.

Also flags vocal vs. instrumental tracks (MusicGen is an instrumental
model -- it doesn't produce coherent lyrics/vocals, so mixing in a lot of
vocal-heavy training audio just adds noise). Use --exclude_vocals to drop
tracks confirmed to have vocals from the training set.

Produces manifest_with_captions.csv and an audiofolder-format metadata.csv,
which 03_prepare_dataset.py consumes.

Usage:
    python 02_caption_chunks.py --manifest "D:\musicgen_data\chunks\manifest.csv"
    python 02_caption_chunks.py --manifest "...\manifest.csv" --exclude_vocals
"""
import argparse
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from progress_util import ProgressWriter

STOPWORDS = {
    "the", "a", "an", "of", "feat", "featuring", "remaster", "remastered",
    "official", "audio", "video", "lyric", "lyrics", "hq", "hd", "full",
    "album", "track", "master", "final", "mix", "edit",
}

VOCAL_KEYWORDS = re.compile(r"\b(vocal|vocals|vocalist|male vocal|female vocal|sung|singing)\b", re.IGNORECASE)


def base_title(name: str) -> str:
    """Normalize 'NNN - Track Name [a1b2c3d4].mp3' / 'NNN - Track Name lyrics.txt' -> 'Track Name'."""
    n = re.sub(r"\.(mp3|wav|flac|m4a|ogg|txt)$", "", name, flags=re.IGNORECASE)
    n = re.sub(r"^\d+\s*-\s*", "", n)
    n = re.sub(r"\s*\[[0-9a-fA-F]{6,8}\]$", "", n)
    n = re.sub(r"\s*lyrics$", "", n, flags=re.IGNORECASE)
    return n.strip()


def filename_hints(source_file: str) -> str:
    stem = Path(source_file).stem
    stem = re.sub(r"^\d+[\s._-]*", "", stem)
    words = re.split(r"[\s._\-()\[\]]+", stem.lower())
    words = [w for w in words if w and not w.isdigit() and w not in STOPWORDS]
    return " ".join(words[:6])


def energy_label(rms_mean: float, quantiles) -> str:
    low, high = quantiles
    if rms_mean < low:
        return "low"
    if rms_mean > high:
        return "high"
    return "medium"


def analyze(path: str):
    y, sr = librosa.load(path, sr=32000, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).reshape(-1)[0])
    rms = float(np.mean(librosa.feature.rms(y=y)))
    return tempo, rms


def parse_lyrics_file(path: Path):
    """Returns (tags_str, has_vocals) or (None, None) if unparseable."""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None, None

    tags = None
    for line in lines:
        if line.strip().lower().startswith("tags:"):
            tags = line.split(":", 1)[1].strip()
            break
    if not tags:
        return None, None

    # Body after the blank line: "[Instrumental]" marker means no vocals,
    # anything else (actual lyric lines) means vocals are present.
    body_lines = [l.strip() for l in lines[2:] if l.strip()]
    body_text = " ".join(body_lines)
    is_marked_instrumental = bool(re.fullmatch(r"\[?\s*instrumental\s*\]?", body_text, re.IGNORECASE))
    has_vocals = (not is_marked_instrumental) or bool(VOCAL_KEYWORDS.search(tags))

    return tags, has_vocals


def load_tags_lookup(source_files: pd.Series, extra_dirs=None) -> dict:
    """Scan every unique parent folder of source_file (plus any extra_dirs --
    useful when files were renumbered/consolidated elsewhere and the tags
    .txt sidecars only still exist at their original location) for '*.txt'
    sidecars and build {base_title: (tags, has_vocals)}."""
    parent_dirs = {Path(f).parent for f in source_files.unique()}
    if extra_dirs:
        parent_dirs |= {Path(d) for d in extra_dirs}
    lookup = {}
    for d in parent_dirs:
        for txt_path in d.glob("*.txt"):
            tags, has_vocals = parse_lyrics_file(txt_path)
            if tags is None:
                continue
            lookup[base_title(txt_path.name)] = (tags, has_vocals)
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--genre_tag", default="chillout", help="used only for chunks with no matching tags file")
    ap.add_argument("--exclude_vocals", action="store_true", help="drop chunks confirmed to have vocals")
    ap.add_argument("--extra_tags_dir", action="append", default=None,
                     help="additional folder(s) to search for title-matched tags .txt sidecars "
                          "(repeatable) -- use when source files were renumbered/consolidated "
                          "elsewhere and the .txt files only still exist at their original location")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    df = pd.read_csv(manifest_path)

    tags_lookup = load_tags_lookup(df["source_file"], extra_dirs=args.extra_tags_dir)
    print(f"Loaded {len(tags_lookup)} title -> tags entries from lyrics/tags .txt sidecars")

    df["base_title"] = df["source_file"].apply(lambda f: base_title(Path(f).name))
    matched_mask = df["base_title"].isin(tags_lookup)
    print(f"{matched_mask.sum()} / {len(df)} chunks matched to a tags file; "
          f"{(~matched_mask).sum()} will use auto tempo/energy captioning")

    # Auto-analyze only the unmatched chunks (matched ones already have rich captions)
    tempos, rms_values = {}, {}
    unmatched_idx = df.index[~matched_mask]
    progress = ProgressWriter(stage="captioning", total=len(unmatched_idx))
    for n, i in enumerate(tqdm(unmatched_idx, desc="Analyzing audio (unmatched chunks only)"), 1):
        try:
            tempo, rms = analyze(df.loc[i, "chunk_path"])
        except Exception as e:
            print(f"  warn: analysis failed for {df.loc[i, 'chunk_path']} ({e})")
            tempo, rms = 90.0, 0.0
        tempos[i] = round(tempo)
        rms_values[i] = rms
        progress.update(n, Path(df.loc[i, "chunk_path"]).name)
    progress.finish()

    if rms_values:
        quantiles = (np.quantile(list(rms_values.values()), 0.33), np.quantile(list(rms_values.values()), 0.66))
    else:
        quantiles = (0.0, 0.0)

    captions, vocal_status, sources = [], [], []
    for idx, row in df.iterrows():
        title = row["base_title"]
        if title in tags_lookup:
            tags, has_vocals = tags_lookup[title]
            captions.append(tags)
            vocal_status.append("vocals" if has_vocals else "instrumental")
            sources.append("tags_file")
        else:
            hints = filename_hints(row["source_file"])
            energy = energy_label(rms_values.get(idx, 0.0), quantiles)
            parts = [args.genre_tag, f"{tempos.get(idx, 90)} bpm", f"{energy} energy"]
            if hints:
                parts.append(hints)
            captions.append(", ".join(parts))
            vocal_status.append("unknown")
            sources.append("auto")

    df["caption"] = captions
    df["vocal_status"] = vocal_status
    df["caption_source"] = sources
    df = df.drop(columns=["base_title"])

    print("\nVocal status breakdown:")
    print(df["vocal_status"].value_counts().to_string())

    if args.exclude_vocals:
        before = len(df)
        df = df[df["vocal_status"] != "vocals"].reset_index(drop=True)
        print(f"\n--exclude_vocals: dropped {before - len(df)} chunks, {len(df)} remain")

    out_path = manifest_path.parent / "manifest_with_captions.csv"
    df.to_csv(out_path, index=False)

    # HF `datasets` "audiofolder" format (file_name, text) for 03_prepare_dataset.py
    metadata = pd.DataFrame({
        "file_name": [Path(p).name for p in df["chunk_path"]],
        "text": df["caption"],
    })
    metadata_path = manifest_path.parent / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    print(f"\nWrote captions for {len(df)} chunks -> {out_path}")
    print(f"Wrote audiofolder metadata -> {metadata_path}")
    print("Spot-check a few rows and hand-edit captions before training if needed.")


if __name__ == "__main__":
    main()
