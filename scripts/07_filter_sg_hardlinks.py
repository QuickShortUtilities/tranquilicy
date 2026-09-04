"""
Build a filtered view of the sg chunks (excluding confirmed-vocal chunks)
using hardlinks -- same-volume, zero extra disk space, no copying.
"""
import os
from pathlib import Path

import pandas as pd

SG_CHUNKS_DIR = Path(r"C:\Users\Gaming PC\musicgen_data\chunks")
FILTERED_DIR = Path(r"C:\Users\Gaming PC\musicgen_data\chunks_filtered")

def main():
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SG_CHUNKS_DIR / "manifest_with_captions.csv")
    df = df[df["vocal_status"] != "vocals"].reset_index(drop=True)
    print(f"Hardlinking {len(df)} non-vocal chunks...")

    rows = []
    linked, skipped = 0, 0
    for _, row in df.iterrows():
        src = Path(row["chunk_path"])
        dst = FILTERED_DIR / src.name
        if not dst.exists():
            try:
                os.link(src, dst)
                linked += 1
            except FileNotFoundError:
                skipped += 1
                continue
        rows.append({"file_name": src.name, "text": row["caption"]})

    metadata = pd.DataFrame(rows)
    metadata.to_csv(FILTERED_DIR / "metadata.csv", index=False)
    print(f"Linked {linked} new, {skipped} skipped (missing source), {len(rows)} total in metadata.csv")


if __name__ == "__main__":
    main()
