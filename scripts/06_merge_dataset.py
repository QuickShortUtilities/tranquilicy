"""
Merge the original sg-derived chunks (excluding confirmed-vocal chunks) with
the 4 new CC-licensed chunk buckets into one combined audiofolder dataset,
without copying the (large) original chunks -- a directory junction points
at them in place.

Usage:
    python 06_merge_dataset.py
"""
import subprocess
from pathlib import Path

import pandas as pd

FINAL_DIR = Path(r"D:\musicgen_data\chunks_final")
SG_CHUNKS_DIR = Path(r"C:\Users\Gaming PC\musicgen_data\chunks")
CC_BUCKETS = ["cc_meditation_relaxing", "cc_deep_house_yoga", "cc_triphop_downtempo", "cc_liquid_dnb"]


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    sg_link = FINAL_DIR / "sg"
    if not sg_link.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(sg_link), str(SG_CHUNKS_DIR)], check=True)
        print(f"Created junction {sg_link} -> {SG_CHUNKS_DIR}")

    rows = []

    sg_df = pd.read_csv(SG_CHUNKS_DIR / "manifest_with_captions.csv")
    before = len(sg_df)
    sg_df = sg_df[sg_df["vocal_status"] != "vocals"].reset_index(drop=True)
    print(f"sg: {before} -> {len(sg_df)} chunks after excluding confirmed vocals")
    for _, row in sg_df.iterrows():
        rows.append({"file_name": f"sg/{Path(row['chunk_path']).name}", "text": row["caption"]})

    for bucket in CC_BUCKETS:
        bdir = FINAL_DIR / bucket
        bdf = pd.read_csv(bdir / "manifest_with_captions.csv")
        print(f"{bucket}: {len(bdf)} chunks")
        for _, row in bdf.iterrows():
            rows.append({"file_name": f"{bucket}/{Path(row['chunk_path']).name}", "text": row["caption"]})

    metadata = pd.DataFrame(rows)
    out_path = FINAL_DIR / "metadata.csv"
    metadata.to_csv(out_path, index=False)
    print(f"\nWrote combined metadata.csv with {len(metadata)} rows -> {out_path}")


if __name__ == "__main__":
    main()
