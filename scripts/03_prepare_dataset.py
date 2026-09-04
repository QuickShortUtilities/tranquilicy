"""
Sanity-check that the chunks folder from 01/02 loads correctly as a
Hugging Face `datasets` dataset before kicking off a multi-hour training run.

No Hugging Face account or upload needed -- `datasets` auto-detects a local
folder of audio files + metadata.csv as a single "train" split, which is
exactly what 04_train_lora.ps1 points at directly.

Usage:
    python 03_prepare_dataset.py --chunks_dir "C:\musicgen_data\chunks"
"""
import argparse

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", required=True)
    ap.add_argument("--sample_size", type=int, default=50)
    args = ap.parse_args()

    print(f"Loading a {args.sample_size}-row sample from {args.chunks_dir} ...")
    ds = load_dataset(args.chunks_dir, split=f"train[:{args.sample_size}]")
    print(ds)

    row = ds[0]
    print("\nSample row:")
    print("  text:", row["text"])
    print("  audio sampling_rate:", row["audio"]["sampling_rate"])
    print("  audio array length:", len(row["audio"]["array"]))

    print(f"\nLooks good. Point 04_train_lora.ps1 -DatasetDir at:\n  {args.chunks_dir}")


if __name__ == "__main__":
    main()
