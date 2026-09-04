"""
Generate a long-form (multi-minute) track by chaining MusicGen generations:
each new segment is conditioned on the tail of the previous one, so the
piece continues rather than restarting. MusicGen's own single-pass limit is
~20-30s (its trained context window), so this is the standard workaround.

Note: this produces a continuously-evolving texture/jam, not a structured
song with intro/build/outro -- MusicGen isn't trained on song arrangement,
so don't expect verse/chorus structure just because it's long.

Usage:
    python 08_generate_long.py --lora_dir "D:\musicgen_data\lora-out-v2" --prompt "..." --out out.wav --duration_sec 180
    python 08_generate_long.py --base_model_only --prompt "..." --out out.wav --duration_sec 180
"""
import argparse

import numpy as np
import soundfile as sf
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForTextToWaveform, AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_dir", default=None, help="trained LoRA adapter dir; omit with --base_model_only")
    ap.add_argument("--base_model_only", action="store_true", help="use unmodified facebook/musicgen-medium, no LoRA")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="output_long.wav")
    ap.add_argument("--duration_sec", type=float, default=180.0)
    ap.add_argument("--segment_new_tokens", type=int, default=1000, help="~20s of new audio per segment")
    ap.add_argument("--seed_sec", type=float, default=3.0, help="how much of the previous segment's tail to condition on")
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.base_model_only:
        model_id = "facebook/musicgen-medium"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForTextToWaveform.from_pretrained(model_id, dtype=torch.float16).to(device)
    else:
        config = PeftConfig.from_pretrained(args.lora_dir)
        model = AutoModelForTextToWaveform.from_pretrained(config.base_model_name_or_path, dtype=torch.float16)
        model = PeftModel.from_pretrained(model, args.lora_dir).to(device)
        processor = AutoProcessor.from_pretrained(args.lora_dir)

    sr = model.config.audio_encoder.sampling_rate
    seed_len = int(args.seed_sec * sr)

    full_audio = None
    total_len = 0
    target_len = int(args.duration_sec * sr)
    segment_num = 0

    while total_len < target_len:
        segment_num += 1
        if full_audio is None:
            inputs = processor(text=[args.prompt], padding=True, return_tensors="pt").to(device)
        else:
            seed = full_audio[-seed_len:]
            inputs = processor(text=[args.prompt], audio=seed, sampling_rate=sr, padding=True, return_tensors="pt").to(device)
            inputs["input_values"] = inputs["input_values"].half()

        out = model.generate(**inputs, do_sample=True, guidance_scale=args.guidance_scale, max_new_tokens=args.segment_new_tokens)
        seg = out[0, 0].detach().cpu().float().numpy()

        if full_audio is None:
            full_audio = seg
        else:
            full_audio = np.concatenate([full_audio, seg[seed_len:]])

        total_len = len(full_audio)
        print(f"segment {segment_num}: total so far {total_len / sr:.1f}s / {args.duration_sec:.0f}s")

    sf.write(args.out, full_audio[:target_len], sr)
    print(f"Wrote {args.out} ({len(full_audio[:target_len]) / sr:.1f}s @ {sr}Hz, {segment_num} segments)")


if __name__ == "__main__":
    main()
