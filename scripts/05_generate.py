"""
Generate a new chillout clip from your fine-tuned LoRA adapter.

Usage:
    python 05_generate.py --lora_dir "./musicgen-dreamboothing/chillout-musicgen-lora" --prompt "chillout, 85 bpm, low energy, warm rhodes chords, vinyl crackle" --out output.wav
"""
import argparse

import soundfile as sf
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForTextToWaveform, AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_dir", required=True, help="local dir or hub repo id of the trained LoRA adapter")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="output.wav")
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    ap.add_argument("--max_new_tokens", type=int, default=1024, help="~256 tokens per 5s of audio")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = PeftConfig.from_pretrained(args.lora_dir)
    model = AutoModelForTextToWaveform.from_pretrained(
        config.base_model_name_or_path, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(model, args.lora_dir).to(device)
    processor = AutoProcessor.from_pretrained(args.lora_dir)

    inputs = processor(text=[args.prompt], padding=True, return_tensors="pt").to(device)

    audio_values = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=args.guidance_scale,
        max_new_tokens=args.max_new_tokens,
    )

    sampling_rate = model.config.audio_encoder.sampling_rate
    sf.write(args.out, audio_values[0].T.cpu().float().numpy(), sampling_rate)
    print(f"Wrote {args.out} ({audio_values.shape[-1] / sampling_rate:.1f}s @ {sampling_rate}Hz)")


if __name__ == "__main__":
    main()
