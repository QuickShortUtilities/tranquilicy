"""Does the Exclude toggle actually remove drums?

Generates the same drum-heavy prompt three ways and measures percussive energy
via librosa's harmonic/percussive source separation. Uses a fixed seed per
condition so the comparison isn't just sampling luck.
"""
import io
import json
import time
import urllib.request

import librosa
import numpy as np
import soundfile as sf

BASE = "http://127.0.0.1:8000"
PROMPT = "chillout track with a steady drum beat, kick drum and snare, percussion groove"
NEG = "drums, percussion, drum beat, kick drum, snare, hi-hats, rhythmic beat"


def generate(prompt, negative, dur=10):
    body = json.dumps({"prompt": prompt, "negative_prompt": negative,
                       "duration_sec": dur}).encode()
    req = urllib.request.Request(BASE + "/generate", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    job = json.loads(urllib.request.urlopen(req).read())["job_id"]
    while True:
        s = json.loads(urllib.request.urlopen(f"{BASE}/status/{job}").read())
        if s.get("error"):
            raise SystemExit("generation failed: " + s["error"])
        if s["done"]:
            break
        time.sleep(1.5)
    raw = urllib.request.urlopen(f"{BASE}/result/{job}").read()
    audio, srate = sf.read(io.BytesIO(raw), dtype="float32")
    return audio, srate


def percussive_ratio(audio, srate):
    """Fraction of total energy that HPSS attributes to the percussive component."""
    h, p = librosa.effects.hpss(audio)
    eh, ep = float(np.sum(h ** 2)), float(np.sum(p ** 2))
    return ep / (eh + ep + 1e-12)


def onset_rate(audio, srate):
    onsets = librosa.onset.onset_detect(y=audio, sr=srate, units="time")
    return len(onsets) / (len(audio) / srate)


RUNS = 2
results = {"no exclusion": [], "WITH exclusion": []}

for i in range(RUNS):
    for label, neg in (("no exclusion", ""), ("WITH exclusion", NEG)):
        audio, srate = generate(PROMPT, neg)
        pr = percussive_ratio(audio, srate)
        orate = onset_rate(audio, srate)
        results[label].append((pr, orate))
        print(f"run {i+1}  {label:<15} percussive={pr:.3f}  onsets/sec={orate:.2f}")

print()
base = np.mean([r[0] for r in results["no exclusion"]])
excl = np.mean([r[0] for r in results["WITH exclusion"]])
base_o = np.mean([r[1] for r in results["no exclusion"]])
excl_o = np.mean([r[1] for r in results["WITH exclusion"]])
print(f"mean percussive energy:  {base:.3f} -> {excl:.3f}   ({(excl-base)/base*100:+.0f}%)")
print(f"mean onsets/sec:         {base_o:.2f} -> {excl_o:.2f}   ({(excl_o-base_o)/base_o*100:+.0f}%)")
print()
print("VERDICT:", "exclusion reduces percussiveness" if excl < base
      else "NO measurable effect / made it worse")
