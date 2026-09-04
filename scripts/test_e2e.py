"""End-to-end: drive the real server exactly as the browser does."""
import io
import json
import time
import urllib.request

import soundfile as sf

BASE = "http://127.0.0.1:8000"


def post(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def get_json(path):
    return json.loads(urllib.request.urlopen(BASE + path).read())


print("starting a short generation...")
job = post("/generate", {"prompt": "calm ambient pad, soft", "duration_sec": 6})["job_id"]

t0 = time.time()
while True:
    s = get_json(f"/status/{job}")
    if s.get("error"):
        raise SystemExit("generation failed: " + s["error"])
    if s["done"]:
        break
    if time.time() - t0 > 300:
        raise SystemExit("timed out")
    time.sleep(2)
print(f"generated in {time.time() - t0:.0f}s")

raw = urllib.request.urlopen(f"{BASE}/result/{job}").read()
base_audio, base_sr = sf.read(io.BytesIO(raw), dtype="float32")
print(f"raw: {base_audio.shape} @ {base_sr}Hz, {len(raw)/1024:.0f} KB, "
      f"{len(base_audio)/base_sr:.2f}s, {'mono' if base_audio.ndim == 1 else 'stereo'}")

fails = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   <- {detail}"))
    if not cond:
        fails.append(name)


def export(**params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    data = urllib.request.urlopen(f"{BASE}/master/{job}?{q}").read()
    audio, srate = sf.read(io.BytesIO(data), dtype="float32")
    return data, audio, srate


# plain WAV export, no processing
d, a, s = export(preset="off", fmt="WAV")
check("plain export matches source length", abs(len(a) - len(base_audio)) <= 1,
      f"{len(a)} vs {len(base_audio)}")

# fades
d, a, s = export(preset="off", fmt="WAV", fade_in=2, fade_out=2)
check("fade in reaches silence at the start", abs(float(a[0])) < 1e-5, f"{a[0]:.2e}")
check("fade out reaches silence at the end", abs(float(a[-1])) < 1e-4, f"{a[-1]:.2e}")

# seamless loop
d, a, s = export(preset="off", fmt="WAV", seamless="true")
expected = len(base_audio) - int(2.0 * base_sr)
check("seamless trims exactly the crossfade", abs(len(a) - expected) <= 1, f"{len(a)} vs {expected}")

# seamless must override fades rather than applying both
d2, a2, _ = export(preset="off", fmt="WAV", seamless="true", fade_in=3, fade_out=3)
check("seamless ignores fades (no silence at edges)", abs(float(a2[0])) > 1e-5 or abs(float(a2[-1])) > 1e-5,
      f"start {a2[0]:.2e} end {a2[-1]:.2e}")

# stereo widening
d, a, s = export(preset="off", fmt="WAV", width=0.8)
check("width produces stereo", a.ndim == 2 and a.shape[1] == 2, str(a.shape))
if a.ndim == 2:
    mono_sum = a.mean(axis=1)
    check("widened export still sums to the original mono",
          float(max(abs(mono_sum[:len(base_audio)] - base_audio[:len(mono_sum)]))) < 1e-4,
          "mono compatibility")

# loudness presets actually change level
levels = {}
for p in ("off", "gentle", "streaming", "loud"):
    _, a, _ = export(preset=p, fmt="WAV")
    levels[p] = float((a ** 2).mean() ** 0.5)
print("   RMS by preset:", {k: round(v, 4) for k, v in levels.items()})
check("loud is louder than gentle", levels["loud"] > levels["gentle"], str(levels))

# every advertised format, all options at once
page = urllib.request.urlopen(BASE + "/").read().decode()
for fmt in ("WAV", "FLAC", "OGG", "MP3"):
    if f'value="{fmt}"' not in page:
        print(f"   ({fmt} not offered, skipping)")
        continue
    d, a, s = export(preset="streaming", fmt=fmt, width=0.5, fade_in=1, fade_out=1)
    check(f"{fmt} full-chain export decodes ({len(d)/1024:.0f} KB)",
          a.size > 0 and a.ndim == 2, str(a.shape))

print()
print("FAILURES:", fails if fails else "none")
raise SystemExit(1 if fails else 0)
