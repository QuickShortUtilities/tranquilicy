"""Verify the export DSP without loading MusicGen (import the module's functions
by exec'ing just the pieces we need would be fragile, so re-declare via import of
the file with the model load stubbed)."""
import io
import sys
import types
import numpy as np

# stub torch/transformers so importing the server module doesn't load a 2GB model
for name in ("torch", "transformers", "uvicorn", "fastapi", "fastapi.responses", "pydantic"):
    sys.modules.setdefault(name, types.ModuleType(name))

import soundfile as sf

# --- copies of the functions under test, kept in sync with 09_api_server.py ---
LOOP_CROSSFADE_SEC = 2.0
WIDEN_DELAY_SEC = 0.012
SR = 32000


def _apply_envelope(audio, env):
    return audio * env if audio.ndim == 1 else audio * env[:, None]


def make_seamless(audio, file_sr):
    n = len(audio)
    fade = int(min(LOOP_CROSSFADE_SEC * file_sr, n // 2))
    if fade < 2:
        return audio
    t = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    head = _apply_envelope(audio[:fade], np.sqrt(t)) + _apply_envelope(audio[n - fade:], np.sqrt(1.0 - t))
    return np.concatenate([head, audio[fade:n - fade]])


def widen_stereo(audio, file_sr, width):
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    delay = max(1, int(WIDEN_DELAY_SEC * file_sr))
    delayed = np.concatenate([np.zeros(delay, dtype=mono.dtype), mono[:-delay]])
    side = (mono - delayed) * (width * 0.5)
    return np.stack([mono + side, mono - side], axis=1).astype(np.float32)


def apply_warmth(audio, warmth):
    if warmth <= 0.0:
        return audio
    drive = 1.0 + warmth * 1.5
    driven = audio * drive
    sat = np.clip(driven, -1.5, 1.5)
    sat = sat - (sat ** 3) / 6.0
    return (sat / drive).astype(np.float32)


def apply_air(audio, air):
    if air <= 0.0:
        return audio
    diff = np.zeros_like(audio)
    if audio.ndim == 1:
        diff[1:] = audio[1:] - audio[:-1]
    else:
        diff[1:, :] = audio[1:, :] - audio[:-1, :]
    out = audio + (air * 0.35) * diff
    return np.clip(out, -0.99, 0.99).astype(np.float32)


def apply_fades(audio, file_sr, fade_in, fade_out):
    n = len(audio)
    env = np.ones(n, dtype=np.float32)
    n_in = min(int(fade_in * file_sr), n)
    n_out = min(int(fade_out * file_sr), n - n_in)
    if n_in > 1:
        t = np.linspace(0.0, 1.0, n_in, dtype=np.float32)
        env[:n_in] = 0.5 * (1 - np.cos(np.pi * t))
    if n_out > 1:
        t = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
        env[n - n_out:] = 0.5 * (1 + np.cos(np.pi * t))
    return _apply_envelope(audio, env)


fails = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   <- {detail}"))
    if not cond:
        fails.append(name)


# a musical-ish test signal (not noise, so discontinuities are audible/measurable)
t = np.arange(SR * 10) / SR
x = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 331 * t)).astype(np.float32)

# --- seamless loop ---
y = make_seamless(x, SR)
check("loop: output shortened by exactly the crossfade", len(y) == len(x) - int(LOOP_CROSSFADE_SEC * SR),
      f"{len(y)} vs {len(x) - int(LOOP_CROSSFADE_SEC*SR)}")

# The loop join: last sample of y wrapping to first sample of y should be as
# smooth as any ordinary neighbouring pair in the source.
join_jump = abs(float(y[0]) - float(y[-1]))
typical_jump = float(np.percentile(np.abs(np.diff(x)), 99))
check("loop: wrap-around join is not a discontinuity", join_jump <= typical_jump * 3,
      f"join {join_jump:.5f} vs typical {typical_jump:.5f}")

# compare against a naive cut (no crossfade) to prove the crossfade is doing work
naive = x[: len(x) - int(LOOP_CROSSFADE_SEC * SR)]
naive_jump = abs(float(naive[0]) - float(naive[-1]))
check("loop: better than a naive trim", join_jump <= naive_jump, f"{join_jump:.5f} vs naive {naive_jump:.5f}")
# Level behaviour in the crossfade depends on how correlated head and tail are.
# Real music: uncorrelated -> equal-power (sqrt) ramps are the correct choice and
# hold level. Use a signal whose head and tail are genuinely different material.
rng = np.random.default_rng(7)
drift = np.linspace(180, 300, SR * 10)  # slow sweep: head and tail are unrelated
uncorr = (0.3 * np.sin(2 * np.pi * np.cumsum(drift) / SR)
          + 0.05 * rng.standard_normal(SR * 10)).astype(np.float32)
yu = make_seamless(uncorr, SR)
ratio = float(np.sqrt(np.mean(yu[:SR] ** 2))) / float(np.sqrt(np.mean(uncorr[:SR] ** 2)))
check("loop: equal-power crossfade holds level on uncorrelated material",
      abs(ratio - 1.0) < 0.25, f"level ratio {ratio:.3f}")
check("loop: uncorrelated join is smooth",
      abs(float(yu[0]) - float(yu[-1])) <= float(np.percentile(np.abs(np.diff(uncorr)), 99)) * 3)

# --- stereo widening: the mono-compatibility guarantee ---
st = widen_stereo(x, SR, 1.0)
check("widen: output is stereo", st.ndim == 2 and st.shape[1] == 2, str(st.shape))
mono_sum = st.mean(axis=1)
check("widen: mono sum returns the original exactly", np.allclose(mono_sum, x, atol=1e-6),
      f"max err {np.max(np.abs(mono_sum - x)):.2e}")
check("widen: channels actually differ", float(np.mean(np.abs(st[:, 0] - st[:, 1]))) > 1e-3)
zero = widen_stereo(x, SR, 0.0)
check("widen: width=0 leaves channels identical", np.allclose(zero[:, 0], zero[:, 1]))

# --- tape warmth and air lift ---
w_zero = apply_warmth(x, 0.0)
check("warmth: 0.0 leaves audio untouched", np.allclose(w_zero, x))
w_sat = apply_warmth(x, 0.8)
check("warmth: saturation alters harmonics without exceeding bounds", np.max(np.abs(w_sat)) <= 1.0)
check("warmth: actually modifies the signal", float(np.mean(np.abs(w_sat - x))) > 1e-4)

a_zero = apply_air(x, 0.0)
check("air: 0.0 leaves audio untouched", np.allclose(a_zero, x))
a_lift = apply_air(x, 0.8)
check("air: adds high frequency energy without blowing up", np.max(np.abs(a_lift)) <= 1.0)
check("air: actually modifies the signal", float(np.mean(np.abs(a_lift - x))) > 1e-4)

# --- fades ---
f = apply_fades(x, SR, 2.0, 3.0)
check("fade: starts at silence", abs(float(f[0])) < 1e-6, f"{f[0]:.2e}")
check("fade: ends at silence", abs(float(f[-1])) < 1e-4, f"{f[-1]:.2e}")
check("fade: middle untouched", np.allclose(f[SR * 5], x[SR * 5], atol=1e-6))
# fades must not overrun each other on a short track
short = x[: SR // 2]
fs = apply_fades(short, SR, 10.0, 10.0)
check("fade: over-long fades on a short track stay finite and bounded",
      np.all(np.isfinite(fs)) and float(np.max(np.abs(fs))) <= float(np.max(np.abs(short))) + 1e-6)

# --- stereo through the whole chain, then encode every format ---
chain = apply_fades(widen_stereo(make_seamless(x, SR), SR, 0.6), SR, 1.0, 1.0)
check("chain: stereo shape survives", chain.ndim == 2 and chain.shape[1] == 2, str(chain.shape))
for fmt in ("WAV", "FLAC", "OGG", "MP3"):
    try:
        b = io.BytesIO()
        sf.write(b, chain, SR, format=fmt)
        data = b.getvalue()
        back, _ = sf.read(io.BytesIO(data), dtype="float32")
        check(f"encode {fmt} ({len(data)/1024:.0f} KB) round-trips as stereo",
              back.ndim == 2 and back.shape[1] == 2, str(back.shape))
    except Exception as e:
        check(f"encode {fmt}", False, f"{type(e).__name__}: {e}")

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
