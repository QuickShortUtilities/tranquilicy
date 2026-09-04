"""
Local web app for generating chillout music: loads MusicGen once, serves a
browser GUI styled to match the Tranquil Soul Music / Tranquilicy brand.
Supports durations beyond MusicGen's ~30s single-pass limit by chaining
continuation segments together, with a real (not simulated) progress bar
driven by a StoppingCriteria hook that fires on every generation step.

Usage:
    python 09_api_server.py
Then open http://localhost:8000 in a browser.
"""
import io
import threading
import time
import uuid

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForTextToWaveform, AutoProcessor, StoppingCriteria, StoppingCriteriaList

APP_VERSION = "1.7.0"
MODEL_ID = "facebook/musicgen-medium"
SEGMENT_NEW_TOKENS = 1000  # ~20s of new audio per segment, and MusicGen's rough single-pass ceiling
# MusicGen's delay pattern spends the first few steps of every segment filling the
# staggered codebooks, so a segment returns slightly LESS audio than
# max_new_tokens implies. Ask for a little extra, and never ask for a sliver:
# a continuation budgeted at a handful of tokens emits no new audio at all, which
# stalls the chaining loop instead of finishing the track.
TOKEN_HEADROOM = 12
MIN_SEGMENT_TOKENS = 60
FRAMES_PER_SEC = 50  # EnCodec frame rate -- tokens-per-second of audio
SEED_SEC = 3.0
MIN_DURATION, MAX_DURATION = 3.0, 300.0
JOB_TTL_SEC = 30 * 60  # prune finished jobs after this long so `jobs` doesn't grow forever

app = FastAPI()
jobs = {}  # job_id -> {progress, done, error, audio, created_at}
gen_lock = threading.Lock()  # serialize GPU access -- two concurrent generate() calls on
                              # one CUDA model/device is a real race condition, not just slow


def prune_old_jobs():
    cutoff = time.time() - JOB_TTL_SEC
    for jid in [j for j, v in jobs.items() if v["done"] and v["created_at"] < cutoff]:
        del jobs[jid]

print(f"Loading {MODEL_ID} ...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForTextToWaveform.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)
sr = model.config.audio_encoder.sampling_rate
seed_len = int(SEED_SEC * sr)
print("Model loaded, ready.")


# ---- Negative prompting -------------------------------------------------
# MusicGen's text encoder has no concept of negation: putting "no drums" in the
# prompt just feeds it the token "drums", which tends to make drums MORE likely.
# Classifier-free guidance, though, already runs a second "unconditional" branch
# that transformers fills with zeros (see _prepare_text_encoder_kwargs_for_
# generation). Swapping those zeros for an encoded negative prompt turns
# guidance into a force pushing away from those terms -- which is the only
# mechanism here that actually removes an instrument.
_negative_ctx = {"text": None}
_orig_prepare_text_kwargs = model._prepare_text_encoder_kwargs_for_generation


def _encode_negative(text: str, seq_len: int):
    """Encode the negative prompt, shaped to match the positive branch's length."""
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        hidden = model.text_encoder(input_ids=inputs.input_ids,
                                    attention_mask=inputs.attention_mask).last_hidden_state
    mask = inputs.attention_mask
    cur = hidden.shape[1]
    if cur < seq_len:  # pad: zeros + mask 0 is exactly how the null branch pads anyway
        pad = seq_len - cur
        hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad))
    elif cur > seq_len:
        hidden, mask = hidden[:, :seq_len], mask[:, :seq_len]
    return hidden, mask


def _prepare_text_with_negative(*args, **kwargs):
    model_kwargs = _orig_prepare_text_kwargs(*args, **kwargs)
    text = _negative_ctx.get("text")
    enc = model_kwargs.get("encoder_outputs")
    if not text or enc is None:
        return model_kwargs

    hidden = enc.last_hidden_state
    if hidden.shape[0] < 2 or hidden.shape[0] % 2 != 0:
        return model_kwargs  # guidance off -> there is no null branch to replace
    half = hidden.shape[0] // 2

    neg_hidden, neg_mask = _encode_negative(text, hidden.shape[1])
    hidden[half:] = neg_hidden.to(hidden.dtype).expand(half, -1, -1)
    attn = model_kwargs.get("attention_mask")
    if attn is not None and attn.shape[0] == 2 * half:
        # the null branch is masked out entirely; the negative branch must be
        # attended to, or the model never "sees" what it is being pushed away from
        attn[half:] = neg_mask.to(attn.dtype).expand(half, -1)
    return model_kwargs


model._prepare_text_encoder_kwargs_for_generation = _prepare_text_with_negative


class StepCounter(StoppingCriteria):
    """Fires on every generation step: reports real per-token progress back to
    the job's status, and stops generation early if the job has been cancelled
    (this is the only place a long generation can be interrupted -- once
    model.generate() is running, nothing else gets a look-in)."""

    def __init__(self, on_step, should_stop=None):
        self.on_step = on_step
        self.should_stop = should_stop
        self.steps = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.steps += 1
        self.on_step(self.steps)
        return bool(self.should_stop and self.should_stop())


class GenerateRequest(BaseModel):
    prompt: str
    duration_sec: float = 20.0
    guidance_scale: float = 3.0
    negative_prompt: str = ""
    seed: int | None = None


def run_job(job_id: str, prompt: str, duration_sec: float, guidance_scale: float,
            negative_prompt: str = "", seed: int | None = None):
    try:
        target_len = int(duration_sec * sr)
        approx_samples_per_segment = SEGMENT_NEW_TOKENS * (sr / FRAMES_PER_SEC)
        total_segments = max(1, int(np.ceil(target_len / approx_samples_per_segment)))

        full_audio = None
        completed_segments = 0
        # Safety valve: if a segment ever comes back shorter than requested (e.g. the model
        # stops early), the while-loop below would otherwise retry forever without making
        # progress. Cap total attempts well above the expected segment count.
        max_iterations = total_segments * 3 + 5

        with gen_lock:  # one generation at a time -- this model/GPU isn't safe for concurrent calls
            # set inside the lock: the patched encoder-prep reads this global, so it
            # must not be visible to any other generation
            # every run assigns this before generating (None when there's nothing to
            # exclude), so a value left behind by a failed run can never leak into
            # the next one -- no cleanup needed
            _negative_ctx["text"] = negative_prompt or None
            if seed is not None:
                torch.manual_seed(seed)  # reproducible takes, and paired A/B comparisons
            iterations = 0
            while full_audio is None or len(full_audio) < target_len:
                iterations += 1
                if iterations > max_iterations:
                    break  # keep the audio we have rather than discarding a good take
                # Only ask for as many tokens as this segment actually still needs, capped at
                # SEGMENT_NEW_TOKENS -- otherwise a short (e.g. 5s) request still pays for a
                # full ~20s generation internally and throws most of it away.
                remaining_samples = target_len - (len(full_audio) if full_audio is not None else 0)
                remaining_tokens = int(np.ceil(remaining_samples / (sr / FRAMES_PER_SEC))) + TOKEN_HEADROOM
                seg_max_tokens = max(MIN_SEGMENT_TOKENS, min(SEGMENT_NEW_TOKENS, remaining_tokens))

                def on_step(step, _seg=completed_segments, _max=seg_max_tokens):
                    overall = (_seg + min(step / _max, 1.0)) / total_segments
                    jobs[job_id]["progress"] = min(overall, 0.99)

                stopper = StepCounter(on_step, lambda: jobs.get(job_id, {}).get("cancelled", False))
                if full_audio is None:
                    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
                else:
                    seed = full_audio[-seed_len:]
                    inputs = processor(text=[prompt], audio=seed, sampling_rate=sr, padding=True, return_tensors="pt").to(device)
                    inputs["input_values"] = inputs["input_values"].half()

                out = model.generate(
                    **inputs, do_sample=True, guidance_scale=guidance_scale,
                    max_new_tokens=seg_max_tokens, stopping_criteria=StoppingCriteriaList([stopper]),
                )
                if jobs[job_id]["cancelled"]:
                    jobs[job_id]["error"] = "Cancelled"
                    jobs[job_id]["done"] = True
                    return

                seg = out[0, 0].detach().cpu().float().numpy()
                before = 0 if full_audio is None else len(full_audio)
                full_audio = seg if full_audio is None else np.concatenate([full_audio, seg[seed_len:]])
                completed_segments += 1
                if len(full_audio) <= before:
                    break  # this segment added nothing; another round would too

        if full_audio is None or len(full_audio) < sr * 0.5:
            raise RuntimeError("generation produced no usable audio")

        full_audio = full_audio[:target_len]
        buf = io.BytesIO()
        sf.write(buf, full_audio, sr, format="WAV")
        jobs[job_id]["audio"] = buf.getvalue()
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["done"] = True
    except Exception as e:
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["done"] = True


@app.post("/generate")
def generate(req: GenerateRequest):
    prune_old_jobs()
    duration_sec = max(MIN_DURATION, min(MAX_DURATION, req.duration_sec))
    guidance_scale = max(1.0, min(10.0, req.guidance_scale))

    prompt = req.prompt.strip()[:800]  # guard against pathologically long input; MusicGen's text
                                        # encoder has a fixed context window, so more just gets wasted
    negative_prompt = req.negative_prompt.strip()[:400]
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"progress": 0.0, "done": False, "error": None, "audio": None,
                    "cancelled": False, "created_at": time.time()}
    threading.Thread(target=run_job,
                     args=(job_id, prompt, duration_sec, guidance_scale, negative_prompt, req.seed),
                     daemon=True).start()
    return {"job_id": job_id}


@app.post("/cancel/{job_id}")
def cancel(job_id: str):
    """Flags the job; the running generation's StoppingCriteria picks this up on
    its next step and bails out, so the GPU is freed within a step or two rather
    than after the full remaining duration."""
    job = jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    job["cancelled"] = True
    return {"ok": True}


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {"progress": job["progress"], "done": job["done"], "error": job["error"]}


@app.get("/result/{job_id}")
def result(job_id: str):
    job = jobs.get(job_id)
    if job is None or job["audio"] is None:
        return JSONResponse({"error": "not ready"}, status_code=404)
    buf = io.BytesIO(job["audio"])
    return StreamingResponse(buf, media_type="audio/wav")


MASTER_PRESETS = {"off": None, "gentle": -16.0, "streaming": -14.0, "loud": -9.0}
LOOP_CROSSFADE_SEC = 2.0
WIDEN_DELAY_SEC = 0.012  # ~12ms: enough decorrelation to open the image, short enough to stay natural
MAX_FADE_SEC = 15.0

# Only offer what this libsndfile build can actually write (MP3 needs >= 1.1).
_FORMAT_MEDIA_TYPES = {"WAV": "audio/wav", "FLAC": "audio/flac", "OGG": "audio/ogg", "MP3": "audio/mpeg"}
_FORMAT_LABELS = {"WAV": "WAV — lossless, largest", "FLAC": "FLAC — lossless, ~40% smaller",
                  "OGG": "OGG Vorbis — small", "MP3": "MP3 — smallest, most portable"}
AVAILABLE_FORMATS = [f for f in ("WAV", "FLAC", "OGG", "MP3") if f in sf.available_formats()]


def _apply_envelope(audio: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Multiply by a per-sample gain envelope, for mono or (N, 2) stereo."""
    return audio * env if audio.ndim == 1 else audio * env[:, None]


def apply_mastering(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Gain-matches to a target RMS level and gently limits any peaks the gain
    pushes past full scale. This is a simple loudness match + soft limiter,
    not ITU-R BS.1770 LUFS measurement or real multiband mastering -- the GUI
    labels it accordingly."""
    rms = float(np.sqrt(np.mean(np.square(audio)))) + 1e-9
    current_dbfs = 20 * np.log10(rms)
    gain = 10 ** ((target_dbfs - current_dbfs) / 20)
    out = audio * gain
    limit = 0.98
    out = np.tanh(out / limit) * limit
    return out.astype(np.float32)


def make_seamless(audio: np.ndarray, file_sr: int) -> np.ndarray:
    """Crossfade the tail back over the head so the track loops with no seam.

    The last L samples are blended over the first L and then dropped, so playing
    the result on repeat runs ...x[N-L-1] -> x[N-L] continuously -- the join is
    sample-contiguous in the original audio rather than a hard cut. Equal-power
    (sqrt) ramps keep the crossfade region from dipping in level.
    """
    n = len(audio)
    fade = int(min(LOOP_CROSSFADE_SEC * file_sr, n // 2))
    if fade < 2:
        return audio
    t = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    head = _apply_envelope(audio[:fade], np.sqrt(t)) + _apply_envelope(audio[n - fade:], np.sqrt(1.0 - t))
    return np.concatenate([head, audio[fade:n - fade]])


def widen_stereo(audio: np.ndarray, file_sr: int, width: float) -> np.ndarray:
    """Mono -> stereo with a decorrelated side channel.

    Built as mid +/- side, so summing to mono cancels the side signal exactly and
    returns the original audio -- no comb filtering on mono playback, which a
    plain Haas delay would cause.
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    delay = max(1, int(WIDEN_DELAY_SEC * file_sr))
    delayed = np.concatenate([np.zeros(delay, dtype=mono.dtype), mono[:-delay]])
    side = (mono - delayed) * (width * 0.5)
    return np.stack([mono + side, mono - side], axis=1).astype(np.float32)


def apply_fades(audio: np.ndarray, file_sr: int, fade_in: float, fade_out: float) -> np.ndarray:
    """Raised-cosine fades -- smoother into and out of silence than a linear ramp."""
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


@app.get("/master/{job_id}")
def master(job_id: str, preset: str = "streaming", fade_in: float = 0.0, fade_out: float = 0.0,
           seamless: bool = False, width: float = 0.0, fmt: str = "WAV"):
    job = jobs.get(job_id)
    if job is None or job["audio"] is None:
        return JSONResponse({"error": "not ready"}, status_code=404)

    fmt = fmt.upper()
    if fmt not in AVAILABLE_FORMATS:
        fmt = "WAV"
    target_dbfs = MASTER_PRESETS.get(preset, MASTER_PRESETS["streaming"])
    fade_in = max(0.0, min(MAX_FADE_SEC, fade_in))
    fade_out = max(0.0, min(MAX_FADE_SEC, fade_out))
    width = max(0.0, min(1.0, width))

    audio, file_sr = sf.read(io.BytesIO(job["audio"]), dtype="float32")

    # Order matters: loop first (on the untouched audio), widen before limiting so
    # the limiter catches any peaks widening adds, and fade last so nothing after
    # it can lift the tails back off silence.
    if seamless:
        audio = make_seamless(audio, file_sr)
    if width > 0:
        audio = widen_stereo(audio, file_sr, width)
    if target_dbfs is not None:
        audio = apply_mastering(audio, target_dbfs)
    if not seamless:
        # a looping track must not fade -- that would reintroduce the seam
        audio = apply_fades(audio, file_sr, fade_in, fade_out)

    buf = io.BytesIO()
    sf.write(buf, audio, file_sr, format=fmt)
    buf.seek(0)
    return StreamingResponse(buf, media_type=_FORMAT_MEDIA_TYPES.get(fmt, "application/octet-stream"))


@app.get("/", response_class=HTMLResponse)
def index():
    # Placeholder substitution rather than an f-string: the page below is full of
    # literal CSS/JS braces, which an f-string would require doubling everywhere.
    format_options = "".join(
        f'<option value="{f}"{" selected" if f == "WAV" else ""}>{_FORMAT_LABELS[f]}</option>'
        for f in AVAILABLE_FORMATS
    )
    html = (
        INDEX_HTML
        .replace("__APP_VERSION__", APP_VERSION)
        .replace("__FORMAT_OPTIONS__", format_options)
    )
    # The page is edited and the server restarted constantly during development;
    # without this the browser can serve a cached copy and hide the new build.
    return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<title>Tranquilicy Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,400&family=Montserrat:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --gold: #C1A673; --gold-deep: #A58B58; --gold-text: #D4B97A; --gold-glow: rgba(193,166,115,.18); --on-gold: #fff;
    --bg: #090807; --bg-2: #100F0D; --bg-3: #181612;
    --text: #F2EFE9; --text-2: #9A9188; --text-3: #524D48;
    --line: rgba(255,255,255,.07); --line-soft: rgba(255,255,255,.045); --line-hi: rgba(255,255,255,.13);
    --display: "Cormorant Garamond", Georgia, serif;
    --ui: "Montserrat", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --max: 1360px; --radius: 20px; --radius-sm: 12px; --radius-xs: 8px;
    --lift: 0 1px 3px rgba(0,0,0,.5), 0 12px 32px rgba(0,0,0,.6);
    --ease: cubic-bezier(.22,.61,.36,1); --ease-out: cubic-bezier(.16,1,.3,1);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--ui); font-weight: 300; overflow-x: hidden; }
  #featherCanvas { position: fixed; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; }
  .wrap { position: relative; z-index: 1; max-width: var(--max); margin: 0 auto; padding: 72px 24px 60px; }
  .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 48px; }
  .brand .ring { width: 34px; height: 34px; border-radius: 50%; border: 1.5px solid var(--gold); position: relative; flex: none; }
  .brand .ring::after { content: ""; position: absolute; inset: 8px; border-radius: 50%; border: 1.5px solid var(--gold-glow); }
  .brand span { font-family: var(--ui); letter-spacing: .12em; text-transform: uppercase; font-size: 13px; color: var(--text-2); }
  h1 { font-family: var(--display); font-weight: 400; font-style: italic; font-size: 40px; margin: 0 0 8px; color: var(--text); }
  .sub { color: var(--text-2); font-size: 14px; margin-bottom: 40px; letter-spacing: .01em; }
  .card { background: var(--bg-2); border: 1px solid var(--line); border-radius: var(--radius); padding: 28px; box-shadow: var(--lift); }
  label { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .12em; color: var(--text-2); margin-bottom: 10px; }
  textarea { width: 100%; min-height: 90px; margin-top: 10px; background: var(--bg-3); color: var(--text); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 14px; font-family: var(--ui); font-size: 14px; font-weight: 300; resize: vertical; transition: border-color .2s var(--ease); }
  textarea:focus { outline: none; border-color: var(--gold-glow); }
  .row { display: flex; justify-content: space-between; align-items: center; margin-top: 26px; }
  .row .val { color: var(--gold-text); font-family: var(--display); font-size: 18px; font-style: italic; }
  input[type=range] { width: 100%; margin-top: 10px; accent-color: var(--gold); }
  .btn-primary {
    width: 100%; margin-top: 28px; padding: 15px; border: none; border-radius: var(--radius-sm);
    background: linear-gradient(180deg, var(--gold), var(--gold-deep)); color: var(--on-gold);
    font-family: var(--ui); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .14em;
    cursor: pointer; transition: transform .2s var(--ease-out), box-shadow .2s var(--ease-out);
    box-shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 20px rgba(193,166,115,.18);
  }
  .btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 2px 6px rgba(0,0,0,.5), 0 16px 32px rgba(193,166,115,.38); }
  .btn-primary:active:not(:disabled) { transform: scale(.975); }
  .btn-primary:disabled { opacity: .5; cursor: not-allowed; }
  .link-btn {
    display: inline-block; padding: 0; border: none; background: none;
    font-family: var(--ui); font-weight: 400; font-size: 11px; text-transform: uppercase;
    letter-spacing: .1em; color: var(--gold-text); text-decoration: none; cursor: pointer;
    transition: opacity .2s var(--ease);
  }
  .link-btn:hover { opacity: .7; }
  .link-btn:focus-visible { outline: 2px solid var(--gold-glow); outline-offset: 3px; border-radius: 2px; }
  #barOuter { background: var(--bg-3); border: 1px solid var(--line); border-radius: 999px; height: 8px; overflow: hidden; }
  #barInner { background: linear-gradient(90deg, var(--gold-deep), var(--gold)); height: 100%; width: 0%; transition: width .3s var(--ease-out); border-radius: 999px; }
  #errorText { color: #e08a8a; font-size: 12px; margin-top: 10px; display: none; }
  audio { width: 100%; margin-top: 20px; border-radius: var(--radius-xs); }
  audio::-webkit-media-controls-panel { background: var(--bg-3); }
  .btn-ghost {
    display: block; width: 100%; margin-top: 12px; padding: 13px; border: 1px solid var(--line-hi);
    border-radius: var(--radius-sm); background: transparent; color: var(--text-2);
    font-family: var(--ui); font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: .14em;
    text-align: center; text-decoration: none; cursor: pointer; box-sizing: border-box;
    transition: background .2s var(--ease), border-color .2s var(--ease), color .2s var(--ease);
  }
  .btn-ghost:hover { background: rgba(193,166,115,.06); border-color: var(--gold-glow); color: var(--gold-text); }

  .dials-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px 12px; margin: 22px 0 6px; }
  .dial-wrap { text-align: center; user-select: none; }
  .knob {
    width: 52px; height: 52px; border-radius: 50%; margin: 0 auto; position: relative; cursor: ns-resize;
    background: radial-gradient(circle at 35% 30%, var(--bg-3), var(--bg-2) 70%); border: 1px solid var(--line-hi);
    box-shadow: inset 0 1px 2px rgba(0,0,0,.5); touch-action: none;
  }
  .knob::before {
    content: ""; position: absolute; top: 5px; left: 50%; width: 2px; height: 14px; background: var(--gold);
    border-radius: 2px; transform-origin: 50% 21px; transform: translateX(-50%) rotate(var(--rot, -135deg));
    transition: transform .05s linear;
  }
  .knob:hover { border-color: var(--gold-glow); }
  .knob:focus-visible { outline: none; border-color: var(--gold); box-shadow: inset 0 1px 2px rgba(0,0,0,.5), 0 0 0 3px var(--gold-glow); }
  .knob.dragging { border-color: var(--gold); }
  .dial-label { font-size: 10px; text-transform: uppercase; letter-spacing: .1em; color: var(--text-2); margin-top: 9px; }
  .dial-value { font-family: var(--display); font-style: italic; color: var(--gold-text); font-size: 13px; margin-top: 2px; }

  .star-wrap { display: flex; justify-content: center; margin: 8px 0 20px; }
  /* cursor is set from JS per-hover (grab only when actually over a handle) --
     a blanket `cursor: grab` here would promise draggability across the whole chart */
  #starChart { touch-action: none; }
  /* while a drag is in progress the cursor must not flicker as the pointer
     passes over other elements, so the whole document takes the drag cursor */
  body.dragging-knob, body.dragging-knob * { cursor: ns-resize !important; }
  body.dragging-star, body.dragging-star * { cursor: grabbing !important; }

  /* a vertical list: three checkboxes across wrapped 1-then-2 and looked broken */
  .toggles { display: flex; flex-direction: column; gap: 11px; margin-top: 14px; }
  .toggle { display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--text-2); cursor: pointer; }
  .toggle input { accent-color: var(--gold); width: 15px; height: 15px; cursor: pointer; flex: none; }
  .toggle:hover { color: var(--text); }

  .section-label { font-size: 11px; text-transform: uppercase; letter-spacing: .12em; color: var(--text-2); margin: 24px 0 4px; border-top: 1px solid var(--line-soft); padding-top: 20px; }
  .section-label:first-of-type { border-top: none; padding-top: 0; margin-top: 0; }

  .layout { display: grid; grid-template-columns: 1fr 1fr 1.05fr; gap: 24px; align-items: stretch; }
  .layout .card { display: flex; flex-direction: column; }
  .layout .card-title { font-family: var(--display); font-style: italic; font-size: 20px; color: var(--text); margin: 0 0 4px; }
  .layout .card-hint { font-size: 12px; color: var(--text-2); margin: 0 0 24px; }
  @media (max-width: 1180px) {
    .layout { grid-template-columns: 1fr 1fr; }
    .layout > .card:first-child { grid-column: 1 / -1; }
  }
  @media (max-width: 720px) {
    .layout { grid-template-columns: 1fr; }
    .layout > .card:first-child { grid-column: auto; }
  }

  select, input[type=text] {
    width: 100%; background: var(--bg-3); color: var(--text); border: 1px solid var(--line);
    border-radius: var(--radius-sm); padding: 11px 12px; font-family: var(--ui); font-size: 13px;
    font-weight: 300; margin-top: 8px; transition: border-color .2s var(--ease);
  }
  select:focus, input[type=text]:focus { outline: none; border-color: var(--gold-glow); }
  /* some browsers render the dropdown list with system colours unless told otherwise */
  option { background: var(--bg-3); color: var(--text); }
  .field { margin-top: 22px; }
  .field:first-child { margin-top: 0; }
  .hint { font-size: 11px; color: var(--text-3); margin-top: 6px; line-height: 1.5; }
  .card.disabled { opacity: .4; pointer-events: none; }
  #exportCanvas {
    display: block; margin: 14px auto 0; max-width: 100%; max-height: 300px; width: auto; height: auto;
    border-radius: var(--radius-sm); border: 1px solid var(--line); background: #090807;
  }
  .footer {
    margin-top: 48px; text-align: center; font-size: 10px; letter-spacing: .12em;
    text-transform: uppercase; color: var(--text-3);
  }
  .footer .sep { opacity: .5; margin: 0 6px; }

  /* ---- Step rail: the three columns read as one numbered flow ---- */
  .rail { display: flex; align-items: center; margin: 0 0 38px; }
  .rail-node { display: flex; align-items: center; gap: 11px; flex: none; }
  .rail-dot {
    width: 30px; height: 30px; border-radius: 50%; border: 1px solid var(--line-hi);
    display: grid; place-items: center; font-size: 10px; letter-spacing: .06em;
    color: var(--text-3); flex: none;
    transition: color .5s var(--ease), border-color .5s var(--ease), background .5s var(--ease);
  }
  .rail-label { font-size: 10px; text-transform: uppercase; letter-spacing: .14em; color: var(--text-3); transition: color .5s var(--ease); white-space: nowrap; }
  .rail-line { flex: 1 1 auto; height: 1px; background: var(--line); margin: 0 16px; position: relative; }
  .rail-line::after {
    content: ""; position: absolute; inset: 0; border-radius: 1px;
    background: linear-gradient(90deg, var(--gold-deep), var(--gold));
    transform: scaleX(0); transform-origin: left; transition: transform .9s var(--ease-out);
  }
  .rail-line.filled::after { transform: scaleX(1); }
  .rail-line.delayed::after { transition-delay: .25s; }
  .rail-node.active .rail-dot { border-color: var(--gold); color: var(--gold-text); animation: haloPulse 2.6s var(--ease) infinite; }
  .rail-node.active .rail-label { color: var(--text-2); }
  .rail-node.done .rail-dot { background: linear-gradient(180deg, var(--gold), var(--gold-deep)); border-color: transparent; color: var(--on-gold); }
  .rail-node.done .rail-label { color: var(--text-2); }
  @keyframes haloPulse {
    0%   { box-shadow: 0 0 0 0 rgba(193,166,115,.34); }
    70%  { box-shadow: 0 0 0 11px rgba(193,166,115,0); }
    100% { box-shadow: 0 0 0 0 rgba(193,166,115,0); }
  }
  @media (max-width: 720px) { .rail-label { display: none; } .rail-line { margin: 0 8px; } }

  /* ---- Card step states ---- */
  /* position:relative anchors the unlock sweep overlay */
  .card { position: relative; transition: opacity .6s var(--ease), border-color .6s var(--ease); }
  .card-head { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
  .step-badge {
    width: 27px; height: 27px; border-radius: 50%; border: 1px solid var(--line-hi);
    display: grid; place-items: center; font-size: 10px; color: var(--text-3); flex: none;
    transition: color .5s var(--ease), border-color .5s var(--ease), background .5s var(--ease);
  }
  .card.is-active .step-badge { border-color: var(--gold); color: var(--gold-text); }
  .card.is-done .step-badge { background: linear-gradient(180deg, var(--gold), var(--gold-deep)); border-color: transparent; color: var(--on-gold); }
  .card.is-active { border-color: var(--gold-glow); }

  /* the moment a step unlocks: rise into place with a single gold sweep */
  .card.unlocking { animation: unlockRise .75s var(--ease-out) both; overflow: hidden; }
  .card.unlocking::after {
    content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
    background: linear-gradient(105deg, transparent 32%, rgba(193,166,115,.13) 50%, transparent 68%);
    transform: translateX(-100%); animation: unlockSweep 1.15s var(--ease-out) .12s forwards;
  }
  @keyframes unlockRise { from { opacity: .35; transform: translateY(12px); } to { opacity: 1; transform: none; } }
  @keyframes unlockSweep { to { transform: translateX(100%); } }

  /* Each card's closing action group is pinned to the bottom, so all three
     columns terminate on the same baseline instead of leaving ragged
     whitespace of three different heights below their buttons. */
  .card-actions { margin-top: auto; }

  /* ---- Output bar: one continuous strip under the three columns, holding
     every result (progress, player) and every download. ---- */
  .outbar { margin-top: 24px; display: flex; align-items: center; gap: 30px; padding: 22px 26px; }
  .outbar-main { flex: 1 1 auto; min-width: 0; }
  .outbar-line { display: flex; align-items: center; gap: 12px; font-size: 12px; color: var(--text-2); letter-spacing: .02em; margin-bottom: 12px; }
  #statusLabel { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #statusPct { color: var(--gold-text); font-family: var(--display); font-style: italic; font-size: 15px; flex: none; }
  @media (max-width: 900px) { .outbar { flex-direction: column; align-items: stretch; gap: 20px; } }

  .eq { display: flex; align-items: flex-end; gap: 3px; height: 18px; flex: none; }
  .eq i { width: 2px; height: 30%; border-radius: 2px; background: var(--text-3); animation: eqIdle 3.2s var(--ease) infinite; }
  .eq i:nth-child(2) { animation-delay: .35s; }
  .eq i:nth-child(3) { animation-delay: .7s; }
  .eq i:nth-child(4) { animation-delay: 1.05s; }
  .eq i:nth-child(5) { animation-delay: 1.4s; }
  @keyframes eqIdle { 0%, 100% { height: 22%; } 50% { height: 72%; } }
  .outbar.busy .eq i { background: var(--gold); animation-duration: .95s; }

  #playerWrap { opacity: 0; transition: opacity .55s var(--ease); }
  #playerWrap.ready { opacity: 1; }
  #playerWrap audio { margin-top: 14px; }

  /* download chips: dim until their artefact actually exists */
  .chips { display: flex; gap: 10px; flex: none; flex-wrap: wrap; }
  .chip {
    display: inline-flex; align-items: center; gap: 9px; padding: 11px 17px;
    border: 1px solid var(--line-hi); border-radius: 999px; background: transparent;
    color: var(--text-2); font-family: var(--ui); font-weight: 500; font-size: 11px;
    text-transform: uppercase; letter-spacing: .12em; text-decoration: none; cursor: pointer;
    white-space: nowrap;
    transition: background .2s var(--ease), border-color .2s var(--ease), color .2s var(--ease), opacity .4s var(--ease);
  }
  .chip:hover { background: rgba(193,166,115,.06); border-color: var(--gold-glow); color: var(--gold-text); }
  .chip .dot { width: 5px; height: 5px; border-radius: 50%; background: var(--gold); flex: none; transition: opacity .4s var(--ease); }
  .chip.empty { opacity: .26; pointer-events: none; }
  .chip.empty .dot { opacity: 0; }

  @media (prefers-reduced-motion: reduce) {
    .card.unlocking, .card.unlocking::after, .rail-node.active .rail-dot, .eq i { animation: none; }
    .rail-line::after { transition: none; }
  }
</style>
</head>
<body>
<canvas id="featherCanvas"></canvas>
<div class="wrap">
  <div class="brand"><div class="ring"></div><span>Tranquil Soul Music</span></div>
  <h1>Create a track</h1>
  <div class="sub">Generate, then master and export it — three steps, one flow.</div>

  <div class="rail">
    <div class="rail-node" id="railStep1"><span class="rail-dot">01</span><span class="rail-label">Generate</span></div>
    <div class="rail-line" id="railLine1"></div>
    <div class="rail-node" id="railStep2"><span class="rail-dot">02</span><span class="rail-label">Master</span></div>
    <div class="rail-line delayed" id="railLine2"></div>
    <div class="rail-node" id="railStep3"><span class="rail-dot">03</span><span class="rail-label">Video</span></div>
  </div>

  <div class="layout">
  <div class="card" id="generateCard">
    <div class="card-head">
      <span class="step-badge">01</span>
      <div class="card-title">Generate</div>
    </div>
    <div class="card-hint">Describe the mood, tempo and instrumentation — Tranquilicy will render it.</div>
    <div class="row">
      <label style="margin:0">Prompt</label>
      <button type="button" class="link-btn" id="randomiseBtn" onclick="randomisePrompt()">⟳ Randomise</button>
    </div>
    <textarea id="prompt" placeholder="e.g. vinyl crackle, rain on window, specific instrument... or hit Randomise"></textarea>

    <div class="section-label">Six dials, shape the sound</div>
    <div class="dials-grid" id="dialsGrid"></div>
    <div class="star-wrap">
      <svg id="starChart" width="180" height="180" viewBox="0 0 180 180"></svg>
    </div>

    <div class="section-label">Exclude</div>
    <div class="toggles">
      <label class="toggle"><input type="checkbox" id="noDrums"> No drums / percussion</label>
      <label class="toggle"><input type="checkbox" id="noVocals"> No vocals</label>
      <label class="toggle"><input type="checkbox" id="noBass"> No bass</label>
    </div>
    <div class="hint">Steered away via the guidance model's negative branch — strong, but not absolute.</div>

    <div class="row">
      <label style="margin:0">Duration</label>
      <span class="val"><span id="durVal">20</span>s</span>
    </div>
    <input type="range" id="duration" min="5" max="180" value="20">

    <div class="card-actions">
      <button id="genBtn" class="btn-primary" onclick="generate()">Generate</button>
    </div>
  </div>

  <div class="card disabled" id="audioCard" inert aria-hidden="true">
    <div class="card-head">
      <span class="step-badge">02</span>
      <div class="card-title">Master</div>
    </div>
    <div class="card-hint">Name it, shape it, and export a finished mix.</div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Track title</label>
        <button type="button" class="link-btn" onclick="rerollTitle()">🎲 Re-roll</button>
      </div>
      <input type="text" id="trackTitle" placeholder="Untitled Chillout Track">
    </div>

    <div class="field">
      <label>Filename style</label>
      <select id="namePattern">
        <option value="plain">Title only</option>
        <option value="numbered-dash">01 - Title</option>
        <option value="numbered-dot">01. Title</option>
        <option value="extended">Title (Extended Mix)</option>
        <option value="instrumental">Title (Instrumental Mix)</option>
        <option value="slowed">Title [Slowed + Reverb]</option>
      </select>
    </div>

    <div class="section-label">Master</div>
    <div class="field">
      <label>Loudness</label>
      <select id="masterPreset">
        <option value="off">Off — raw generation</option>
        <option value="gentle">Gentle</option>
        <option value="streaming" selected>Streaming-ready</option>
        <option value="loud">Loud</option>
      </select>
      <div class="hint">Level-matches and limits peaks. Not studio-grade LUFS mastering.</div>
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Stereo width</label>
        <span class="val"><span id="widthVal">0</span></span>
      </div>
      <input type="range" id="stereoWidth" min="0" max="100" value="0">
      <div class="hint">Mono source, widened mono-safely.</div>
    </div>

    <div class="section-label">Shape</div>
    <label class="toggle" style="margin-top:14px">
      <input type="checkbox" id="seamlessLoop"> Seamless loop
    </label>
    <div class="hint">Repeats with no audible join. Trims 2s and replaces fades.</div>

    <div id="fadeFields">
      <div class="row">
        <label style="margin:0">Fade in</label>
        <span class="val"><span id="fadeInVal">0</span>s</span>
      </div>
      <input type="range" id="fadeIn" min="0" max="15" step="0.5" value="0">
      <div class="row">
        <label style="margin:0">Fade out</label>
        <span class="val"><span id="fadeOutVal">0</span>s</span>
      </div>
      <input type="range" id="fadeOut" min="0" max="15" step="0.5" value="0">
    </div>

    <div class="card-actions">
      <div class="section-label">Export</div>
      <div class="field">
        <label>Format</label>
        <select id="exportFormat">__FORMAT_OPTIONS__</select>
      </div>
      <button class="btn-primary" id="masterBtn" onclick="downloadMastered()">Master audio</button>
      <div id="masterStatus" class="hint" style="display:none; text-align:center;"></div>
    </div>
  </div>

  <div class="card disabled" id="videoCard" inert aria-hidden="true">
    <div class="card-head">
      <span class="step-badge">03</span>
      <div class="card-title">Video</div>
    </div>
    <div class="card-hint">Render a waveform video sized for wherever you're posting it.</div>

    <div class="field">
      <label>Waveform style</label>
      <select id="vidStyle">
        <option value="bars">Bars</option>
        <option value="wave" selected>Wave</option>
        <option value="pulse">Pulse</option>
        <option value="off">Off</option>
      </select>
    </div>
    <div class="field">
      <label>Palette</label>
      <select id="vidPalette">
        <option value="gold">Gold</option>
        <option value="ember">Ember</option>
        <option value="moonlit">Moonlit</option>
      </select>
    </div>
    <div class="field">
      <label>Backdrop</label>
      <select id="vidBackdrop">
        <option value="feathers">Feather drift</option>
        <option value="bloom">Gold bloom</option>
        <option value="minimal">Minimal dark</option>
      </select>
    </div>
    <div class="field">
      <label>Aspect ratio</label>
      <select id="vidAspect">
        <option value="9:16">9:16 — Reels / TikTok / Shorts</option>
        <option value="1:1">1:1 — Square</option>
        <option value="16:9">16:9 — YouTube / desktop</option>
      </select>
    </div>
    <div class="field">
      <label>Watermark</label>
      <select id="vidWatermark">
        <option value="wordmark">Tranquilicy wordmark</option>
        <option value="ring">Ring icon only</option>
        <option value="none">None</option>
      </select>
    </div>
    <div class="field">
      <label>Length</label>
      <select id="vidLength">
        <option value="full">Full track</option>
        <option value="loop15">15s social loop</option>
      </select>
    </div>

    <canvas id="exportCanvas"></canvas>

    <div class="card-actions">
      <button class="btn-primary" id="renderVidBtn" onclick="renderVideo()">Render video</button>
      <div id="vidStatus" class="hint" style="display:none; text-align:center; margin-top:10px;"></div>
      <button class="btn-ghost" id="stillBtn" onclick="saveStillFrame()">Capture still frame</button>
    </div>
  </div>
  </div>

  <div class="card outbar" id="outbar">
    <div class="outbar-main">
      <div class="outbar-line">
        <span class="eq"><i></i><i></i><i></i><i></i><i></i></span>
        <span id="statusLabel">No track yet — hit Generate</span>
        <button type="button" class="link-btn" id="cancelBtn" style="display:none" onclick="cancelGeneration()">Cancel</button>
        <span id="statusPct"></span>
      </div>
      <div id="barOuter"><div id="barInner"></div></div>
      <div id="playerWrap" hidden><audio id="player" controls></audio></div>
      <div id="errorText"></div>
    </div>
    <div class="chips">
      <a id="downloadBtn" class="chip empty" download><span class="dot"></span>WAV</a>
      <a id="masterChip" class="chip empty" download><span class="dot"></span><span id="masterChipLabel">Master</span></a>
      <a id="downloadVideoBtn" class="chip empty" download><span class="dot"></span><span id="videoChipLabel">Video</span></a>
      <a id="stillChip" class="chip empty" download><span class="dot"></span>Still</a>
    </div>
  </div>

  <div class="footer">
    Tranquil Soul Music <span class="sep">·</span> Tranquilicy Studio <span class="sep">·</span> v__APP_VERSION__
  </div>
</div>

<script>
document.getElementById('duration').oninput = e => document.getElementById('durVal').textContent = e.target.value;

let lastAudioUrl = null;  // each generate() call creates a new blob URL; without revoking the
                           // previous one, every generation leaks the last audio buffer from memory
let lastVideoUrl = null;
let lastMasterUrl = null;
let lastStillUrl = null;
let currentJobId = null;

// Every download lives as a chip in the output bar; a chip stays dimmed and
// unclickable until the thing it points at actually exists.
function setChip(id, url, filename, label) {
  const chip = document.getElementById(id);
  if (!url) {
    chip.classList.add('empty');
    chip.removeAttribute('href');
    return;
  }
  chip.href = url;
  chip.download = filename;
  chip.classList.remove('empty');
  if (label) {
    const el = chip.querySelector('span:not(.dot)');
    if (el) el.textContent = label;
  }
}

// ---- Shared feather-shape drawing (used by the page background AND the
// export video's "Feather drift" backdrop, so both stay visually identical) ----
function drawFeatherShape(ctx, size, colorTop, colorBottom, strokeStyle) {
  const hgt = size * 2.6, wid = size;
  const grad = ctx.createLinearGradient(0, -hgt / 2, 0, hgt / 2);
  grad.addColorStop(0, colorTop);
  grad.addColorStop(1, colorBottom);
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(0, -hgt / 2);
  ctx.bezierCurveTo(wid / 2, -hgt / 4, wid / 2, hgt / 4, 0, hgt / 2);
  ctx.bezierCurveTo(-wid / 2, hgt / 4, -wid / 2, -hgt / 4, 0, -hgt / 2);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = 0.6;
  ctx.beginPath();
  ctx.moveTo(0, -hgt / 2);
  ctx.lineTo(0, hgt / 2);
  ctx.stroke();
}

// ---- Falling gold feathers background ----
(function () {
  const canvas = document.getElementById('featherCanvas');
  const ctx = canvas.getContext('2d');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const GOLD = '#C1A673', GOLD_LIGHT = '#D4B97A';
  let dpr = Math.min(window.devicePixelRatio || 1, 2);
  let w = 0, h = 0, particles = [];

  function featherCount() {
    // fewer particles on small/mobile viewports to keep this cheap
    return Math.round(Math.max(14, Math.min(34, (window.innerWidth * window.innerHeight) / 45000)));
  }

  function makeParticle(spawnAbove) {
    const size = 8 + Math.random() * 10;
    return {
      x: Math.random() * w,
      y: spawnAbove ? -20 - Math.random() * h : Math.random() * h,
      size,
      speedY: 10 + Math.random() * 14,          // px/sec
      swayAmp: 18 + Math.random() * 26,
      swayFreq: 0.15 + Math.random() * 0.25,     // Hz
      phase: Math.random() * Math.PI * 2,
      baseX: 0,
      rot: Math.random() * Math.PI * 2,
      rotSpeed: (Math.random() - 0.5) * 0.8,     // rad/sec
      opacity: 0.10 + Math.random() * 0.22,
    };
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const count = featherCount();
    if (particles.length === 0) {
      particles = Array.from({ length: count }, () => makeParticle(false));
      particles.forEach(p => { p.baseX = p.x; });
    } else if (particles.length < count) {
      const extra = Array.from({ length: count - particles.length }, () => makeParticle(true));
      extra.forEach(p => { p.baseX = p.x; });
      particles = particles.concat(extra);
    } else {
      particles = particles.slice(0, count);
    }
  }

  function drawFeather(p) {
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rot);
    ctx.globalAlpha = p.opacity;
    drawFeatherShape(ctx, p.size, GOLD_LIGHT, GOLD, 'rgba(9,8,7,.25)');
    ctx.restore();
  }

  let lastT = null;
  function frame(t) {
    if (lastT === null) lastT = t;
    const dt = Math.min((t - lastT) / 1000, 0.05);
    lastT = t;
    ctx.clearRect(0, 0, w, h);
    for (const p of particles) {
      p.y += p.speedY * dt;
      p.rot += p.rotSpeed * dt;
      if (p.y - p.size * 1.3 > h) {
        p.y = -p.size * 1.3;
        p.baseX = Math.random() * w;
        p.phase = Math.random() * Math.PI * 2;
      }
      p.x = p.baseX + Math.sin(t / 1000 * p.swayFreq * Math.PI * 2 + p.phase) * p.swayAmp;
      drawFeather(p);
    }
    requestAnimationFrame(frame);
  }

  window.addEventListener('resize', resize);
  resize();
  if (!reduceMotion) {
    requestAnimationFrame(frame);
  } else {
    // static single frame respecting the user's reduced-motion preference
    ctx.clearRect(0, 0, w, h);
    particles.forEach(p => { p.x = p.baseX; drawFeather(p); });
  }
})();

// ---- Six dials + star chart ----
const DIALS = [
  { key: 'atmosphere', label: 'Atmosphere', default: 60, describe: v => v < 33 ? 'minimal ambience' : v < 66 ? 'gentle atmosphere' : 'deeply atmospheric, immersive' },
  { key: 'tempo',      label: 'Tempo',      default: 40, describe: v => Math.round(60 + v / 100 * 80) + ' bpm' },
  { key: 'warmth',     label: 'Warmth',     default: 65, describe: v => v < 33 ? 'cool, clean tone' : v < 66 ? 'warm tone' : 'deeply warm, analog tone' },
  { key: 'bass',       label: 'Bass',       default: 40, describe: v => v < 33 ? 'light bass' : v < 66 ? 'moderate bass' : 'deep sub bass' },
  { key: 'melody',     label: 'Melody',     default: 55, describe: v => v < 33 ? 'drone-like, minimal melody' : v < 66 ? 'gentle melodic lead' : 'strong melodic lead' },
  { key: 'rhythm',     label: 'Rhythm',     default: 30, describe: v => v < 33 ? 'free-form, no strong rhythm' : v < 66 ? 'soft rhythmic pulse' : 'steady rhythmic groove' },
];
const knobs = {};

function setupKnob(el, valueEl, initial, onChange) {
  // NOTE: render() only touches this knob's own display -- it must NOT call
  // onChange() during initial setup. onChange triggers drawStarChart(), which
  // reads every dial's value; during buildDials()'s first iteration the other
  // five knobs don't exist in `knobs` yet, so calling onChange here throws and
  // silently aborts the rest of the dial-building loop (this was the actual
  // bug behind "only one dial rendered, rest of the card is blank").
  let value = initial;
  const render = () => {
    el.style.setProperty('--rot', (-135 + (value / 100) * 270) + 'deg');
    valueEl.textContent = Math.round(value);
    el.setAttribute('aria-valuenow', Math.round(value));
  };
  render();

  let dragging = false, startY = 0, startVal = value;
  const endDrag = () => {
    if (!dragging) return;
    dragging = false;
    el.classList.remove('dragging');
    document.body.classList.remove('dragging-knob');
  };
  el.addEventListener('pointerdown', e => {
    dragging = true;
    startY = e.clientY;
    startVal = value;
    el.setPointerCapture(e.pointerId);
    el.classList.add('dragging');
    document.body.classList.add('dragging-knob');
    el.focus();
  });
  el.addEventListener('pointermove', e => {
    if (!dragging) return;
    value = Math.max(0, Math.min(100, startVal + (startY - e.clientY) * 0.6));
    render();
    onChange(value);
  });
  el.addEventListener('pointerup', endDrag);
  el.addEventListener('pointercancel', endDrag);
  el.addEventListener('lostpointercapture', endDrag);

  // keyboard control: arrows nudge, shift+arrows jump, home/end snap to the ends
  el.addEventListener('keydown', e => {
    const step = e.shiftKey ? 10 : 2;
    let next = value;
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') next = value + step;
    else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') next = value - step;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = 100;
    else return;
    e.preventDefault();
    value = Math.max(0, Math.min(100, next));
    render();
    onChange(value);
  });

  return {
    get value() { return value; },
    // lets the star chart drive a dial directly (dragging its vertex) without
    // duplicating the knob's rotation/label rendering logic
    set value(v) { value = Math.max(0, Math.min(100, v)); render(); },
  };
}

function buildDials() {
  const grid = document.getElementById('dialsGrid');
  DIALS.forEach(d => {
    const wrap = document.createElement('div');
    wrap.className = 'dial-wrap';
    wrap.innerHTML = `<div class="knob" id="knob-${d.key}" tabindex="0" role="slider"
        aria-label="${d.label}" aria-valuemin="0" aria-valuemax="100"></div>
      <div class="dial-label">${d.label}</div>
      <div class="dial-value" id="val-${d.key}"></div>`;
    grid.appendChild(wrap);
    const knobEl = wrap.querySelector('#knob-' + d.key);
    const valEl = wrap.querySelector('#val-' + d.key);
    knobs[d.key] = setupKnob(knobEl, valEl, d.default, () => drawStarChart());
  });
}

function drawStarChart() {
  const svg = document.getElementById('starChart');
  const cx = 90, cy = 90, maxR = 68;
  const n = DIALS.length;
  const angleFor = i => -Math.PI / 2 + i * (2 * Math.PI / n);
  const pt = (i, frac) => {
    const a = angleFor(i);
    return [cx + Math.cos(a) * maxR * frac, cy + Math.sin(a) * maxR * frac];
  };
  let svgHtml = '';
  // background rings
  [0.33, 0.66, 1].forEach(frac => {
    const ring = DIALS.map((_, i) => pt(i, frac).join(',')).join(' ');
    svgHtml += `<polygon points="${ring}" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="1"/>`;
  });
  // axis lines
  DIALS.forEach((_, i) => {
    const [x, y] = pt(i, 1);
    svgHtml += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="rgba(255,255,255,.07)" stroke-width="1"/>`;
  });
  // value polygon (the "star")
  const valuePts = DIALS.map((d, i) => pt(i, knobs[d.key].value / 100).join(',')).join(' ');
  svgHtml += `<polygon points="${valuePts}" fill="rgba(193,166,115,.16)" stroke="#C1A673" stroke-width="1.5"/>`;
  DIALS.forEach((d, i) => {
    const [x, y] = pt(i, knobs[d.key].value / 100);
    const active = draggingDialKey === d.key || hoverDialKey === d.key;
    if (active) {
      svgHtml += `<circle cx="${x}" cy="${y}" r="10" fill="rgba(193,166,115,.18)"/>`;
    }
    svgHtml += `<circle cx="${x}" cy="${y}" r="${active ? 5.5 : 4}" fill="#D4B97A" stroke="#090807" stroke-width="1"/>`;
  });
  // axis labels
  DIALS.forEach((d, i) => {
    const [x, y] = pt(i, 1.22);
    svgHtml += `<text x="${x}" y="${y}" fill="#9A9188" font-size="8" font-family="Montserrat, sans-serif" text-anchor="middle" dominant-baseline="middle">${d.label}</text>`;
  });
  svg.innerHTML = svgHtml;
}

// ---- Star chart drag: grab a vertex directly instead of only the round knobs ----
let draggingDialKey = null, hoverDialKey = null;
function setupStarChartDrag() {
  const svg = document.getElementById('starChart');
  const cx = 90, cy = 90, maxR = 68;
  const n = DIALS.length;
  const angleFor = i => -Math.PI / 2 + i * (2 * Math.PI / n);

  function svgPoint(e) {
    const rect = svg.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (180 / rect.width),
      y: (e.clientY - rect.top) * (180 / rect.height),
    };
  }

  function nearestVertex(p) {
    let best = null, bestDist = Infinity;
    DIALS.forEach((d, i) => {
      const a = angleFor(i);
      const frac = knobs[d.key].value / 100;
      const vx = cx + Math.cos(a) * maxR * frac, vy = cy + Math.sin(a) * maxR * frac;
      const dist = Math.hypot(p.x - vx, p.y - vy);
      if (dist < bestDist) { bestDist = dist; best = { key: d.key, i, dist }; }
    });
    return best;
  }

  function valueFromPointer(p, i) {
    const a = angleFor(i);
    const proj = (p.x - cx) * Math.cos(a) + (p.y - cy) * Math.sin(a);  // project onto this dial's axis
    return Math.max(0, Math.min(1, proj / maxR)) * 100;
  }

  const GRAB_RADIUS = 16;

  svg.addEventListener('pointerdown', e => {
    const p = svgPoint(e);
    const nearest = nearestVertex(p);
    if (!nearest || nearest.dist > GRAB_RADIUS) return;  // must grab close to an actual vertex
    draggingDialKey = nearest.key;
    svg.setPointerCapture(e.pointerId);
    document.body.classList.add('dragging-star');
    knobs[nearest.key].value = valueFromPointer(p, nearest.i);
    drawStarChart();
    e.preventDefault();
  });

  svg.addEventListener('pointermove', e => {
    const p = svgPoint(e);
    if (!draggingDialKey) {
      // hover feedback: only promise "grab" when the pointer is actually over a
      // handle, since anywhere else on the chart does nothing when you press
      const nearest = nearestVertex(p);
      const overHandle = nearest && nearest.dist <= GRAB_RADIUS;
      if (hoverDialKey !== (overHandle ? nearest.key : null)) {
        hoverDialKey = overHandle ? nearest.key : null;
        drawStarChart();
      }
      svg.style.cursor = overHandle ? 'grab' : 'default';
      return;
    }
    const i = DIALS.findIndex(d => d.key === draggingDialKey);
    knobs[draggingDialKey].value = valueFromPointer(p, i);
    drawStarChart();
  });

  svg.addEventListener('pointerleave', () => {
    if (hoverDialKey !== null) { hoverDialKey = null; drawStarChart(); }
  });

  const stop = () => {
    if (!draggingDialKey) return;
    draggingDialKey = null;
    document.body.classList.remove('dragging-star');
    drawStarChart();
  };
  svg.addEventListener('pointerup', stop);
  svg.addEventListener('pointercancel', stop);
  svg.addEventListener('lostpointercapture', stop);
}

buildDials();
drawStarChart();
setupStarChartDrag();

// ---- Randomise: combines word banks for a huge variety of chillout prompts ----
const BANKS = {
  scene: ['rainy city window', 'quiet forest cabin', 'empty beach at dawn', 'late-night study room',
    'mountain cabin in winter', 'desert night sky', 'abandoned greenhouse', 'rooftop at sunset',
    'misty harbor', 'old bookstore', 'train window at dusk', 'candlelit room', 'snowfall outside a cafe',
    'riverside dock', 'attic with dusty sunlight', 'empty subway platform', 'lighthouse at low tide',
    'greenhouse in the rain', 'observatory at midnight', 'porch during a thunderstorm'],
  texture: ['warm rhodes chords', 'soft analog pads', 'vinyl crackle', 'gentle piano', 'muted trumpet',
    'acoustic guitar harmonics', 'field recordings of rain', 'tape hiss', 'felt piano', 'glockenspiel',
    'cello drone', 'music box melody', 'granular synth textures', 'soft flute', 'detuned synth strings',
    'hand percussion', 'wind chimes', 'distant thunder', 'shortwave radio static', 'bowed vibraphone',
    'warm sub bass', 'nylon guitar', 'mellotron strings', 'soft marimba'],
  mood: ['nostalgic', 'weightless', 'hazy', 'introspective', 'tender', 'wistful', 'serene', 'dreamy',
    'melancholic but warm', 'hopeful', 'quiet and unhurried', 'softly euphoric', 'contemplative', 'gently uplifting'],
  light: ['golden hour light', 'blue hour stillness', '3am quiet', 'early morning fog', 'late afternoon haze',
    'midnight calm', 'first light', 'overcast afternoon', 'dusk settling in'],
};
function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
function pickTwoDistinct(arr) {
  const a = pick(arr);
  let b = pick(arr);
  while (b === a) b = pick(arr);
  return [a, b];
}
function randomisePrompt() {
  const [tex1, tex2] = pickTwoDistinct(BANKS.texture);
  const parts = [pick(BANKS.mood), pick(BANKS.scene), tex1, tex2, pick(BANKS.light)];
  document.getElementById('prompt').value = parts.join(', ');
}

function buildPrompt() {
  const parts = ['chillout'];
  DIALS.forEach(d => parts.push(d.describe(knobs[d.key].value)));
  // Positive steers only. The exclusions go to the negative branch instead --
  // writing "no drums" here would just feed the model the word "drums".
  if (document.getElementById('noDrums').checked) parts.push('beatless, free-time, sustained');
  if (document.getElementById('noVocals').checked) parts.push('instrumental');
  const extra = document.getElementById('prompt').value.trim();
  if (extra) parts.push(extra);
  return parts.join(', ');
}

// Terms fed to the classifier-free-guidance negative branch, which the model is
// actively pushed away from.
const EXCLUDE_TERMS = {
  noDrums: 'drums, percussion, drum beat, kick drum, snare, hi-hats, rhythmic beat',
  noVocals: 'vocals, singing, voice, choir, lyrics, spoken word',
  noBass: 'bass guitar, deep bass, sub bass, heavy low end',
};
function buildNegativePrompt() {
  return Object.keys(EXCLUDE_TERMS)
    .filter(id => document.getElementById(id).checked)
    .map(id => EXCLUDE_TERMS[id])
    .join(', ');
}

async function generate() {
  const btn = document.getElementById('genBtn');
  const outbar = document.getElementById('outbar');
  const barInner = document.getElementById('barInner');
  const statusPct = document.getElementById('statusPct');
  const statusLabel = document.getElementById('statusLabel');
  const errorText = document.getElementById('errorText');
  const player = document.getElementById('player');
  const dur = parseFloat(document.getElementById('duration').value);

  const cancelBtn = document.getElementById('cancelBtn');
  const playerWrap = document.getElementById('playerWrap');

  btn.disabled = true;
  errorText.style.display = 'none';
  playerWrap.hidden = true;
  playerWrap.classList.remove('ready');
  outbar.classList.add('busy');
  cancelBtn.style.display = 'inline-block';
  barInner.style.width = '0%';
  statusPct.textContent = '0%';
  statusLabel.textContent = dur > 30 ? 'Generating (chained)...' : 'Generating...';

  // Everything exported from the PREVIOUS track is now stale -- without this the
  // chips keep pointing at old renders under the new track's filenames, which is
  // worse than offering nothing at all.
  ['downloadBtn', 'masterChip', 'downloadVideoBtn', 'stillChip'].forEach(id => setChip(id, null));
  [lastVideoUrl, lastMasterUrl, lastStillUrl].forEach(u => { if (u) URL.revokeObjectURL(u); });
  lastVideoUrl = lastMasterUrl = lastStillUrl = null;
  setVidStatus('');
  document.getElementById('masterStatus').style.display = 'none';
  // steps 02/03 were completed against the OLD track, so they're no longer done
  flow.mastered = false;
  flow.rendered = false;
  renderFlow();

  try {
    const startRes = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: buildPrompt(),
        negative_prompt: buildNegativePrompt(),
        duration_sec: dur,
      })
    });
    if (!startRes.ok) throw new Error('Server error: ' + startRes.status);
    const { job_id } = await startRes.json();
    currentJobId = job_id;

    while (true) {
      await new Promise(r => setTimeout(r, 400));
      const s = await (await fetch('/status/' + job_id)).json();
      // check the error first: an error payload carries no `progress`, and
      // rendering it anyway flashes "NaN%" before the throw lands
      if (s.error) throw new Error(s.error);
      if (typeof s.progress === 'number') {
        const pct = Math.round(s.progress * 100);
        barInner.style.width = pct + '%';
        statusPct.textContent = pct + '%';
      }
      if (s.done) break;
    }

    const audioRes = await fetch('/result/' + job_id);
    const blob = await audioRes.blob();
    if (lastAudioUrl) URL.revokeObjectURL(lastAudioUrl);
    const url = URL.createObjectURL(blob);
    lastAudioUrl = url;
    player.src = url;

    playerWrap.hidden = false;
    // next frame, so the browser has a non-hidden element to transition from
    requestAnimationFrame(() => playerWrap.classList.add('ready'));
    player.play().catch(() => {});  // autoplay can be refused; not worth failing the run over

    statusLabel.textContent = `Ready · ${dur}s`;
    statusPct.textContent = '';

    flow.generated = true;
    renderFlow();
    unlockExportSteps();
    if (!document.getElementById('trackTitle').value.trim()) rerollTitle();
    updateDownloadNames();  // also lights the WAV chip
    refreshPreview();
  } catch (e) {
    errorText.textContent = e.message === 'Cancelled' ? 'Cancelled.' : 'Error: ' + e.message;
    errorText.style.display = 'block';
    barInner.style.width = '0%';
    statusPct.textContent = '';
    // the previous track's blob URL is only revoked on success, so if this run
    // failed the old one is still playable -- restore it rather than blanking
    if (flow.generated && player.src) {
      playerWrap.hidden = false;
      requestAnimationFrame(() => playerWrap.classList.add('ready'));
      statusLabel.textContent = 'Previous track still loaded';
      setChip('downloadBtn', lastAudioUrl, buildFilename('wav'));
    } else {
      statusLabel.textContent = 'No track yet — hit Generate';
    }
  } finally {
    btn.disabled = false;
    cancelBtn.style.display = 'none';
    outbar.classList.remove('busy');
  }
}

async function cancelGeneration() {
  if (!currentJobId) return;
  const cancelBtn = document.getElementById('cancelBtn');
  cancelBtn.textContent = 'Cancelling...';
  try {
    await fetch('/cancel/' + currentJobId, { method: 'POST' });
  } catch (e) { /* the poll loop will surface whatever actually happened */ }
  cancelBtn.textContent = 'Cancel';
}

// ---- Creative Export: title, mastering, social video/waveform studio ----

const TITLE_WORDS = {
  adj: ['Amber', 'Velvet', 'Quiet', 'Hollow', 'Distant', 'Gentle', 'Faded', 'Midnight',
    'Golden', 'Hazy', 'Slow', 'Soft', 'Drifting', 'Muted', 'Warm', 'Still'],
  noun: ['Tideline', 'Hush', 'Static', 'Bloom', 'Horizon', 'Ember', 'Fog', 'Reverie',
    'Lantern', 'Echo', 'Harbor', 'Stillwater', 'Glow', 'Dust', 'Current', 'Sanctuary'],
};
function rerollTitle() {
  document.getElementById('trackTitle').value = `${pick(TITLE_WORDS.adj)} ${pick(TITLE_WORDS.noun)}`;
  updateDownloadNames();
  refreshPreview();
}

// Every chip agrees on naming, and re-derives it whenever the title or filename
// pattern changes.
function updateDownloadNames() {
  if (lastAudioUrl) setChip('downloadBtn', lastAudioUrl, buildFilename('wav'));
  if (lastMasterUrl) document.getElementById('masterChip').download = buildFilename(lastMasterExt);
  if (lastVideoUrl) document.getElementById('downloadVideoBtn').download = buildFilename('webm');
  if (lastStillUrl) document.getElementById('stillChip').download = buildFilename('png');
}
let lastMasterExt = 'wav';

// `pointer-events: none` alone still leaves everything inside a "disabled" card
// reachable by Tab, so the cards use `inert` to actually take them out of play.
function setCardEnabled(id, enabled) {
  const card = document.getElementById(id);
  const wasDisabled = card.classList.contains('disabled');
  card.classList.toggle('disabled', !enabled);
  card.inert = !enabled;
  card.setAttribute('aria-hidden', enabled ? 'false' : 'true');
  return wasDisabled && enabled;  // true only on the locked -> unlocked transition
}

// ---- Step flow: 01 Generate -> 02 Master -> 03 Video ----
const flow = { generated: false, mastered: false, rendered: false };

function setNodeState(nodeId, cardId, state) {
  document.getElementById(nodeId).className = 'rail-node' + (state ? ' ' + state : '');
  const card = document.getElementById(cardId);
  card.classList.toggle('is-active', state === 'active');
  card.classList.toggle('is-done', state === 'done');
}

function renderFlow() {
  setNodeState('railStep1', 'generateCard', flow.generated ? 'done' : 'active');
  setNodeState('railStep2', 'audioCard', !flow.generated ? '' : (flow.mastered ? 'done' : 'active'));
  setNodeState('railStep3', 'videoCard', !flow.generated ? '' : (flow.rendered ? 'done' : 'active'));
  document.getElementById('railLine1').classList.toggle('filled', flow.generated);
  document.getElementById('railLine2').classList.toggle('filled', flow.generated);
}

// Unlocking steps 02 and 03 together, staggered, so the eye follows the flow
// left to right rather than both cards lighting up at once.
function unlockExportSteps() {
  [['audioCard', 0], ['videoCard', 160]].forEach(([id, delay]) => {
    const justUnlocked = setCardEnabled(id, true);
    if (!justUnlocked) return;
    const card = document.getElementById(id);
    setTimeout(() => {
      card.classList.add('unlocking');
      card.addEventListener('animationend', function done(e) {
        if (e.animationName !== 'unlockRise') return;
        card.classList.remove('unlocking');
        card.removeEventListener('animationend', done);
      });
    }, delay);
  });
}

renderFlow();

function buildFilename(ext) {
  const title = (document.getElementById('trackTitle').value || 'Untitled Chillout Track').trim();
  const pattern = document.getElementById('namePattern').value;
  let name;
  switch (pattern) {
    case 'numbered-dash': name = '01 - ' + title; break;
    case 'numbered-dot': name = '01. ' + title; break;
    case 'extended': name = title + ' (Extended Mix)'; break;
    case 'instrumental': name = title + ' (Instrumental Mix)'; break;
    case 'slowed': name = title + ' [Slowed + Reverb]'; break;
    default: name = title;
  }
  const safe = name.replace(/[\\/:*?"<>|]/g, '').trim() || 'track';
  return `${safe}.${ext}`;
}

// ---- Audio shaping controls ----
function bindSlider(id, valId) {
  const el = document.getElementById(id), out = document.getElementById(valId);
  const sync = () => { out.textContent = el.value; };
  el.addEventListener('input', sync);
  sync();
}
bindSlider('stereoWidth', 'widthVal');
bindSlider('fadeIn', 'fadeInVal');
bindSlider('fadeOut', 'fadeOutVal');

// A looping track must not fade, so the server ignores the fades when seamless is
// on -- reflect that in the UI instead of leaving dead controls that look live.
const seamlessEl = document.getElementById('seamlessLoop');
function syncSeamlessState() {
  const fields = document.getElementById('fadeFields');
  fields.inert = seamlessEl.checked;
  fields.style.opacity = seamlessEl.checked ? '.35' : '1';
}
seamlessEl.addEventListener('change', syncSeamlessState);
syncSeamlessState();

function humanSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function downloadMastered() {
  if (!currentJobId) return;
  const btn = document.getElementById('masterBtn');
  const status = document.getElementById('masterStatus');
  const fmt = document.getElementById('exportFormat').value;
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Exporting...';
  status.style.display = 'none';
  try {
    const params = new URLSearchParams({
      preset: document.getElementById('masterPreset').value,
      fmt,
      width: (parseFloat(document.getElementById('stereoWidth').value) / 100).toFixed(3),
      seamless: seamlessEl.checked ? 'true' : 'false',
      fade_in: document.getElementById('fadeIn').value,
      fade_out: document.getElementById('fadeOut').value,
    });
    const res = await fetch(`/master/${currentJobId}?${params}`);
    if (!res.ok) throw new Error('Server error: ' + res.status);
    const blob = await res.blob();
    if (lastMasterUrl) URL.revokeObjectURL(lastMasterUrl);
    lastMasterUrl = URL.createObjectURL(blob);
    lastMasterExt = fmt.toLowerCase();
    // the result lands as a chip in the output bar rather than downloading
    // straight away, so every artefact lives in one place
    setChip('masterChip', lastMasterUrl, buildFilename(lastMasterExt),
            `${fmt} · ${humanSize(blob.size)}`);
    status.textContent = 'Ready in the bar below';
    status.style.display = 'block';
    flow.mastered = true;
    renderFlow();
  } catch (e) {
    status.textContent = 'Export failed: ' + e.message;
    status.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

// -- Video/waveform studio --
const PALETTES = {
  gold: ['#D4B97A', '#C1A673'],
  ember: ['#E3A868', '#8C5A34'],
  moonlit: ['#DCE6E2', '#7C8F8C'],
};
const ASPECTS = { '9:16': [540, 960], '1:1': [720, 720], '16:9': [960, 540] };

let audioCtx = null, analyserNode = null, mediaStreamDest = null;
async function ensureAudioGraph() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaElementSource(document.getElementById('player'));
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 1024;
    mediaStreamDest = audioCtx.createMediaStreamDestination();
    source.connect(analyserNode);
    analyserNode.connect(audioCtx.destination);
    analyserNode.connect(mediaStreamDest);
  }
  // A context can be created suspended under browser autoplay policy. Routing the
  // player through a suspended context silences BOTH playback and the recorded
  // audio track, so this resume is what keeps rendered videos from coming out mute.
  if (audioCtx.state === 'suspended') await audioCtx.resume();
}

function sizeExportCanvas() {
  const canvas = document.getElementById('exportCanvas');
  const [w, h] = ASPECTS[document.getElementById('vidAspect').value];
  canvas.width = w;
  canvas.height = h;
}

let videoFeathers = null;
function ensureVideoFeathers(w, h) {
  if (videoFeathers && videoFeathers.w === w && videoFeathers.h === h) return videoFeathers;
  const count = 16;
  videoFeathers = {
    w, h,
    particles: Array.from({ length: count }, () => ({
      x: Math.random() * w, y: Math.random() * h, baseX: 0,
      size: 6 + Math.random() * 7,
      speedY: 14 + Math.random() * 16,
      swayAmp: 14 + Math.random() * 18,
      swayFreq: 0.15 + Math.random() * 0.25,
      phase: Math.random() * Math.PI * 2,
      rot: Math.random() * Math.PI * 2,
      rotSpeed: (Math.random() - 0.5) * 0.8,
      opacity: 0.14 + Math.random() * 0.24,
    })),
  };
  videoFeathers.particles.forEach(p => { p.baseX = p.x; });
  return videoFeathers;
}

let backdropLastT = null;
function drawBackdrop(ctx, w, h, t, mode) {
  ctx.fillStyle = '#090807';
  ctx.fillRect(0, 0, w, h);
  if (mode === 'minimal') return;
  if (mode === 'bloom') {
    const cx1 = w * 0.3 + Math.sin(t * 0.3) * w * 0.08, cy1 = h * 0.35 + Math.cos(t * 0.24) * h * 0.06;
    const g1 = ctx.createRadialGradient(cx1, cy1, 0, cx1, cy1, Math.max(w, h) * 0.55);
    g1.addColorStop(0, 'rgba(193,166,115,0.32)');
    g1.addColorStop(1, 'rgba(193,166,115,0)');
    ctx.fillStyle = g1; ctx.fillRect(0, 0, w, h);
    const cx2 = w * 0.75 + Math.cos(t * 0.18) * w * 0.07, cy2 = h * 0.7 + Math.sin(t * 0.27) * h * 0.07;
    const g2 = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, Math.max(w, h) * 0.45);
    g2.addColorStop(0, 'rgba(212,185,122,0.2)');
    g2.addColorStop(1, 'rgba(212,185,122,0)');
    ctx.fillStyle = g2; ctx.fillRect(0, 0, w, h);
    return;
  }
  // feathers -- advanced by real elapsed time, not a hardcoded frame rate, so the
  // drift looks the same whether we're rendering at 60fps or redrawing a single
  // preview frame
  const field = ensureVideoFeathers(w, h);
  const dt = backdropLastT === null ? 0 : Math.min(Math.max(t - backdropLastT, 0), 0.1);
  backdropLastT = t;
  for (const p of field.particles) {
    p.y += p.speedY * dt;
    p.rot += p.rotSpeed * dt;
    if (p.y - p.size * 1.3 > h) { p.y = -p.size * 1.3; p.baseX = Math.random() * w; p.phase = Math.random() * Math.PI * 2; }
    p.x = p.baseX + Math.sin(t * p.swayFreq * Math.PI * 2 + p.phase) * p.swayAmp;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rot);
    ctx.globalAlpha = p.opacity;
    drawFeatherShape(ctx, p.size, '#D4B97A', '#C1A673', 'rgba(9,8,7,.25)');
    ctx.restore();
  }
}

// When nothing is playing (static preview, still frame, or before the audio graph
// exists) the analyser would hand back pure silence -- an empty frame or a dead
// flat line. Synthesise a calm, deterministic shape instead so previews and
// exported stills still look like the product.
const IDLE_SHAPE = Array.from({ length: 512 }, (_, i) => {
  const x = i / 512;
  const env = Math.sin(Math.PI * x);
  return 128 + env * 46 * (Math.sin(x * 34) * 0.6 + Math.sin(x * 11 + 1.7) * 0.4);
});
function waveformData(kind) {
  const live = analyserNode && !document.getElementById('player').paused;
  if (kind === 'freq') {
    const data = new Uint8Array(live ? analyserNode.frequencyBinCount : 512);
    if (live) analyserNode.getByteFrequencyData(data);
    else for (let i = 0; i < data.length; i++) {
      data[i] = Math.max(0, Math.abs(IDLE_SHAPE[i] - 128) * 3.2 * (1 - i / data.length));
    }
    return data;
  }
  const data = new Uint8Array(live ? analyserNode.fftSize : IDLE_SHAPE.length);
  if (live) analyserNode.getByteTimeDomainData(data);
  else for (let i = 0; i < data.length; i++) data[i] = IDLE_SHAPE[i];
  return data;
}

function drawWaveform(ctx, w, h, style, palette) {
  if (style === 'off') return;
  const [top, bottom] = PALETTES[palette] || PALETTES.gold;
  ctx.globalAlpha = 1;
  if (style === 'bars') {
    const data = waveformData('freq');
    const bars = 40, step = Math.floor(data.length / bars);
    const barW = w / bars * 0.6, gap = w / bars;
    for (let i = 0; i < bars; i++) {
      const v = data[i * step] / 255;
      const barH = v * h * 0.5;
      const grad = ctx.createLinearGradient(0, h / 2 - barH, 0, h / 2 + barH);
      grad.addColorStop(0, top); grad.addColorStop(1, bottom);
      ctx.fillStyle = grad;
      ctx.fillRect(i * gap + (gap - barW) / 2, h / 2 - barH, barW, barH * 2);
    }
  } else if (style === 'wave') {
    const data = waveformData('time');
    ctx.beginPath();
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, bottom); grad.addColorStop(0.5, top); grad.addColorStop(1, bottom);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 3;
    for (let i = 0; i < data.length; i++) {
      const x = (i / data.length) * w;
      const y = h / 2 + (data[i] / 128 - 1) * h * 0.28;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  } else if (style === 'pulse') {
    const data = waveformData('time');
    let sumSq = 0;
    for (let i = 0; i < data.length; i++) { const v = data[i] / 128 - 1; sumSq += v * v; }
    const rms = Math.sqrt(sumSq / data.length);
    const cx = w / 2, cy = h / 2;
    [1, 0.7, 0.45].forEach((f, idx) => {
      const r = Math.min(w, h) * (0.14 + rms * 0.5) * f;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = idx === 0 ? top : `${bottom}66`;
      ctx.lineWidth = idx === 0 ? 2.5 : 1.2;
      ctx.stroke();
    });
  }
}

function drawWatermark(ctx, w, h, mode) {
  if (mode === 'none') return;
  ctx.save();
  ctx.globalAlpha = 0.55;
  ctx.fillStyle = '#D4B97A';
  ctx.font = '600 ' + Math.round(w * 0.022) + 'px Montserrat, sans-serif';
  ctx.textAlign = 'right';
  const label = mode === 'wordmark' ? 'TRANQUILICY' : '◎';
  ctx.fillText(label, w - w * 0.05, h - h * 0.045);
  ctx.restore();
}

function drawTitleText(ctx, w, h) {
  const title = document.getElementById('trackTitle').value.trim() || 'Untitled Chillout Track';
  ctx.save();
  ctx.textAlign = 'center';
  ctx.fillStyle = '#F2EFE9';
  ctx.font = 'italic 400 ' + Math.round(w * 0.055) + 'px "Cormorant Garamond", Georgia, serif';
  ctx.globalAlpha = 0.92;
  ctx.fillText(title, w / 2, h * 0.13);
  ctx.restore();
}

function drawExportFrame(t) {
  const canvas = document.getElementById('exportCanvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  drawBackdrop(ctx, w, h, t, document.getElementById('vidBackdrop').value);
  drawTitleText(ctx, w, h);
  drawWaveform(ctx, w, h, document.getElementById('vidStyle').value, document.getElementById('vidPalette').value);
  drawWatermark(ctx, w, h, document.getElementById('vidWatermark').value);
}

// Every control that affects the frame refreshes the preview immediately.
// (Resizing the canvas clears it, so changing aspect ratio used to leave a blank
// black box until you hit Render.)
['vidStyle', 'vidPalette', 'vidBackdrop', 'vidAspect', 'vidWatermark'].forEach(id => {
  document.getElementById(id).addEventListener('change', refreshPreview);
});
document.getElementById('trackTitle').addEventListener('input', () => {
  refreshPreview();
  updateDownloadNames();
});
document.getElementById('namePattern').addEventListener('change', updateDownloadNames);

function refreshPreview() {
  sizeExportCanvas();
  drawExportFrame(performance.now() / 1000);
}
refreshPreview();

function setVidStatus(msg) {
  const status = document.getElementById('vidStatus');
  status.style.display = msg ? 'block' : 'none';
  status.textContent = msg || '';
}

async function renderVideo() {
  const player = document.getElementById('player');
  const btn = document.getElementById('renderVidBtn');

  if (!player.src) { setVidStatus('Generate a track first.'); return; }
  if (typeof MediaRecorder === 'undefined') {
    setVidStatus('This browser cannot record video.');
    return;
  }

  btn.disabled = true;
  setChip('downloadVideoBtn', null);  // the old render no longer matches these settings
  setVidStatus('Rendering — plays the track once...');

  try {
    await renderVideoInner(player, btn);
  } catch (e) {
    // without this the button stays disabled forever on any failure
    setVidStatus('Render failed: ' + e.message);
    btn.disabled = false;
  }
}

async function renderVideoInner(player, btn) {
  await ensureAudioGraph();
  sizeExportCanvas();
  const canvas = document.getElementById('exportCanvas');

  const canvasStream = canvas.captureStream(30);
  const combined = new MediaStream([...canvasStream.getVideoTracks(), ...mediaStreamDest.stream.getAudioTracks()]);
  const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus') ? 'video/webm;codecs=vp9,opus' : 'video/webm';
  const recorder = new MediaRecorder(combined, { mimeType, videoBitsPerSecond: 4_000_000 });
  const chunks = [];
  recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };

  let rafId, stopped = false;
  const startT = performance.now();
  function loop() {
    drawExportFrame((performance.now() - startT) / 1000);
    rafId = requestAnimationFrame(loop);
  }
  function stopEverything() {
    if (stopped) return;
    stopped = true;
    cancelAnimationFrame(rafId);
    if (recorder.state !== 'inactive') recorder.stop();
    player.pause();
    player.removeEventListener('ended', stopEverything);
    // each render calls captureStream() afresh; without stopping the old tracks
    // every render leaves a live video track attached to the canvas
    canvasStream.getTracks().forEach(t => t.stop());
  }
  // stop on real playback end as well as the timer, so the video length follows
  // the audio rather than a wall-clock estimate that drifts if playback stalls
  player.addEventListener('ended', stopEverything);

  recorder.onstop = () => {
    if (lastVideoUrl) URL.revokeObjectURL(lastVideoUrl);
    const blob = new Blob(chunks, { type: 'video/webm' });
    lastVideoUrl = URL.createObjectURL(blob);
    setChip('downloadVideoBtn', lastVideoUrl, buildFilename('webm'),
            `Video · ${humanSize(blob.size)}`);
    setVidStatus('Ready in the bar below');
    btn.disabled = false;
    flow.rendered = true;
    renderFlow();
  };

  const lengthMode = document.getElementById('vidLength').value;
  const fullDuration = isFinite(player.duration) && player.duration > 0 ? player.duration : 20;
  const targetSec = lengthMode === 'loop15' ? Math.min(15, fullDuration) : fullDuration;

  player.currentTime = 0;
  recorder.start();
  loop();
  try {
    await player.play();
  } catch (e) {
    stopEverything();
    throw new Error('could not play the track back for recording');
  }
  setTimeout(stopEverything, targetSec * 1000 + 250);
}

function saveStillFrame() {
  sizeExportCanvas();
  drawExportFrame(0.001);
  document.getElementById('exportCanvas').toBlob(blob => {
    if (lastStillUrl) URL.revokeObjectURL(lastStillUrl);
    lastStillUrl = URL.createObjectURL(blob);
    setChip('stillChip', lastStillUrl, buildFilename('png'), `Still · ${humanSize(blob.size)}`);
    setVidStatus('Still frame ready in the bar below');
  }, 'image/png');
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
