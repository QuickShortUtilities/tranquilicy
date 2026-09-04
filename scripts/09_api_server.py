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
from fastapi import FastAPI, Request
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

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/gpu")
def gpu_status():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    mem_allocated = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
    mem_total = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0
    return {
        "status": "ready",
        "device": str(device),
        "gpu_name": gpu_name,
        "vram_allocated_gb": round(mem_allocated, 2),
        "vram_total_gb": round(mem_total, 2),
        "model": MODEL_ID,
        "version": APP_VERSION,
    }


# ---- Security, IP Quotas & GPU Concurrency Queue ---------------------------
MAX_GENERATIONS_PER_IP = 5
MAX_DOWNLOADS_PER_IP = 5
QUOTA_WINDOW_SEC = 86400.0  # 24 hours
MAX_QUEUE_WAITING = 2       # max 2 jobs waiting in queue behind the 1 running

ip_quotas = {}  # ip -> {"generations": int, "downloads": int, "window_start": float, "active_job_id": str | None}
quota_lock = threading.Lock()

def get_client_ip(request: Request) -> str:
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"

def is_admin_ip(ip: str) -> bool:
    return ip in ("127.0.0.1", "::1", "localhost") or ip.startswith("192.168.") or ip.startswith("10.")

def get_or_create_quota(ip: str) -> dict:
    now = time.time()
    with quota_lock:
        if ip not in ip_quotas or (now - ip_quotas[ip]["window_start"] > QUOTA_WINDOW_SEC):
            ip_quotas[ip] = {
                "generations": 0,
                "downloads": 0,
                "window_start": now,
                "active_job_id": None
            }
        return ip_quotas[ip]

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
            if job_id in jobs:
                jobs[job_id]["started"] = True
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
def generate(req: GenerateRequest, request: Request):
    prune_old_jobs()
    ip = get_client_ip(request)
    admin = is_admin_ip(ip)

    # 1. Concurrency queue capacity check
    waiting_jobs = [j for j in jobs.values() if not j["done"] and not j.get("started", False)]
    if not admin and len(waiting_jobs) >= MAX_QUEUE_WAITING:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Studio is currently at peak capacity ({len(waiting_jobs)} creators in queue). Please try again in 30 seconds."}
        )

    # 2. Per-IP quotas and single active job check
    if not admin:
        q = get_or_create_quota(ip)
        # Check active job
        active_id = q.get("active_job_id")
        if active_id and active_id in jobs and not jobs[active_id]["done"]:
            return JSONResponse(
                status_code=429,
                content={"detail": "You already have a generation running or in queue. Please wait for it to complete."}
            )

        # Check 5 generations limit
        if q["generations"] >= MAX_GENERATIONS_PER_IP:
            now = time.time()
            hours_left = max(1, int((QUOTA_WINDOW_SEC - (now - q["window_start"])) / 3600))
            return JSONResponse(
                status_code=429,
                content={"detail": f"Daily limit reached ({MAX_GENERATIONS_PER_IP}/{MAX_GENERATIONS_PER_IP} generations). Resets in ~{hours_left}h."}
            )
        q["generations"] += 1

    duration_sec = max(MIN_DURATION, min(MAX_DURATION, req.duration_sec))
    guidance_scale = max(1.0, min(10.0, req.guidance_scale))

    prompt = req.prompt.strip()[:800]
    negative_prompt = req.negative_prompt.strip()[:400]
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "progress": 0.0, "done": False, "error": None, "audio": None,
        "cancelled": False, "created_at": time.time(), "ip": ip, "started": False
    }
    if not admin:
        q["active_job_id"] = job_id

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
    
    in_queue = not job.get("started", False) and not job["done"]
    queue_pos = 0
    est_sec = 0
    if in_queue:
        earlier_jobs = [j for j in jobs.values() if not j["done"] and j["created_at"] < job["created_at"]]
        queue_pos = len(earlier_jobs) + 1
        est_sec = queue_pos * 12

    return {
        "progress": job["progress"],
        "done": job["done"],
        "error": job["error"],
        "in_queue": in_queue,
        "queue_position": queue_pos,
        "estimated_sec": est_sec
    }


@app.get("/result/{job_id}")
def result(job_id: str, request: Request):
    job = jobs.get(job_id)
    if job is None or job["audio"] is None:
        return JSONResponse({"error": "not ready"}, status_code=404)
    
    ip = get_client_ip(request)
    if not is_admin_ip(ip):
        q = get_or_create_quota(ip)
        if q["downloads"] >= MAX_DOWNLOADS_PER_IP:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Daily download limit ({MAX_DOWNLOADS_PER_IP}/{MAX_DOWNLOADS_PER_IP}) reached for this IP. Audio remains playable in browser."}
            )
        q["downloads"] += 1

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


def apply_warmth(audio: np.ndarray, warmth: float) -> np.ndarray:
    """Soft analog tape saturation using gentle cubic non-linearity."""
    if warmth <= 0.0:
        return audio
    drive = 1.0 + warmth * 1.5
    driven = audio * drive
    sat = np.clip(driven, -1.5, 1.5)
    sat = sat - (sat ** 3) / 6.0
    return (sat / drive).astype(np.float32)


def apply_air(audio: np.ndarray, air: float) -> np.ndarray:
    """High-frequency air lift."""
    if air <= 0.0:
        return audio
    diff = np.zeros_like(audio)
    if audio.ndim == 1:
        diff[1:] = audio[1:] - audio[:-1]
    else:
        diff[1:, :] = audio[1:, :] - audio[:-1, :]
    out = audio + (air * 0.35) * diff
    return np.clip(out, -0.99, 0.99).astype(np.float32)


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



@app.get("/quota")
def quota(request: Request):
    ip = get_client_ip(request)
    admin = is_admin_ip(ip)
    now = time.time()
    queue_len = len([j for j in jobs.values() if not j["done"]])
    if admin:
        return {
            "is_admin": True,
            "generations_remaining": 999,
            "generations_max": 999,
            "downloads_remaining": 999,
            "downloads_max": 999,
            "queue_depth": queue_len,
            "resets_in_sec": 0
        }
    q = get_or_create_quota(ip)
    elapsed = now - q["window_start"]
    resets_in = max(0, int(QUOTA_WINDOW_SEC - elapsed))
    return {
        "is_admin": False,
        "generations_remaining": max(0, MAX_GENERATIONS_PER_IP - q["generations"]),
        "generations_max": MAX_GENERATIONS_PER_IP,
        "downloads_remaining": max(0, MAX_DOWNLOADS_PER_IP - q["downloads"]),
        "downloads_max": MAX_DOWNLOADS_PER_IP,
        "queue_depth": queue_len,
        "resets_in_sec": resets_in
    }

@app.get("/master/{job_id}")
def master(job_id: str, preset: str = "streaming", fade_in: float = 0.0, fade_out: float = 0.0,
           seamless: bool = False, width: float = 0.0, fmt: str = "WAV",
           warmth: float = 0.0, air: float = 0.0):
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
    if warmth > 0:
        audio = apply_warmth(audio, warmth)
    if air > 0:
        audio = apply_air(audio, air)
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
    from pathlib import Path
    disk_html = Path(__file__).resolve().parent.parent / "index.html"
    source = disk_html.read_text(encoding="utf-8") if disk_html.is_file() else INDEX_HTML
    html = (
        source
        .replace("__APP_VERSION__", APP_VERSION)
        .replace("__FORMAT_OPTIONS__", format_options)
    )
    # The page is edited and the server restarted constantly during development;
    # without this the browser can serve a cached copy and hide the new build.
    return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})


INDEX_HTML = """<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<title>Tranquilicy Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,400&family=Montserrat:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --gold: #C1A673; --gold-deep: #A58B58; --gold-text: #D4B97A; --gold-glow: rgba(193,166,115,.18); --on-gold: #fff;
    --bg: #090807; --bg-2: #100F0D; --bg-3: #181612; --bg-4: #221F1A;
    --text: #F2EFE9; --text-2: #9A9188; --text-3: #524D48;
    --line: rgba(255,255,255,.07); --line-soft: rgba(255,255,255,.045); --line-hi: rgba(255,255,255,.13);
    --display: "Cormorant Garamond", Georgia, serif;
    --ui: "Montserrat", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --max: 1380px; --radius: 20px; --radius-sm: 12px; --radius-xs: 8px;
    --lift: 0 1px 3px rgba(0,0,0,.5), 0 12px 32px rgba(0,0,0,.6);
    --ease: cubic-bezier(.22,.61,.36,1); --ease-out: cubic-bezier(.16,1,.3,1);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--ui); font-weight: 300; overflow-x: hidden; }
  #featherCanvas { position: fixed; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; }
  .wrap { position: relative; z-index: 1; max-width: var(--max); margin: 0 auto; padding: 60px 24px 60px; }
  
  
  /* ---- GPU Status Badge & Modal ---- */
  .brand-wrap { display: flex; justify-content: space-between; align-items: center; margin-bottom: 34px; flex-wrap: wrap; gap: 14px; }
  .gpu-badge {
    display: inline-flex; align-items: center; gap: 8px; padding: 7px 14px;
    background: var(--bg-2); border: 1px solid var(--line); border-radius: 999px;
    font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--text-2);
    cursor: pointer; transition: all .2s var(--ease); box-shadow: var(--lift);
  }
  .gpu-badge:hover { border-color: var(--gold); background: var(--bg-3); }
  .gpu-dot { width: 7px; height: 7px; border-radius: 50%; background: #9A9188; transition: all .3s var(--ease); }
  .gpu-badge.online .gpu-dot { background: #4ADE80; box-shadow: 0 0 10px rgba(74,222,128,.7); }
  .gpu-badge.offline .gpu-dot { background: #F87171; box-shadow: 0 0 8px rgba(248,113,113,.5); }
  
  .modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,.75); backdrop-filter: blur(6px);
    display: flex; align-items: center; justify-content: center; z-index: 10000; padding: 20px;
  }
  .modal-card {
    background: var(--bg-2); border: 1px solid var(--line-hi); border-radius: var(--radius);
    padding: 28px; max-width: 480px; width: 100%; box-shadow: 0 12px 48px rgba(0,0,0,.8);
  }

  .brand { display: flex; align-items: center; gap: 14px; margin-bottom: 36px; }
  .brand .ring { width: 36px; height: 36px; border-radius: 50%; border: 1.5px solid var(--gold); position: relative; flex: none; box-shadow: 0 0 16px var(--gold-glow); }
  .brand .ring::after { content: ""; position: absolute; inset: 8px; border-radius: 50%; border: 1.5px solid var(--gold-glow); }
  .brand-title { display: flex; flex-direction: column; }
  .brand-title span:first-child { font-family: var(--ui); letter-spacing: .16em; text-transform: uppercase; font-size: 13px; font-weight: 500; color: var(--gold-text); }
  .brand-title span:last-child { font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--text-3); margin-top: 2px; }
  
  h1 { font-family: var(--display); font-weight: 400; font-style: italic; font-size: 42px; margin: 0 0 8px; color: var(--text); letter-spacing: .01em; }
  .sub { color: var(--text-2); font-size: 14px; margin-bottom: 34px; letter-spacing: .01em; line-height: 1.5; }
  .card { background: var(--bg-2); border: 1px solid var(--line); border-radius: var(--radius); padding: 26px; box-shadow: var(--lift); position: relative; }
  
  label { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .12em; color: var(--text-2); margin-bottom: 10px; }
  textarea { width: 100%; min-height: 96px; margin-top: 8px; background: var(--bg-3); color: var(--text); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 14px; font-family: var(--ui); font-size: 13.5px; font-weight: 300; line-height: 1.5; resize: vertical; transition: border-color .2s var(--ease); }
  textarea:focus { outline: none; border-color: var(--gold-glow); box-shadow: 0 0 0 3px rgba(193,166,115,.1); }
  
  .row { display: flex; justify-content: space-between; align-items: center; margin-top: 22px; }
  .row .val { color: var(--gold-text); font-family: var(--display); font-size: 18px; font-style: italic; }
  input[type=range] { width: 100%; margin-top: 8px; accent-color: var(--gold); }
  
  .btn-primary {
    width: 100%; margin-top: 24px; padding: 15px; border: none; border-radius: var(--radius-sm);
    background: linear-gradient(180deg, var(--gold), var(--gold-deep)); color: var(--on-gold);
    font-family: var(--ui); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .14em;
    cursor: pointer; transition: transform .2s var(--ease-out), box-shadow .2s var(--ease-out);
    box-shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 20px rgba(193,166,115,.18);
  }
  .btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 2px 6px rgba(0,0,0,.5), 0 16px 32px rgba(193,166,115,.38); }
  .btn-primary:active:not(:disabled) { transform: scale(.975); }
  .btn-primary:disabled { opacity: .5; cursor: not-allowed; }
  
  .link-btn {
    display: inline-flex; align-items: center; gap: 5px; padding: 0; border: none; background: none;
    font-family: var(--ui); font-weight: 400; font-size: 11px; text-transform: uppercase;
    letter-spacing: .1em; color: var(--gold-text); text-decoration: none; cursor: pointer;
    transition: opacity .2s var(--ease), color .2s var(--ease);
  }
  .link-btn:hover { opacity: .8; color: #fff; }
  .link-btn:focus-visible { outline: 2px solid var(--gold-glow); outline-offset: 3px; border-radius: 2px; }
  
  .action-row-mini { display: flex; gap: 10px; align-items: center; }
  
  #barOuter { background: var(--bg-3); border: 1px solid var(--line); border-radius: 999px; height: 8px; overflow: hidden; }
  #barInner { background: linear-gradient(90deg, var(--gold-deep), var(--gold)); height: 100%; width: 0%; transition: width .3s var(--ease-out); border-radius: 999px; }
  #errorText { color: #e08a8a; font-size: 12px; margin-top: 10px; display: none; }
  
  audio { width: 100%; margin-top: 14px; border-radius: var(--radius-xs); }
  audio::-webkit-media-controls-panel { background: var(--bg-3); }
  
  .btn-ghost {
    display: block; width: 100%; margin-top: 10px; padding: 12px; border: 1px solid var(--line-hi);
    border-radius: var(--radius-sm); background: transparent; color: var(--text-2);
    font-family: var(--ui); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: .14em;
    text-align: center; text-decoration: none; cursor: pointer; box-sizing: border-box;
    transition: background .2s var(--ease), border-color .2s var(--ease), color .2s var(--ease);
  }
  .btn-ghost:hover { background: rgba(193,166,115,.06); border-color: var(--gold-glow); color: var(--gold-text); }
  
  /* Archetypes & Vibe Matrix */
  .archetype-shelf { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; margin-bottom: 16px; }
  .archetype-btn {
    background: var(--bg-3); border: 1px solid var(--line); border-radius: 999px;
    padding: 6px 12px; font-size: 10.5px; letter-spacing: .04em; color: var(--text-2);
    cursor: pointer; transition: all .2s var(--ease);
  }
  .archetype-btn:hover { border-color: var(--gold); color: var(--gold-text); background: var(--bg-4); transform: translateY(-1px); }
  
  .vibe-section { margin-top: 18px; }
  .vibe-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .vibe-nav { display: flex; gap: 6px; }
  .vibe-nav-btn {
    background: transparent; border: none; font-size: 10px; text-transform: uppercase;
    letter-spacing: .1em; color: var(--text-3); cursor: pointer; padding: 3px 6px;
    border-radius: 4px; transition: color .2s var(--ease);
  }
  .vibe-nav-btn.active, .vibe-nav-btn:hover { color: var(--gold-text); }
  
  .tag-tray { display: flex; gap: 6px; flex-wrap: wrap; max-height: 125px; overflow-y: auto; padding: 4px 2px; }
  .tag-tray::-webkit-scrollbar { width: 4px; }
  .tag-tray::-webkit-scrollbar-thumb { background: var(--line-hi); border-radius: 4px; }
  .tag-chip {
    display: inline-flex; align-items: center; gap: 4px; padding: 5px 10px;
    border-radius: 999px; font-size: 10.5px; background: var(--bg-3); border: 1px solid var(--line);
    color: var(--text-2); cursor: pointer; user-select: none; transition: all .18s var(--ease);
  }
  .tag-chip:hover { border-color: var(--gold-glow); color: var(--text); }
  .tag-chip.active { background: rgba(193,166,115,.15); border-color: var(--gold); color: var(--gold-text); box-shadow: 0 0 8px rgba(193,166,115,.15); }
  
  /* Title Drawer */
  .title-shelf { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; margin-bottom: 6px; }
  .title-pill {
    background: var(--bg-3); border: 1px solid var(--line); border-radius: 999px;
    padding: 5px 11px; font-size: 11px; color: var(--text-2); cursor: pointer;
    transition: all .2s var(--ease); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%;
  }
  .title-pill:hover { border-color: var(--gold); color: var(--gold-text); background: var(--bg-4); }
  
  /* Dials & Star */
  .dials-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px 10px; margin: 18px 0 6px; }
  .dial-wrap { text-align: center; user-select: none; }
  .knob {
    width: 50px; height: 50px; border-radius: 50%; margin: 0 auto; position: relative; cursor: ns-resize;
    background: radial-gradient(circle at 35% 30%, var(--bg-3), var(--bg-2) 70%); border: 1px solid var(--line-hi);
    box-shadow: inset 0 1px 2px rgba(0,0,0,.5); touch-action: none;
  }
  .knob::before {
    content: ""; position: absolute; top: 5px; left: 50%; width: 2px; height: 13px; background: var(--gold);
    border-radius: 2px; transform-origin: 50% 20px; transform: translateX(-50%) rotate(var(--rot, -135deg));
    transition: transform .05s linear;
  }
  .knob:hover { border-color: var(--gold-glow); }
  .knob:focus-visible { outline: none; border-color: var(--gold); box-shadow: inset 0 1px 2px rgba(0,0,0,.5), 0 0 0 3px var(--gold-glow); }
  .knob.dragging { border-color: var(--gold); }
  .dial-label { font-size: 10px; text-transform: uppercase; letter-spacing: .1em; color: var(--text-2); margin-top: 8px; }
  .dial-value { font-family: var(--display); font-style: italic; color: var(--gold-text); font-size: 13px; margin-top: 2px; }
  
  .star-wrap {
    display: flex; justify-content: center; align-items: center;
    margin: 14px auto 20px; width: 100%; max-width: 360px; position: relative;
    background: radial-gradient(circle at center, rgba(193,166,115,.08) 0%, rgba(16,15,13,.4) 65%, transparent 100%);
    border-radius: 50%; padding: 8px;
  }
  #starChart {
    display: block; width: 100%; height: auto; max-width: 330px;
    aspect-ratio: 1 / 1; touch-action: none; filter: drop-shadow(0 4px 20px rgba(0,0,0,.65));
  }
  .spin-ring {
    display: inline-block; width: 13px; height: 13px; border: 2px solid rgba(255,255,255,.25);
    border-top-color: #D4B97A; border-radius: 50%; animation: spinRing .7s linear infinite;
    vertical-align: middle; margin-right: 8px;
  }
  @keyframes spinRing { to { transform: rotate(360deg); } }
  .gen-error-banner {
    margin-top: 12px; padding: 10px 14px; background: rgba(239,68,68,.12);
    border: 1px solid rgba(239,68,68,.35); border-radius: var(--radius-sm);
    color: #FCA5A5; font-size: 12px; line-height: 1.5; text-align: left;
  }
  body.dragging-knob, body.dragging-knob * { cursor: ns-resize !important; }
  body.dragging-star, body.dragging-star * { cursor: grabbing !important; }
  
  /* Toggles & Fields */
  .toggles { display: flex; flex-direction: column; gap: 10px; margin-top: 12px; }
  .toggle { display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--text-2); cursor: pointer; }
  .toggle input { accent-color: var(--gold); width: 15px; height: 15px; cursor: pointer; flex: none; }
  .toggle:hover { color: var(--text); }
  
  .section-label { font-size: 11px; text-transform: uppercase; letter-spacing: .12em; color: var(--text-2); margin: 22px 0 6px; border-top: 1px solid var(--line-soft); padding-top: 18px; }
  .section-label:first-of-type { border-top: none; padding-top: 0; margin-top: 0; }
  
  .layout { display: grid; grid-template-columns: 1.05fr 1fr 1.15fr; gap: 24px; align-items: stretch; }
  .layout .card { display: flex; flex-direction: column; }
  .layout .card-title { font-family: var(--display); font-style: italic; font-size: 22px; color: var(--text); margin: 0 0 4px; }
  .layout .card-hint { font-size: 12px; color: var(--text-2); margin: 0 0 20px; line-height: 1.4; }
  
  @media (max-width: 1280px) {
    .layout { grid-template-columns: 1fr 1fr; }
    .layout > .card:first-child { grid-column: 1 / -1; }
  }
  @media (max-width: 760px) {
    .layout { grid-template-columns: 1fr; }
    .layout > .card:first-child { grid-column: auto; }
  }
  
  select, input[type=text] {
    width: 100%; background: var(--bg-3); color: var(--text); border: 1px solid var(--line);
    border-radius: var(--radius-sm); padding: 10px 12px; font-family: var(--ui); font-size: 13px;
    font-weight: 300; margin-top: 7px; transition: border-color .2s var(--ease);
  }
  select:focus, input[type=text]:focus { outline: none; border-color: var(--gold-glow); }
  option { background: var(--bg-3); color: var(--text); }
  .field { margin-top: 18px; }
  .field:first-child { margin-top: 0; }
  .hint { font-size: 11px; color: var(--text-3); margin-top: 5px; line-height: 1.45; }
  .card.disabled { opacity: .4; pointer-events: none; }
  
  /* Canvas Studio */
  #exportCanvas {
    display: block; margin: 14px auto 0; max-width: 100%; max-height: 320px; width: auto; height: auto;
    border-radius: var(--radius-sm); border: 1px solid var(--line); background: #090807; box-shadow: 0 4px 20px rgba(0,0,0,.6);
  }
  
  /* Rail Step Flow */
  .rail { display: flex; align-items: center; margin: 0 0 34px; }
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
  
  .card-head { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
  .quota-pill {
    margin-left: auto; font-size: 11px; letter-spacing: .06em;
    color: var(--gold-text); background: rgba(193,166,115,.12);
    border: 1px solid rgba(193,166,115,.28); border-radius: 999px;
    padding: 4px 10px; display: inline-flex; align-items: center; gap: 5px;
    font-family: var(--ui); font-weight: 500; text-transform: uppercase;
  }
  .step-badge {
    width: 27px; height: 27px; border-radius: 50%; border: 1px solid var(--line-hi);
    display: grid; place-items: center; font-size: 10px; color: var(--text-3); flex: none;
    transition: color .5s var(--ease), border-color .5s var(--ease), background .5s var(--ease);
  }
  .card.is-active .step-badge { border-color: var(--gold); color: var(--gold-text); }
  .card.is-done .step-badge { background: linear-gradient(180deg, var(--gold), var(--gold-deep)); border-color: transparent; color: var(--on-gold); }
  .card.is-active { border-color: var(--gold-glow); }
  .card.unlocking { animation: unlockRise .75s var(--ease-out) both; overflow: hidden; }
  .card.unlocking::after {
    content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
    background: linear-gradient(105deg, transparent 32%, rgba(193,166,115,.13) 50%, transparent 68%);
    transform: translateX(-100%); animation: unlockSweep 1.15s var(--ease-out) .12s forwards;
  }
  @keyframes unlockRise { from { opacity: .35; transform: translateY(12px); } to { opacity: 1; transform: none; } }
  @keyframes unlockSweep { to { transform: translateX(100%); } }
  .card-actions { margin-top: auto; padding-top: 14px; }
  
  /* Output Bar */
  .outbar { margin-top: 24px; display: flex; align-items: center; gap: 26px; padding: 20px 24px; }
  .outbar-main { flex: 1 1 auto; min-width: 0; }
  .outbar-line { display: flex; align-items: center; gap: 12px; font-size: 12px; color: var(--text-2); letter-spacing: .02em; margin-bottom: 12px; }
  #statusLabel { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #statusPct { color: var(--gold-text); font-family: var(--display); font-style: italic; font-size: 15px; flex: none; }
  @media (max-width: 960px) { .outbar { flex-direction: column; align-items: stretch; gap: 18px; } }
  
  .eq { display: flex; align-items: flex-end; gap: 3px; height: 18px; flex: none; }
  .eq i { width: 2px; height: 30%; border-radius: 2px; background: var(--text-3); animation: eqIdle 3.2s var(--ease) infinite; }
  .eq i:nth-child(2) { animation-delay: .35s; }
  .eq i:nth-child(3) { animation-delay: .7s; }
  .eq i:nth-child(4) { animation-delay: 1.05s; }
  .eq i:nth-child(5) { animation-delay: 1.4s; }
  @keyframes eqIdle { 0%, 100% { height: 22%; } 50% { height: 72%; } }
  .outbar.busy .eq i { background: var(--gold); animation-duration: .95s; }
  
  #playerWrap { opacity: 0; transition: opacity .55s var(--ease); margin-top: 10px; }
  #playerWrap.ready { opacity: 1; }
  
  /* Player Utility Row */
  .player-tools { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 10px; flex-wrap: wrap; }
  .speed-group { display: flex; gap: 4px; align-items: center; }
  .speed-btn {
    background: var(--bg-3); border: 1px solid var(--line); border-radius: 4px;
    padding: 3px 8px; font-size: 10px; color: var(--text-2); cursor: pointer;
  }
  .speed-btn.active { border-color: var(--gold); color: var(--gold-text); background: var(--bg-4); }
  
  .chips { display: flex; gap: 10px; flex: none; flex-wrap: wrap; }
  .chip {
    display: inline-flex; align-items: center; gap: 9px; padding: 11px 16px;
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
  
  /* Artwork Drop Area */
  .art-drop {
    border: 1px dashed var(--line-hi); border-radius: var(--radius-sm); padding: 12px;
    text-align: center; cursor: pointer; transition: all .2s var(--ease); margin-top: 8px;
    background: var(--bg-3); display: flex; align-items: center; justify-content: center; gap: 10px;
  }
  .art-drop:hover { border-color: var(--gold); background: var(--bg-4); }
  .art-preview { width: 36px; height: 36px; border-radius: 6px; object-fit: cover; display: none; }
  
  .footer { margin-top: 48px; text-align: center; font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--text-3); }
  .footer .sep { opacity: .5; margin: 0 6px; }
  
  @media (prefers-reduced-motion: reduce) {
    .card.unlocking, .card.unlocking::after, .rail-node.active .rail-dot, .eq i { animation: none; }
    .rail-line::after { transition: none; }
  }

  /* ---- Luxury Custom Gold Cursor ---- */
  @media (hover: hover) and (pointer: fine) {
    body.has-custom-cursor,
    body.has-custom-cursor button,
    body.has-custom-cursor a,
    body.has-custom-cursor input,
    body.has-custom-cursor select,
    body.has-custom-cursor textarea,
    body.has-custom-cursor .knob,
    body.has-custom-cursor .tag-chip,
    body.has-custom-cursor .archetype-btn,
    body.has-custom-cursor .title-pill,
    body.has-custom-cursor .speed-btn {
      cursor: none !important;
    }
    #customCursorDot {
      position: fixed; top: 0; left: 0; width: 6px; height: 6px;
      background: #D4B97A; border-radius: 50%; pointer-events: none; z-index: 999999;
      transform: translate(-50%, -50%);
      box-shadow: 0 0 10px #D4B97A, 0 0 20px rgba(212,185,122,0.6);
      transition: transform 0.08s ease-out, opacity 0.2s ease, width 0.2s ease, height 0.2s ease;
    }
    #customCursorRing {
      position: fixed; top: 0; left: 0; width: 28px; height: 28px;
      border: 1.2px solid rgba(193,166,115,0.5); border-radius: 50%; pointer-events: none; z-index: 999998;
      transform: translate(-50%, -50%);
      box-shadow: 0 0 12px rgba(193,166,115,0.18);
      transition: width 0.22s var(--ease-out), height 0.22s var(--ease-out), border-color 0.22s var(--ease), background 0.22s var(--ease), opacity 0.2s ease;
    }
    #customCursorRing.hovering {
      width: 44px; height: 44px;
      border-color: #D4B97A;
      background: rgba(193,166,115,0.10);
      box-shadow: 0 0 22px rgba(193,166,115,0.35);
    }
    #customCursorRing.clicking {
      width: 20px; height: 20px;
      border-color: #FFF8E7;
      background: rgba(193,166,115,0.25);
    }
    #customCursorDot.clicking {
      transform: translate(-50%, -50%) scale(1.6);
    }
    #customCursorRing.dragging {
      width: 38px; height: 38px;
      border-color: #D4B97A;
      border-style: dashed;
      animation: cursorSpin 3s linear infinite;
    }
    @keyframes cursorSpin { to { transform: translate(-50%, -50%) rotate(360deg); } }
    .cursor-hidden #customCursorDot, .cursor-hidden #customCursorRing {
      opacity: 0;
    }
  }
  @media (hover: none), (pointer: coarse) {
    #customCursorDot, #customCursorRing { display: none !important; }
  }

  /* ---- Take Library Drawer ---- */
  .take-library-section {
    margin-top: 24px; padding: 20px; background: var(--bg-2);
    border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--lift);
  }
  .take-library-head { margin-bottom: 14px; }
  .take-tray {
    display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px;
    scrollbar-width: thin; scrollbar-color: var(--gold-deep) transparent;
  }
  .take-tray::-webkit-scrollbar { height: 6px; }
  .take-tray::-webkit-scrollbar-thumb { background: var(--gold-deep); border-radius: 4px; }
  .take-card {
    flex: 0 0 220px; background: var(--bg-3); border: 1px solid var(--line);
    border-radius: var(--radius-sm); padding: 12px; transition: all .2s var(--ease);
    display: flex; flex-direction: column; gap: 8px; cursor: pointer;
  }
  .take-card:hover, .take-card.active {
    border-color: var(--gold); background: var(--bg-4); transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0,0,0,.4);
  }
  .take-card-title {
    font-family: var(--display); font-size: 15px; font-weight: 500; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .take-card-meta {
    font-size: 11px; color: var(--text-2); display: flex; justify-content: space-between;
  }
  .take-card-actions { display: flex; gap: 6px; margin-top: 4px; }
  .take-btn-mini {
    flex: 1; padding: 4px 8px; font-size: 10.5px; border-radius: 6px;
    border: 1px solid var(--line); background: rgba(255,255,255,.05); color: var(--text);
    cursor: pointer; transition: all .15s var(--ease); text-align: center;
  }
  .take-btn-mini:hover { background: var(--gold); color: var(--on-gold); border-color: var(--gold); }

  /* ---- Interactive Waveform Timeline ---- */
  .waveform-box {
    margin-bottom: 14px; background: var(--bg-3); border: 1px solid var(--line);
    border-radius: var(--radius-sm); padding: 12px 14px; position: relative;
  }
  .waveform-meta {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
    font-size: 11px; font-family: var(--ui);
  }
  .waveform-time { color: var(--text); font-weight: 500; }
  .waveform-loop-tag {
    color: var(--gold-text); background: rgba(193,166,115,.12); border: 1px solid rgba(193,166,115,.25);
    padding: 2px 8px; border-radius: 999px; font-size: 10px; letter-spacing: .04em;
  }
  .waveform-track {
    position: relative; width: 100%; height: 60px; background: rgba(0,0,0,.4);
    border-radius: 6px; cursor: pointer; overflow: hidden; user-select: none;
  }
  #waveformCanvas { width: 100%; height: 100%; display: block; }
  .waveform-playhead {
    position: absolute; top: 0; bottom: 0; width: 2px; background: #fff;
    box-shadow: 0 0 8px #fff; pointer-events: none; transform: translateX(-50%);
    left: 0%; transition: left .05s linear;
  }
  .loop-bracket {
    position: absolute; top: 0; bottom: 0; width: 14px; cursor: ew-resize;
    display: flex; align-items: center; justify-content: center; z-index: 10;
    touch-action: none;
  }
  .loop-bracket span {
    font-size: 9px; font-weight: 700; color: #090807; background: var(--gold);
    width: 13px; height: 16px; border-radius: 3px; display: flex; align-items: center;
    justify-content: center; box-shadow: 0 0 6px var(--gold-glow);
  }
  .loop-bracket.handle-a { left: 0%; transform: translateX(0%); }
  .loop-bracket.handle-b { left: 100%; transform: translateX(-100%); }
  .loop-highlight {
    position: absolute; top: 0; bottom: 0; left: 0%; width: 100%;
    background: rgba(193,166,115,.08); pointer-events: none; border-left: 1px solid var(--gold);
    border-right: 1px solid var(--gold);
  }

  /* ---- 4-Channel Soundscape Mixer Deck ---- */
  .mixer-console {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 12px 0 16px;
  }
  .mixer-strip {
    background: var(--bg-3); border: 1px solid var(--line); border-radius: var(--radius-sm);
    padding: 12px 8px; display: flex; flex-direction: column; align-items: center; gap: 8px;
    transition: all .2s var(--ease);
  }
  .mixer-strip:hover { border-color: rgba(193,166,115,.3); }
  .mixer-strip.muted { opacity: .45; }
  .mixer-strip.soloed { border-color: var(--gold); box-shadow: 0 0 12px var(--gold-glow); }
  .strip-name {
    font-size: 11px; font-weight: 500; letter-spacing: .04em; color: var(--text);
    white-space: nowrap; text-align: center;
  }
  .strip-btns { display: flex; gap: 4px; }
  .strip-btn {
    width: 22px; height: 20px; font-size: 9.5px; font-weight: 700; border-radius: 4px;
    border: 1px solid var(--line); background: rgba(255,255,255,.05); color: var(--text-2);
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: all .15s var(--ease);
  }
  .strip-btn:hover { color: var(--text); background: rgba(255,255,255,.1); }
  .strip-btn.mute-active { background: #EF4444; color: #fff; border-color: #EF4444; }
  .strip-btn.solo-active { background: var(--gold); color: #090807; border-color: var(--gold); }
  .strip-fader-wrap {
    width: 100%; display: flex; justify-content: center; padding: 6px 0;
  }
  .strip-fader {
    width: 90%; accent-color: var(--gold); cursor: pointer;
  }
  .strip-val { font-size: 10.5px; color: var(--gold-text); font-weight: 500; font-family: var(--ui); }

</style>
</head>
<body>
<div id="customCursorDot"></div>
<div id="customCursorRing"></div>
<canvas id="featherCanvas"></canvas>
<div class="wrap">
  <div class="brand-wrap">
    <div class="brand">
      <div class="ring"></div>
      <div class="brand-title">
        <span>Tranquil Soul Music</span>
        <span>Tranquilicy Creative Studio</span>
      </div>
    </div>
    <div class="gpu-badge" id="gpuBadge" onclick="openGpuModal()">
      <span class="gpu-dot"></span>
      <span id="gpuLabel">Local GPU: Checking...</span>
    </div>
  </div>

  <div id="gpuModal" class="modal-overlay" style="display:none">
    <div class="modal-card">
      <div class="card-head" style="margin-bottom:12px">
        <span class="step-badge">GPU</span>
        <div class="card-title">Local RTX 3090 Pipeline</div>
      </div>
      <div class="hint" style="margin-bottom:18px">
        Tranquilicy runs directly on your local NVIDIA GeForce RTX 3090 GPU (24GB VRAM) with zero cloud GPU fees.
      </div>
      <div class="field">
        <label>Backend GPU Server URL</label>
        <input type="text" id="gpuEndpointInput" placeholder="http://localhost:8000 or Cloudflare Tunnel URL">
        <div class="hint" id="gpuStatusText">Testing connection to local backend...</div>
      </div>
      <div style="display:flex; gap:10px; margin-top:24px">
        <button type="button" class="btn-primary" id="gpuSaveBtn" style="margin:0" onclick="saveGpuEndpoint()">Save & Test</button>
        <button type="button" class="btn-ghost" id="gpuCloseBtn" style="margin:0" onclick="closeGpuModal()">Close</button>
      </div>
    </div>
  </div>
  <h1>Create a track</h1>
  <div class="sub">Craft your sonic atmosphere, sculpt the tone, and render social-ready visualizer videos.</div>

  <div class="rail">
    <div class="rail-node" id="railStep1"><span class="rail-dot">01</span><span class="rail-label">Generate</span></div>
    <div class="rail-line" id="railLine1"></div>
    <div class="rail-node" id="railStep2"><span class="rail-dot">02</span><span class="rail-label">Master</span></div>
    <div class="rail-line delayed" id="railLine2"></div>
    <div class="rail-node" id="railStep3"><span class="rail-dot">03</span><span class="rail-label">Video</span></div>
  </div>

  <div class="layout">
  <!-- STEP 1: GENERATE & PROMPTER -->
  <div class="card" id="generateCard">
    <div class="card-head">
      <span class="step-badge">01</span>
      <div class="card-title">Generate</div>
      <div class="quota-pill" id="quotaPill">⚡ <span id="quotaText">5/5 Generations Left</span></div>
    </div>
    <div class="card-hint">Select an archetype or compose your vision using the prompter matrix.</div>

    <label style="margin:0 0 6px">Vibe Archetypes</label>
    <div class="archetype-shelf" id="promptArchetypes">
      <button type="button" class="archetype-btn" onclick="applyArchetype('zen')">🧘 Deep Zen</button>
      <button type="button" class="archetype-btn" onclick="applyArchetype('lofi')">☕ Midnight Lofi</button>
      <button type="button" class="archetype-btn" onclick="applyArchetype('sunset')">🌅 Sunset Chill</button>
      <button type="button" class="archetype-btn" onclick="applyArchetype('celestial')">✨ Celestial</button>
      <button type="button" class="archetype-btn" onclick="applyArchetype('rain')">🌧 Coffee Rain</button>
    </div>

    <div class="row" style="margin-top:10px">
      <label style="margin:0">Prompt Composition</label>
      <div class="action-row-mini">
        <button type="button" class="link-btn" id="alchemistBtn" onclick="alchemistPrompt()">✧ Alchemist</button>
        <button type="button" class="link-btn" id="randomiseBtn" onclick="randomisePrompt()">⟳ Roll</button>
        <button type="button" class="link-btn" id="copyPromptBtn" onclick="copyPrompt()">📋</button>
        <button type="button" class="link-btn" id="clearPromptBtn" onclick="clearPrompt()">✕</button>
      </div>
    </div>
    <textarea id="prompt" placeholder="Describe the atmosphere, textures, instruments... or click any tag below"></textarea>

    <div class="vibe-section">
      <div class="vibe-header">
        <label style="margin:0">Vibe Matrix</label>
        <div class="vibe-nav">
          <button type="button" class="vibe-nav-btn active" id="tabGenre" onclick="switchVibeTab('genre')">Genre</button>
          <button type="button" class="vibe-nav-btn" id="tabTexture" onclick="switchVibeTab('texture')">Texture</button>
          <button type="button" class="vibe-nav-btn" id="tabMood" onclick="switchVibeTab('mood')">Mood</button>
          <button type="button" class="vibe-nav-btn" id="tabMusical" onclick="switchVibeTab('musical')">Tempo & Key</button>
        </div>
      </div>
      <div class="tag-tray" id="vibeTagTray"></div>
    </div>

    <div class="section-label">Sound Dimensions</div>
    <div class="dials-grid" id="dialsGrid"></div>
    <div class="star-wrap">
      <svg id="starChart" width="330" height="330" viewBox="0 0 330 330"></svg>
    </div>

    <div class="section-label">Negative Guidance</div>
    <div class="toggles">
      <label class="toggle"><input type="checkbox" id="noDrums"> No drums / percussion</label>
      <label class="toggle"><input type="checkbox" id="noVocals"> No vocals</label>
      <label class="toggle"><input type="checkbox" id="noBass"> No bass</label>
      <label class="toggle"><input type="checkbox" id="noHarshHighs"> No harsh highs / bright synths</label>
    </div>
    <div class="hint">Guided away via unconditional branch steering.</div>

    <div class="row">
      <label style="margin:0">Duration</label>
      <span class="val"><span id="durVal">20</span>s</span>
    </div>
    <input type="range" id="duration" min="5" max="180" value="20">

    <div class="card-actions">
      <button id="genBtn" class="btn-primary" onclick="generate()">Generate Track</button>
      <div id="genErrorNotice" class="gen-error-banner" style="display:none"></div>
    </div>
  </div>

  <!-- STEP 2: MASTER & AUDIO STUDIO -->
  <div class="card disabled" id="audioCard" inert aria-hidden="true">
    <div class="card-head">
      <span class="step-badge">02</span>
      <div class="card-title">Master & Audio</div>
    </div>
    <div class="card-hint">Curate title, sculpt tone, inject ambient texture, and export mix.</div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Track Title</label>
        <div class="action-row-mini">
          <button type="button" class="link-btn" id="roll5Btn" onclick="generateBatchTitles()">✨ Roll 5</button>
          <button type="button" class="link-btn" onclick="rerollTitle()">🎲 Re-roll</button>
        </div>
      </div>
      <input type="text" id="trackTitle" placeholder="Untitled Chillout Track">
      <div class="title-shelf" id="titlePillTray"></div>
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Theme Universe</label>
        <label style="margin:0">Formula Style</label>
      </div>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
        <select id="titleUniverse" onchange="rerollTitle()">
          <option value="all">All Universes</option>
          <option value="celestial">Celestial & Cosmic</option>
          <option value="tideline">Tideline & Ocean</option>
          <option value="obsidian">Obsidian & Midnight</option>
          <option value="zen">Zen & Sanctuary</option>
          <option value="organic">Warm & Organic</option>
        </select>
        <select id="titleFormula" onchange="rerollTitle()">
          <option value="random">Auto Formula</option>
          <option value="adj_noun">Adj + Noun</option>
          <option value="noun_of_noun">The Noun of Noun</option>
          <option value="verbing">Verb-ing Through</option>
          <option value="japanese">Japanese Aesthetic</option>
          <option value="opus">Opus & Catalog</option>
        </select>
      </div>
    </div>

    <div class="field">
      <label>Filename Style</label>
      <select id="namePattern">
        <option value="plain">Title only</option>
        <option value="numbered-dash">01 - Title</option>
        <option value="numbered-dot">01. Title</option>
        <option value="extended">Title (Extended Mix)</option>
        <option value="instrumental">Title (Instrumental Mix)</option>
        <option value="slowed">Title [Slowed + Reverb]</option>
      </select>
    </div>

    <div class="section-label">4-Channel Soundscape Mixer Deck</div>
    <div class="mixer-console" id="mixerConsole">
      <!-- Channel 1: Music -->
      <div class="mixer-strip" id="stripMusic">
        <span class="strip-name">🎵 Music</span>
        <div class="strip-btns">
          <button type="button" class="strip-btn" id="muteMusicBtn" onclick="toggleMute('music')">M</button>
          <button type="button" class="strip-btn" id="soloMusicBtn" onclick="toggleSolo('music')">S</button>
        </div>
        <div class="strip-fader-wrap">
          <input type="range" class="strip-fader" id="volMusic" min="0" max="120" value="100" oninput="updateMixerVolumes()">
        </div>
        <div class="strip-val"><span id="volMusicVal">100</span>%</div>
      </div>

      <!-- Channel 2: Rain -->
      <div class="mixer-strip" id="stripRain">
        <span class="strip-name">🌧 Rain</span>
        <div class="strip-btns">
          <button type="button" class="strip-btn" id="muteRainBtn" onclick="toggleMute('rain')">M</button>
          <button type="button" class="strip-btn" id="soloRainBtn" onclick="toggleSolo('rain')">S</button>
        </div>
        <div class="strip-fader-wrap">
          <input type="range" class="strip-fader" id="volRain" min="0" max="100" value="0" oninput="updateMixerVolumes()">
        </div>
        <div class="strip-val"><span id="volRainVal">0</span>%</div>
      </div>

      <!-- Channel 3: Vinyl -->
      <div class="mixer-strip" id="stripVinyl">
        <span class="strip-name">📻 Vinyl</span>
        <div class="strip-btns">
          <button type="button" class="strip-btn" id="muteVinylBtn" onclick="toggleMute('vinyl')">M</button>
          <button type="button" class="strip-btn" id="soloVinylBtn" onclick="toggleSolo('vinyl')">S</button>
        </div>
        <div class="strip-fader-wrap">
          <input type="range" class="strip-fader" id="volVinyl" min="0" max="100" value="0" oninput="updateMixerVolumes()">
        </div>
        <div class="strip-val"><span id="volVinylVal">0</span>%</div>
      </div>

      <!-- Channel 4: Binaural Drone -->
      <div class="mixer-strip" id="stripBinaural">
        <span class="strip-name">🔮 Drone</span>
        <div class="strip-btns">
          <button type="button" class="strip-btn" id="muteBinauralBtn" onclick="toggleMute('binaural')">M</button>
          <button type="button" class="strip-btn" id="soloBinauralBtn" onclick="toggleSolo('binaural')">S</button>
        </div>
        <div class="strip-fader-wrap">
          <input type="range" class="strip-fader" id="volBinaural" min="0" max="100" value="0" oninput="updateMixerVolumes()">
        </div>
        <div class="strip-val"><span id="volBinauralVal">0</span>%</div>
      </div>
    </div>

    <!-- Binaural / Solfeggio Preset Selector -->
    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Binaural Frequency Mode</label>
        <span class="val" id="binauralFreqReadout">Alpha (10 Hz Focus)</span>
      </div>
      <select id="binauralPreset" onchange="updateBinauralPreset()">
        <option value="alpha" selected>Alpha Waves (10 Hz · Flow State & Focus)</option>
        <option value="theta">Theta Waves (6 Hz · Deep Meditation & Zen)</option>
        <option value="delta">Delta Waves (2.5 Hz · Restorative REM Sleep)</option>
        <option value="solfeggio432">Solfeggio 432 Hz (Earth Harmony Resonance)</option>
        <option value="solfeggio528">Solfeggio 528 Hz (Transformation Tone)</option>
      </select>
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Analog Warmth (Bass)</label>
        <span class="val"><span id="warmthVal">0</span> dB</span>
      </div>
      <input type="range" id="toneWarmth" min="-6" max="8" step="0.5" value="0" oninput="updateToneEq()">
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Air & Sheen (Highs)</label>
        <span class="val"><span id="airVal">0</span> dB</span>
      </div>
      <input type="range" id="toneAir" min="-6" max="8" step="0.5" value="0" oninput="updateToneEq()">
    </div>

    <div class="field">
      <label class="toggle">
        <input type="checkbox" id="lofiFilter" onchange="updateToneEq()"> Vintage Tape / Lo-Fi Cassette Filter
      </label>
    </div>

    <div class="section-label">Mastering & Spatialization</div>
    <div class="field">
      <label>Loudness Target</label>
      <select id="masterPreset">
        <option value="off">Off — raw generation</option>
        <option value="gentle">Gentle (-16 dBFS)</option>
        <option value="streaming" selected>Streaming-ready (-14 dBFS)</option>
        <option value="loud">Punchy / Loud (-9 dBFS)</option>
      </select>
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Stereo Width</label>
        <span class="val"><span id="widthVal">0</span></span>
      </div>
      <input type="range" id="stereoWidth" min="0" max="100" value="0">
      <div class="hint">Mono source, widened mono-safely without phase cancel.</div>
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Master Tape Saturation</label>
        <span class="val"><span id="masterWarmthVal">0</span>%</span>
      </div>
      <input type="range" id="masterWarmth" min="0" max="100" value="0">
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Master High Air Lift</label>
        <span class="val"><span id="masterAirVal">0</span>%</span>
      </div>
      <input type="range" id="masterAir" min="0" max="100" value="0">
    </div>

    <div class="section-label">Loop & Envelope</div>
    <label class="toggle" style="margin-top:10px">
      <input type="checkbox" id="seamlessLoop"> Seamless loop (replaces fades)
    </label>

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
      <div class="section-label">Export Audio</div>
      <div class="field">
        <label>Format</label>
        <select id="exportFormat">
          <option value="WAV" selected>WAV — lossless, largest</option><option value="FLAC">FLAC — lossless, ~40% smaller</option><option value="OGG">OGG Vorbis — small</option><option value="MP3">MP3 — smallest, most portable</option>
        </select>
      </div>
      <button class="btn-primary" id="masterBtn" onclick="downloadMastered()">Master & Export Audio</button>
      <div id="masterStatus" class="hint" style="display:none; text-align:center;"></div>
    </div>
  </div>

  <!-- STEP 3: VIDEO & VISUALIZER STUDIO -->
  <div class="card disabled" id="videoCard" inert aria-hidden="true">
    <div class="card-head">
      <span class="step-badge">03</span>
      <div class="card-title">Video Studio</div>
    </div>
    <div class="card-hint">Render luxury audio-reactive videos with custom artwork & typography.</div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Cover Artwork</label>
        <button type="button" class="link-btn" id="removeArtBtn" style="display:none" onclick="removeCoverArt()">Remove Art</button>
      </div>
      <div class="art-drop" onclick="document.getElementById('artUpload').click()">
        <img id="artThumb" class="art-preview" alt="Cover preview">
        <span id="artStatus">Click or drop custom album art (JPG/PNG)</span>
      </div>
      <input type="file" id="artUpload" accept="image/*" style="display:none" onchange="handleCoverUpload(event)">
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Waveform Engine</label>
        <label style="margin:0">Color Palette</label>
      </div>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
        <select id="vidStyle">
          <option value="bars">Mirrored Bars</option>
          <option value="wave" selected>Wave Ribbon</option>
          <option value="radial">Radial 360° Halo</option>
          <option value="pulse">Sacred Lotus Pulse</option>
          <option value="stardust">Cosmic Stardust</option>
          <option value="lissajous">Stereo Lissajous</option>
          <option value="off">Off (Artwork Only)</option>
        </select>
        <select id="vidPalette">
          <option value="gold">Gold & Champagne</option>
          <option value="ember">Warm Ember</option>
          <option value="moonlit">Moonlit Ethereal</option>
          <option value="amethyst">Royal Obsidian</option>
          <option value="emerald">Zen Jade</option>
        </select>
      </div>
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Backdrop</label>
        <label style="margin:0">Aspect Ratio</label>
      </div>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
        <select id="vidBackdrop">
          <option value="feathers">Feather Drift</option>
          <option value="bloom">Gold Bloom</option>
          <option value="nebula">Cosmic Starfield</option>
          <option value="custom">Custom Cover Art</option>
          <option value="minimal">Minimal Dark</option>
        </select>
        <select id="vidAspect">
          <option value="16:9">16:9 — YouTube Landscape</option>
          <option value="9:16">9:16 — TikTok / Reels / Shorts</option>
          <option value="1:1">1:1 — Square Post / Canvas</option>
          <option value="4:5">4:5 — Instagram Feed Portrait</option>
        </select>
      </div>
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Backdrop Blur</label>
        <span class="val"><span id="artBlurVal">12</span>px</span>
      </div>
      <input type="range" id="artBlur" min="0" max="30" value="12" oninput="document.getElementById('artBlurVal').textContent = this.value; refreshPreview();">
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Backdrop Dimming</label>
        <span class="val"><span id="artDimVal">50</span>%</span>
      </div>
      <input type="range" id="artDim" min="0" max="90" value="50" oninput="document.getElementById('artDimVal').textContent = this.value; refreshPreview();">
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Bass Camera Breathe</label>
        <span class="val"><span id="beatPulseVal">40</span>%</span>
      </div>
      <input type="range" id="beatPulse" min="0" max="100" value="40" oninput="document.getElementById('beatPulseVal').textContent = this.value; refreshPreview();">
      <div class="hint">Subtle zoom pulse reactive to low-end transients.</div>
    </div>

    <div class="section-label">Typography & Overlays</div>
    <div class="field">
      <label>Artist / Subtitle</label>
      <input type="text" id="trackArtist" value="Tranquil Soul Music" placeholder="Artist Name or Catalog ID">
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Display Font</label>
        <label style="margin:0">Watermark</label>
      </div>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
        <select id="vidFont">
          <option value="Cormorant Garamond">Cormorant Garamond (Serif)</option>
          <option value="Cinzel">Cinzel (Cinematic)</option>
          <option value="Montserrat">Montserrat (Modern Clean)</option>
          <option value="Space Grotesk">Space Grotesk (Neo-Minimal)</option>
        </select>
        <select id="vidWatermark">
          <option value="wordmark">Tranquilicy Wordmark</option>
          <option value="ring">Ring Emblem Only</option>
          <option value="custom">Custom Text</option>
          <option value="none">None</option>
        </select>
      </div>
    </div>

    <div class="field" id="customWatermarkField" style="display:none">
      <label>Custom Watermark Text</label>
      <input type="text" id="customWatermarkText" value="TRANQUILICY">
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Video Export Length</label>
        <span class="val" id="vidLoopEstVal">1x Loop</span>
      </div>
      <select id="vidLoops" onchange="updateVidLoopEst()">
        <option value="1" selected>1x · Original Take (~20–60s)</option>
        <option value="3">3x · Social Reel / Shorts Cut (~1–2 mins)</option>
        <option value="5">5x · Mini Study Loop (~3–5 mins)</option>
        <option value="10">10x · YouTube Extended Cut (~8–15 mins)</option>
      </select>
    </div>

    <div class="section-label">Cinematic Video Overlays</div>
    <div class="toggles">
      <label class="toggle"><input type="checkbox" id="overlayEmbers" checked onchange="refreshPreview()"> Floating Stardust Embers</label>
      <label class="toggle"><input type="checkbox" id="overlayTitleCard" checked onchange="refreshPreview()"> Intro Cinematic Title Card</label>
      <label class="toggle"><input type="checkbox" id="overlayVignette" checked onchange="refreshPreview()"> Film Grain & Vignette Border</label>
    </div>

    <div class="field">
      <label>Watermark Position</label>
      <select id="watermarkPos">
        <option value="br">Bottom Right</option>
        <option value="tr">Top Right</option>
        <option value="bl">Bottom Left</option>
        <option value="tl">Top Left</option>
        <option value="bc">Bottom Center</option>
      </select>
    </div>

    <div class="toggles" style="margin-top:14px">
      <label class="toggle"><input type="checkbox" id="showTitleCheck" checked> Display Track Title</label>
      <label class="toggle"><input type="checkbox" id="showArtistCheck" checked> Display Artist / Subtitle</label>
      <label class="toggle"><input type="checkbox" id="showTimecodeCheck" checked> Display Timecode & Duration</label>
      <label class="toggle"><input type="checkbox" id="showProgressCheck" checked> Bottom Progress Scrubber Bar</label>
      <label class="toggle"><input type="checkbox" id="showVinylCheck" checked> Center Vinyl Artwork Disc</label>
    </div>

    <div class="field">
      <div class="row" style="margin-top:0">
        <label style="margin:0">Render Length</label>
        <label style="margin:0">Video Quality</label>
      </div>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
        <select id="vidLength">
          <option value="full">Full Track</option>
          <option value="loop15">15s Social Loop</option>
          <option value="loop30">30s Social Loop</option>
        </select>
        <select id="vidQuality">
          <option value="ultra">Ultra (8 Mbps, 60 FPS)</option>
          <option value="high" selected>High (5 Mbps, 30 FPS)</option>
          <option value="fast">Compact (2.5 Mbps, 30 FPS)</option>
        </select>
      </div>
    </div>

    <canvas id="exportCanvas"></canvas>

    <div class="card-actions">
      <button class="btn-primary" id="renderVidBtn" onclick="renderVideo()">Render High-Res Video</button>
      <div id="vidStatus" class="hint" style="display:none; text-align:center; margin-top:10px;"></div>
      <button class="btn-ghost" id="stillBtn" onclick="saveStillFrame()">Capture Still Frame</button>
    </div>
  </div>
  </div>

  <!-- SESSION TAKES LIBRARY TRAY -->
  <div class="take-library-section" id="takeLibrarySection" style="display:none">
    <div class="take-library-head">
      <div class="row" style="margin:0; align-items:center;">
        <label style="margin:0; font-family:var(--display); font-size:17px; color:var(--gold-text); letter-spacing:.04em;">Session Takes Library (<span id="takeCount">0</span>)</label>
        <span class="hint" style="margin:0">Instant 1-click A/B recall & downloads</span>
      </div>
    </div>
    <div class="take-tray" id="takeTray"></div>
  </div>

  <!-- OUTPUT CONTROL & ARTIFACT BAR -->
  <div class="card outbar" id="outbar">
    <div class="outbar-main">
      <div class="outbar-line">
        <span class="eq"><i></i><i></i><i></i><i></i><i></i></span>
        <span id="statusLabel">No track yet — hit Generate</span>
        <button type="button" class="link-btn" id="cancelBtn" style="display:none" onclick="cancelGeneration()">Cancel</button>
        <span id="statusPct"></span>
      </div>
      <div id="barOuter"><div id="barInner"></div></div>
      <div id="playerWrap" hidden>
        <!-- Interactive Waveform Timeline with A/B Loop Markers -->
        <div class="waveform-box" id="waveformBox">
          <div class="waveform-meta">
            <span class="waveform-time"><span id="waveCurrentTime">0:00</span> / <span id="waveTotalTime">0:20</span></span>
            <span class="waveform-loop-tag" id="loopTag">A-B Loop: <span id="loopAVal">0.0s</span> - <span id="loopBVal">20.0s</span></span>
          </div>
          <div class="waveform-track" id="waveformTrack" onclick="handleWaveformClick(event)">
            <canvas id="waveformCanvas" width="800" height="60"></canvas>
            <div class="waveform-playhead" id="waveformPlayhead"></div>
            <div class="loop-bracket handle-a" id="loopHandleA" title="Loop Start (A)"><span>A</span></div>
            <div class="loop-bracket handle-b" id="loopHandleB" title="Loop End (B)"><span>B</span></div>
            <div class="loop-highlight" id="loopHighlight"></div>
          </div>
        </div>
        <audio id="player" controls></audio>
        <div class="player-tools">
          <div class="speed-group">
            <span style="font-size:10px; text-transform:uppercase; color:var(--text-3); margin-right:4px;">Speed:</span>
            <button type="button" class="speed-btn" id="speed075Btn" onclick="setSpeed(0.75)">0.75x</button>
            <button type="button" class="speed-btn active" id="speed100Btn" onclick="setSpeed(1.0)">1.0x</button>
            <button type="button" class="speed-btn" id="speed125Btn" onclick="setSpeed(1.25)">1.25x</button>
          </div>
          <button type="button" class="speed-btn" id="playerLoopBtn" onclick="togglePlayerLoop()">🔁 Loop: Off</button>
        </div>
      </div>
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
    Tranquil Soul Music <span class="sep">·</span> Tranquilicy Studio <span class="sep">·</span> v1.7.0
  </div>
</div>

<script>
document.getElementById('duration').oninput = e => document.getElementById('durVal').textContent = e.target.value;

let lastAudioUrl = null;
let lastVideoUrl = null;
let lastMasterUrl = null;
let lastStillUrl = null;
let currentJobId = null;

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

// ---- Falling Feathers Background ----
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

(function () {
  const canvas = document.getElementById('featherCanvas');
  const ctx = canvas.getContext('2d');
  let dpr = Math.min(window.devicePixelRatio || 1, 2);
  let w = 0, h = 0, particles = [];

  function featherCount() {
    return Math.round(Math.max(14, Math.min(34, (window.innerWidth * window.innerHeight) / 45000)));
  }

  function makeParticle(spawnAbove) {
    const size = 8 + Math.random() * 10;
    return {
      x: Math.random() * w,
      y: spawnAbove ? -20 - Math.random() * h : Math.random() * h,
      size,
      speedY: 10 + Math.random() * 14,
      swayAmp: 18 + Math.random() * 26,
      swayFreq: 0.15 + Math.random() * 0.25,
      phase: Math.random() * Math.PI * 2,
      baseX: 0,
      rot: Math.random() * Math.PI * 2,
      rotSpeed: (Math.random() - 0.5) * 0.8,
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
    particles = Array.from({ length: featherCount() }, () => {
      const p = makeParticle(false);
      p.baseX = p.x;
      return p;
    });
  }
  window.addEventListener('resize', resize);
  resize();

  let last = performance.now();
  function loop(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    ctx.clearRect(0, 0, w, h);
    const t = now / 1000;
    for (const p of particles) {
      p.y += p.speedY * dt;
      p.rot += p.rotSpeed * dt;
      if (p.y - p.size * 1.5 > h) {
        p.y = -p.size * 1.5;
        p.baseX = Math.random() * w;
        p.phase = Math.random() * Math.PI * 2;
      }
      p.x = p.baseX + Math.sin(t * p.swayFreq * Math.PI * 2 + p.phase) * p.swayAmp;
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.globalAlpha = p.opacity;
      drawFeatherShape(ctx, p.size, '#D4B97A', '#C1A673', 'rgba(9,8,7,.3)');
      ctx.restore();
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
})();

// ---- Sonic Dials & Star Chart ----
const DIALS = [
  { key: 'warmth',  label: 'Warmth',  default: 72, describe: v => v > 65 ? 'warm, rich low-end' : (v < 35 ? 'airy, lean' : 'balanced tone') },
  { key: 'reverb',  label: 'Reverb',  default: 60, describe: v => v > 65 ? 'cavernous, deep reverb space' : (v < 35 ? 'dry, intimate' : 'moderate hall reverb') },
  { key: 'tempo',   label: 'Tempo',   default: 45, describe: v => v > 65 ? 'gentle pulse, downtempo' : (v < 35 ? 'slow, floating tempo' : 'unhurried pace') },
  { key: 'lofi',    label: 'Lo-fi',   default: 40, describe: v => v > 65 ? 'dusty tape flutter, vinyl warmth' : (v < 35 ? 'clean, modern fidelity' : 'subtle analog patina') },
  { key: 'melodic', label: 'Melodic', default: 68, describe: v => v > 65 ? 'lyrical, expressive melody' : (v < 35 ? 'drone, minimal progression' : 'gentle motif') },
  { key: 'density', label: 'Density', default: 50, describe: v => v > 65 ? 'lush, layered instrumentation' : (v < 35 ? 'sparse, breathing room' : 'balanced arrangement') },
];

const knobs = {};
function buildDials() {
  const grid = document.getElementById('dialsGrid');
  grid.innerHTML = '';
  DIALS.forEach(d => {
    const wrap = document.createElement('div');
    wrap.className = 'dial-wrap';
    wrap.innerHTML = `
      <div class="knob" id="knob_${d.key}" tabindex="0" role="slider" aria-label="${d.label}" aria-valuenow="${d.default}" aria-valuemin="0" aria-valuemax="100"></div>
      <div class="dial-label">${d.label}</div>
      <div class="dial-value" id="val_${d.key}">${d.default}</div>
    `;
    grid.appendChild(wrap);
    const knobEl = wrap.querySelector('.knob');
    const valEl = wrap.querySelector('.dial-value');
    const state = {
      value: d.default,
      el: knobEl,
      valEl,
      set(v) {
        state.value = Math.max(0, Math.min(100, Math.round(v)));
        knobEl.style.setProperty('--rot', (-135 + (state.value / 100) * 270) + 'deg');
        knobEl.setAttribute('aria-valuenow', state.value);
        valEl.textContent = state.value;
      }
    };
    knobs[d.key] = state;
    state.set(d.default);

    let startY = 0, startVal = 0;
    knobEl.addEventListener('pointerdown', e => {
      startY = e.clientY;
      startVal = state.value;
      knobEl.classList.add('dragging');
      document.body.classList.add('dragging-knob');
      knobEl.setPointerCapture(e.pointerId);
    });
    knobEl.addEventListener('pointermove', e => {
      if (!knobEl.classList.contains('dragging')) return;
      const dy = startY - e.clientY;
      state.set(startVal + dy * 0.75);
      drawStarChart();
    });
    const stop = () => {
      if (knobEl.classList.contains('dragging')) {
        knobEl.classList.remove('dragging');
        document.body.classList.remove('dragging-knob');
      }
    };
    knobEl.addEventListener('pointerup', stop);
    knobEl.addEventListener('pointercancel', stop);
  });
}

// ---- Star Chart Drag: grab a vertex directly (Enlarged 330x330) ----
let draggingDialKey = null, hoverDialKey = null;
function setupStarChartDrag() {
  const svg = document.getElementById('starChart');
  const cx = 165, cy = 165, maxR = 112;
  const n = DIALS.length;
  const angleFor = i => -Math.PI / 2 + i * (2 * Math.PI / n);

  function svgPoint(e) {
    const rect = svg.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (330 / rect.width),
      y: (e.clientY - rect.top) * (330 / rect.height),
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
    const proj = (p.x - cx) * Math.cos(a) + (p.y - cy) * Math.sin(a);
    return Math.max(0, Math.min(100, Math.round((proj / maxR) * 100)));
  }

  const GRAB_RADIUS = 28;

  svg.addEventListener('pointerdown', e => {
    const p = svgPoint(e);
    const nearest = nearestVertex(p);
    if (!nearest || nearest.dist > GRAB_RADIUS) return;
    draggingDialKey = nearest.key;
    svg.setPointerCapture(e.pointerId);
    document.body.classList.add('dragging-star');
    knobs[nearest.key].set(valueFromPointer(p, nearest.i));
    drawStarChart();
    e.preventDefault();
  });

  svg.addEventListener('pointermove', e => {
    const p = svgPoint(e);
    if (!draggingDialKey) {
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
    knobs[draggingDialKey].set(valueFromPointer(p, i));
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

function drawStarChart() {
  const svg = document.getElementById('starChart');
  const cx = 165, cy = 165, maxR = 112;
  const n = DIALS.length;
  const angleFor = i => -Math.PI / 2 + i * (2 * Math.PI / n);
  const pt = (i, frac) => {
    const a = angleFor(i);
    return [cx + Math.cos(a) * maxR * frac, cy + Math.sin(a) * maxR * frac];
  };
  let svgHtml = '';
  // Concentric reference rings
  [0.33, 0.66, 1].forEach((frac, idx) => {
    const ring = DIALS.map((_, i) => pt(i, frac).join(',')).join(' ');
    const strokeDash = idx === 2 ? 'none' : '2,3';
    svgHtml += `<polygon points="${ring}" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="1" stroke-dasharray="${strokeDash}"/>`;
  });
  // Radial spokes
  DIALS.forEach((_, i) => {
    const [x, y] = pt(i, 1);
    svgHtml += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="rgba(255,255,255,.08)" stroke-width="1"/>`;
  });
  // Current active value polygon
  const valuePts = DIALS.map((d, i) => pt(i, knobs[d.key].value / 100).join(',')).join(' ');
  svgHtml += `<polygon points="${valuePts}" fill="rgba(193,166,115,.20)" stroke="#C1A673" stroke-width="1.8"/>`;
  
  // Handles and labels
  DIALS.forEach((d, i) => {
    const val = knobs[d.key].value;
    const [x, y] = pt(i, val / 100);
    const active = draggingDialKey === d.key || hoverDialKey === d.key;
    if (active) {
      svgHtml += `<circle cx="${x}" cy="${y}" r="15" fill="rgba(193,166,115,.35)"/>`;
    }
    svgHtml += `<circle cx="${x}" cy="${y}" r="${active ? 7 : 5}" fill="${active ? '#F2EFE9' : '#D4B97A'}" stroke="#090807" stroke-width="1.6"/>`;
    
    // Label positioning
    const [lx, ly] = pt(i, 1.28);
    let anchor = 'middle', baseline = 'middle';
    if (i === 0) { anchor = 'middle'; baseline = 'auto'; }
    else if (i === 3) { anchor = 'middle'; baseline = 'hanging'; }
    else if (i === 1 || i === 2) { anchor = 'start'; baseline = 'middle'; }
    else { anchor = 'end'; baseline = 'middle'; }
    
    const labelColor = active ? '#D4B97A' : '#A59D92';
    const numColor = active ? '#F2EFE9' : '#C1A673';
    svgHtml += `<text x="${lx}" y="${ly}" fill="${labelColor}" font-size="${active ? '11' : '9.5'}" font-family="Montserrat, sans-serif" text-anchor="${anchor}" dominant-baseline="${baseline}"><tspan font-weight="${active ? '600' : '500'}">${d.label}</tspan> <tspan fill="${numColor}" font-size="${active ? '10' : '9'}">${Math.round(val)}</tspan></text>`;
  });
  svg.innerHTML = svgHtml;
}

buildDials();
drawStarChart();
setupStarChartDrag();
if (typeof setupWaveformDrag === 'function') setupWaveformDrag();
if (typeof hookPlayerTimeUpdate === 'function') hookPlayerTimeUpdate();

// ---- Vibe Matrix & Archetypes ----
const VIBE_CATEGORIES = {
  genre: [
    'Lofi Chillhop', 'Ambient Drone', 'Deep House Yoga', 'Liquid Downtempo',
    'Neo-Soul Rhodes', 'Organic Folk Ambient', 'Cosmic Meditation', 'Bossa Chill',
    'Felt Piano Solo', 'Dub Techno Chill', 'Cinematic Ambient', 'Acoustic Zen'
  ],
  texture: [
    'Vinyl Crackle', 'Rain on Window', 'Analog Tape Warmth', 'Muted Rhodes',
    'Warm Sub Bass', 'Felt Piano', 'Bowed Cello', 'Wind Chimes', 'Distant Thunder',
    'Tape Flutter', 'Acoustic Harmonics', 'Glockenspiel', 'Modular Synth Drone'
  ],
  mood: [
    'Nostalgic', 'Weightless', 'Midnight Calm', 'Golden Hour Haze',
    'Introspective Zen', 'Softly Euphoric', 'Quiet Reverie', 'Deep Focus',
    'Tender', 'Wistful', 'Ethereal', 'Gently Uplifting'
  ],
  musical: [
    '60 BPM', '72 BPM', '84 BPM', '95 BPM', 'Beatless Free-time',
    'D Minor (Soulful)', 'F# Major (Ethereal)', 'A Minor (Nocturnal)',
    'C Major (Pure)', 'Pentatonic Serenity', 'Sustained Chords', 'Walking Bass'
  ]
};

let currentVibeTab = 'genre';
function switchVibeTab(tab) {
  currentVibeTab = tab;
  ['genre', 'texture', 'mood', 'musical'].forEach(t => {
    const el = document.getElementById('tab' + t.charAt(0).toUpperCase() + t.slice(1));
    if (el) el.classList.toggle('active', t === tab);
  });
  renderVibeTags();
}

function renderVibeTags() {
  const tray = document.getElementById('vibeTagTray');
  tray.innerHTML = '';
  const list = VIBE_CATEGORIES[currentVibeTab] || [];
  const curPrompt = document.getElementById('prompt').value.toLowerCase();
  list.forEach(tag => {
    const chip = document.createElement('div');
    const isActive = curPrompt.includes(tag.toLowerCase());
    chip.className = 'tag-chip' + (isActive ? ' active' : '');
    chip.textContent = (isActive ? '✓ ' : '+ ') + tag;
    chip.onclick = () => toggleVibeTag(tag);
    tray.appendChild(chip);
  });
}

function toggleVibeTag(tag) {
  const p = document.getElementById('prompt');
  const text = p.value;
  const tagLow = tag.toLowerCase();
  const items = text.split(',').map(s => s.trim()).filter(Boolean);
  const idx = items.findIndex(s => s.toLowerCase() === tagLow);
  if (idx >= 0) {
    items.splice(idx, 1);
  } else {
    items.push(tag);
  }
  p.value = items.join(', ');
  renderVibeTags();
}

renderVibeTags();

const ARCHETYPES = {
  zen: {
    prompt: '432Hz ambient drone, Tibetan singing bowls, warm analog pads, serene, weightless, sustained',
    dials: { warmth: 90, reverb: 95, tempo: 20, lofi: 15, melodic: 60, density: 30 },
    noDrums: true, noVocals: true, noBass: false, noHarshHighs: true
  },
  lofi: {
    prompt: 'dusty vinyl rhodes, slow hip hop beat, rain outside window, tape flutter, nostalgic, warm sub bass',
    dials: { warmth: 80, reverb: 60, tempo: 48, lofi: 85, melodic: 70, density: 60 },
    noDrums: false, noVocals: true, noBass: false, noHarshHighs: false
  },
  sunset: {
    prompt: 'warm nylon acoustic guitar, muted jazz trumpet, golden hour pad, soft marimba, gentle uplifting groove',
    dials: { warmth: 75, reverb: 65, tempo: 55, lofi: 40, melodic: 85, density: 50 },
    noDrums: false, noVocals: true, noBass: false, noHarshHighs: false
  },
  celestial: {
    prompt: 'shimmering glass marimba, weightless interstellar drone, detuned synth strings, bowed vibraphone, cosmic calm',
    dials: { warmth: 60, reverb: 90, tempo: 30, lofi: 25, melodic: 75, density: 40 },
    noDrums: true, noVocals: true, noBass: false, noHarshHighs: true
  },
  rain: {
    prompt: 'gentle upright felt piano, footsteps on pavement, rain on window glass, cozy ambient room, warm rhodes chords',
    dials: { warmth: 85, reverb: 70, tempo: 40, lofi: 65, melodic: 80, density: 45 },
    noDrums: true, noVocals: true, noBass: false, noHarshHighs: false
  }
};

function applyArchetype(key) {
  const arch = ARCHETYPES[key];
  if (!arch) return;
  document.getElementById('prompt').value = arch.prompt;
  Object.keys(arch.dials).forEach(k => {
    if (knobs[k]) knobs[k].set(arch.dials[k]);
  });
  drawStarChart();
  document.getElementById('noDrums').checked = arch.noDrums;
  document.getElementById('noVocals').checked = arch.noVocals;
  document.getElementById('noBass').checked = arch.noBass;
  document.getElementById('noHarshHighs').checked = arch.noHarshHighs;
  renderVibeTags();
}

// Alchemist / Prompt Generator
const ALCHEMIST_BANKS = {
  scenes: ['misty mountain pavilion', 'late-night Kyoto alleyway', 'quiet seaside dock at 4am', 'desert stargazing plateau', 'solitary lighthouse during rain', 'candlelit library nook', 'greenhouse during snowfall', 'rooftop at dusk', 'forest glade with morning dew', 'abandoned cathedral bathed in sunbeams'],
  textures: ['gentle felt piano', 'dusty vinyl rhodes', 'warm acoustic harmonics', 'soft analog Prophet pads', 'bowed vibraphone', 'subtle tape hiss', 'gentle ocean waves', 'distant thunder rolling', 'vintage mellotron strings', 'sub low-end warmth'],
  moods: ['deeply peaceful', 'nostalgic warmth', 'timeless and still', 'tender reverie', 'weightless calm', 'softly euphoric', 'contemplative zen', 'luminous stillness']
};

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function alchemistPrompt() {
  const scene = pick(ALCHEMIST_BANKS.scenes);
  const tex1 = pick(ALCHEMIST_BANKS.textures);
  let tex2 = pick(ALCHEMIST_BANKS.textures);
  while (tex2 === tex1) tex2 = pick(ALCHEMIST_BANKS.textures);
  const mood = pick(ALCHEMIST_BANKS.moods);
  document.getElementById('prompt').value = `${mood}, ${scene}, ${tex1}, ${tex2}`;
  renderVibeTags();
}

function randomisePrompt() {
  alchemistPrompt();
}

function copyPrompt() {
  const val = document.getElementById('prompt').value;
  if (!val) return;
  navigator.clipboard.writeText(val).then(() => {
    const btn = document.getElementById('copyPromptBtn');
    btn.textContent = '✓';
    setTimeout(() => { btn.textContent = '📋'; }, 1500);
  });
}

function clearPrompt() {
  document.getElementById('prompt').value = '';
  renderVibeTags();
}

// ---- Name Generator & Aesthetic Universes ----
const NAME_UNIVERSES = {
  celestial: {
    adj: ['Astral', 'Solstice', 'Starlight', 'Eclipse', 'Nebula', 'Zenith', 'Lunar', 'Supernova', 'Pulsar', 'Andromeda', 'Celestial', 'Eventide', 'Equinox', 'Solar'],
    noun: ['Orbit', 'Corona', 'Constellation', 'Horizon', 'Singularity', 'Void', 'Atmosphere', 'Dust', 'Radiance', 'Aurora', 'Twilight', 'Firmament']
  },
  tideline: {
    adj: ['Stillwater', 'Tidepool', 'Undertow', 'Coastal', 'Deep Blue', 'Driftwood', 'Cascade', 'Shoal', 'Saltwater', 'Aquamarine', 'Pelagic', 'Submerged'],
    noun: ['Current', 'Mist', 'Lagoon', 'Shores', 'Reef', 'Abyss', 'Ripple', 'Sanctuary', 'Breeze', 'Estuary', 'Tides', 'Harbor']
  },
  obsidian: {
    adj: ['Velvet', 'Hollow', 'Midnight', 'Onyx', 'Obsidian', 'Nocturne', 'Dim', 'Shadow', 'Moonlit', 'Smokey', 'Subtle', 'Lowlight'],
    noun: ['Hush', 'Silhouette', 'Echo', 'Lantern', 'Reverie', 'Candle', 'Sanctum', 'Drift', 'Ember', 'Glow', 'Chamber', 'Whisper']
  },
  zen: {
    adj: ['Silent', 'Harmonious', 'Pure', 'Weightless', 'Timeless', 'Serene', 'Unbroken', 'Gentle', 'Sacred', 'Kanso', 'Satori', 'Peaceful'],
    noun: ['Lotus', 'Sanctuary', 'Breath', 'Meadow', 'Solitude', 'Presence', 'Awakening', 'Stone', 'Garden', 'Monolith', 'Stillness', 'Bonsai']
  },
  organic: {
    adj: ['Amber', 'Golden', 'Honey', 'Terracotta', 'Cedar', 'Autumn', 'Linen', 'Sunbeam', 'Faded', 'Warm', 'Botanical', 'Mossy'],
    noun: ['Hearth', 'Petal', 'Bloom', 'Grove', 'Timber', 'Roots', 'Canopy', 'Bough', 'Soil', 'Sandalwood', 'Orchard', 'Branch']
  }
};

const JAPANESE_CONCEPTS = [
  'Komorebi (Sunlight Through Leaves)',
  'Yūgen (Profound Grace)',
  'Mono no Aware (Sweet Transience)',
  'Wabi-Sabi (Flawed Beauty)',
  'Shinrin-yoku (Forest Bathing)',
  'Kanso (Simplicity and Calm)',
  'Shibui (Subtle Refinement)',
  'Satori (Sudden Awakening)'
];

function generateOneTitle() {
  const universeKey = document.getElementById('titleUniverse').value;
  const formula = document.getElementById('titleFormula').value;
  
  let uKeys = universeKey === 'all' ? Object.keys(NAME_UNIVERSES) : [universeKey];
  const u = NAME_UNIVERSES[pick(uKeys)];
  
  let chosenFormula = formula;
  if (formula === 'random') {
    const formulas = ['adj_noun', 'noun_of_noun', 'verbing', 'japanese', 'opus'];
    chosenFormula = pick(formulas);
  }

  if (chosenFormula === 'japanese') {
    return pick(JAPANESE_CONCEPTS);
  } else if (chosenFormula === 'opus') {
    const num = Math.floor(Math.random() * 24) + 1;
    return `Tranquilicy Opus ${num} in ${pick(['D Minor', 'F# Major', 'A Minor', 'C#', 'E Minor'])}`;
  } else if (chosenFormula === 'noun_of_noun') {
    return `The ${pick(u.noun)} of ${pick(u.noun)}`;
  } else if (chosenFormula === 'verbing') {
    const verbs = ['Drifting Through', 'Resting Upon', 'Breathing Inside', 'Wandering Past', 'Gazing Into'];
    return `${pick(verbs)} ${pick(u.noun)}`;
  } else {
    return `${pick(u.adj)} ${pick(u.noun)}`;
  }
}

function rerollTitle() {
  const t = generateOneTitle();
  document.getElementById('trackTitle').value = t;
  updateDownloadNames();
  refreshPreview();
}

function generateBatchTitles() {
  const tray = document.getElementById('titlePillTray');
  tray.innerHTML = '';
  for (let i = 0; i < 5; i++) {
    const t = generateOneTitle();
    const pill = document.createElement('div');
    pill.className = 'title-pill';
    pill.textContent = t;
    pill.onclick = () => {
      document.getElementById('trackTitle').value = t;
      updateDownloadNames();
      refreshPreview();
    };
    tray.appendChild(pill);
  }
}
generateBatchTitles();

// ---- Build Full Prompts for Server ----
function buildPrompt() {
  const parts = ['chillout'];
  DIALS.forEach(d => parts.push(d.describe(knobs[d.key].value)));
  if (document.getElementById('noDrums').checked) parts.push('beatless, free-time, sustained');
  if (document.getElementById('noVocals').checked) parts.push('instrumental');
  const extra = document.getElementById('prompt').value.trim();
  if (extra) parts.push(extra);
  return parts.join(', ');
}

const EXCLUDE_TERMS = {
  noDrums: 'drums, percussion, drum beat, kick drum, snare, hi-hats, rhythmic beat',
  noVocals: 'vocals, singing, voice, choir, lyrics, spoken word',
  noBass: 'bass guitar, deep bass, sub bass, heavy low end',
  noHarshHighs: 'harsh highs, screaming synths, aggressive lead, brass stabs'
};

function buildNegativePrompt() {
  return Object.keys(EXCLUDE_TERMS)
    .filter(id => {
      const el = document.getElementById(id);
      return el && el.checked;
    })
    .map(id => EXCLUDE_TERMS[id])
    .join(', ');
}

// ---- Step Flow Tracking ----
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

function setCardEnabled(id, enabled) {
  const card = document.getElementById(id);
  const wasDisabled = card.classList.contains('disabled');
  card.classList.toggle('disabled', !enabled);
  card.inert = !enabled;
  card.setAttribute('aria-hidden', enabled ? 'false' : 'true');
  return wasDisabled && enabled;
}

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
  const safe = name.replace(/[\/:*?"<>|]/g, '').trim() || 'track';
  return `${safe}.${ext}`;
}

let lastMasterExt = 'wav';
function updateDownloadNames() {
  if (lastAudioUrl) setChip('downloadBtn', lastAudioUrl, buildFilename('wav'));
  if (lastMasterUrl) document.getElementById('masterChip').download = buildFilename(lastMasterExt);
  if (lastVideoUrl) document.getElementById('downloadVideoBtn').download = buildFilename('webm');
  if (lastStillUrl) document.getElementById('stillChip').download = buildFilename('png');
}

// ---- Audio Generation Handler ----
async function generate() {
  const btn = document.getElementById('genBtn');
  const genNotice = document.getElementById('genErrorNotice');
  const outbar = document.getElementById('outbar');
  const barInner = document.getElementById('barInner');
  const statusPct = document.getElementById('statusPct');
  const statusLabel = document.getElementById('statusLabel');
  const errorText = document.getElementById('errorText');
  const player = document.getElementById('player');
  const dur = parseFloat(document.getElementById('duration').value) || 20;

  const cancelBtn = document.getElementById('cancelBtn');
  const playerWrap = document.getElementById('playerWrap');

  btn.disabled = true;
  btn.classList.add('busy');
  const origBtnHtml = 'Generate Track';
  btn.innerHTML = '<span class="spin-ring"></span> Synthesizing...';

  if (genNotice) { genNotice.style.display = 'none'; genNotice.textContent = ''; }
  errorText.style.display = 'none';
  playerWrap.hidden = true;
  playerWrap.classList.remove('ready');
  outbar.classList.add('busy');
  cancelBtn.style.display = 'inline-block';
  barInner.style.width = '0%';
  statusPct.textContent = '0%';
  statusLabel.textContent = dur > 30 ? 'Generating (chained on RTX 3090)...' : 'Generating on RTX 3090...';

  ['downloadBtn', 'masterChip', 'downloadVideoBtn', 'stillChip'].forEach(id => setChip(id, null));
  [lastVideoUrl, lastMasterUrl, lastStillUrl].forEach(u => { if (u) URL.revokeObjectURL(u); });
  lastVideoUrl = lastMasterUrl = lastStillUrl = null;
  setVidStatus('');
  document.getElementById('masterStatus').style.display = 'none';
  flow.mastered = false;
  flow.rendered = false;
  renderFlow();

  try {
    const endpoint = getApiEndpoint();
    console.log('[Tranquilicy] Calling GPU endpoint:', endpoint || '(same origin)');
    
    const targetUrl = getApiUrl('/generate');
    const startRes = await fetch(targetUrl, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: buildPrompt(),
        negative_prompt: buildNegativePrompt(),
        duration_sec: dur,
      })
    });
    
    if (!startRes.ok) {
      let errDetail = 'Server error: ' + startRes.status;
      try {
        const errJson = await startRes.json();
        if (errJson.detail) errDetail = errJson.detail;
        else if (errJson.error) errDetail = errJson.error;
      } catch(e) {}
      throw new Error(errDetail);
    }
    
    const data = await startRes.json();
    if (!data.job_id) {
      throw new Error('Invalid response from server: ' + JSON.stringify(data));
    }
    const { job_id } = data;
    currentJobId = job_id;

    while (true) {
      await new Promise(r => setTimeout(r, 400));
      const sRes = await fetch(getApiUrl('/status/' + job_id));
      if (!sRes.ok) throw new Error('Status check failed: ' + sRes.status);
      const s = await sRes.json();
      if (s.error) throw new Error(s.error);
      if (s.in_queue) {
        btn.innerHTML = `<span class="spin-ring"></span> In Queue (#${s.queue_position})...`;
        statusLabel.textContent = `In Queue (Position #${s.queue_position}) · Est. wait ~${s.estimated_sec}s`;
      } else if (typeof s.progress === 'number') {
        const pct = Math.round(s.progress * 100);
        barInner.style.width = pct + '%';
        statusPct.textContent = pct + '%';
        btn.innerHTML = `<span class="spin-ring"></span> Synthesizing (${pct}%)...`;
        statusLabel.textContent = dur > 30 ? 'Generating (chained on RTX 3090)...' : 'Generating on RTX 3090...';
      }
      if (s.done) break;
    }

    btn.innerHTML = '<span class="spin-ring"></span> Fetching Audio...';
    const audioRes = await fetch(getApiUrl('/result/' + job_id));
    if (!audioRes.ok) throw new Error('Failed to retrieve audio: ' + audioRes.status);
    const blob = await audioRes.blob();
    if (lastAudioUrl) URL.revokeObjectURL(lastAudioUrl);
    const url = URL.createObjectURL(blob);
    lastAudioUrl = url;
    player.src = url;

    playerWrap.hidden = false;
    requestAnimationFrame(() => playerWrap.classList.add('ready'));
    player.play().catch(() => {});

    statusLabel.textContent = `Ready · ${dur}s`; refreshQuota();
    statusPct.textContent = '';
    btn.innerHTML = '✓ Track Ready!';
    setTimeout(() => { btn.innerHTML = origBtnHtml; }, 2200);

    flow.generated = true;
    renderFlow();
    unlockExportSteps();
    if (!document.getElementById('trackTitle').value.trim()) rerollTitle();
    updateDownloadNames();
    refreshPreview();

    // Decode & render interactive waveform timeline
    if (typeof buildWaveformFromBlob === 'function') buildWaveformFromBlob(blob);

    // Store in Session Takes Library
    if (typeof addSessionTake === 'function') {
      addSessionTake({
        id: currentJobId,
        title: document.getElementById('trackTitle').value || 'Take #' + (sessionTakes.length + 1),
        url: url,
        blob: blob,
        duration: dur,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      });
    }

    // Scroll to player so user immediately sees results
    outbar.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) {
    console.error('[Tranquilicy] Generation error:', e);
    const msg = e.message === 'Cancelled' ? 'Cancelled.' : 'Error: ' + e.message;
    errorText.textContent = msg;
    errorText.style.display = 'block';
    if (genNotice) {
      genNotice.style.display = 'block';
      genNotice.innerHTML = `⚠️ <b>Generation issue:</b> ${msg}<br><span style="font-size:11px;opacity:0.8;">Check that scripts/09_api_server.py is running on your RTX 3090, or click the GPU badge in the top-right to inspect connection.</span>`;
    }
    barInner.style.width = '0%';
    statusPct.textContent = '';
    btn.innerHTML = origBtnHtml;
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
    btn.classList.remove('busy');
    cancelBtn.style.display = 'none';
    outbar.classList.remove('busy');
  }
}

async function cancelGeneration() {
  if (!currentJobId) return;
  const cancelBtn = document.getElementById('cancelBtn');
  cancelBtn.textContent = 'Cancelling...';
  try {
    await fetch(getApiUrl('/cancel/' + currentJobId), { method: 'POST' });
  } catch (e) {}
  cancelBtn.textContent = 'Cancel';
}

// ---- Audio Graph, 4-Channel Soundscape Mixer & Binaural Engine ----
let audioCtx = null, analyserNode = null, mediaStreamDest = null;
let toneLowShelf = null, toneMidPeak = null, toneHighShelf = null;
let lofiHighPass = null, lofiLowPass = null;
let musicGain = null, rainGain = null, vinylGain = null, binauralGain = null;
let rainSource = null, vinylSource = null;
let binauralLeftOsc = null, binauralRightOsc = null;

// Mixer State: volume (0..1.2), mute, solo
const mixer = {
  music: { vol: 1.0, mute: false, solo: false },
  rain: { vol: 0.0, mute: false, solo: false },
  vinyl: { vol: 0.0, mute: false, solo: false },
  binaural: { vol: 0.0, mute: false, solo: false }
};

// Binaural & Solfeggio Presets
const BINAURAL_PRESETS = {
  alpha: { base: 216, offset: 10, label: 'Alpha (10 Hz Focus)' },
  theta: { base: 144, offset: 6, label: 'Theta (6 Hz Zen)' },
  delta: { base: 108, offset: 2.5, label: 'Delta (2.5 Hz Sleep)' },
  solfeggio432: { base: 432, offset: 0, label: 'Solfeggio 432 Hz' },
  solfeggio528: { base: 528, offset: 0, label: 'Solfeggio 528 Hz' }
};

async function ensureAudioGraph() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaElementSource(document.getElementById('player'));
    
    // Master Analyser & Recording Destination
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 1024;
    analyserNode.smoothingTimeConstant = 0.82;
    mediaStreamDest = audioCtx.createMediaStreamDestination();

    // 1. Music Channel: Source -> EQ -> Lo-Fi -> Music Gain -> Analyser
    toneLowShelf = audioCtx.createBiquadFilter();
    toneLowShelf.type = 'lowshelf';
    toneLowShelf.frequency.value = 120;
    
    toneMidPeak = audioCtx.createBiquadFilter();
    toneMidPeak.type = 'peaking';
    toneMidPeak.frequency.value = 1500;
    toneMidPeak.Q.value = 1.0;
    
    toneHighShelf = audioCtx.createBiquadFilter();
    toneHighShelf.type = 'highshelf';
    toneHighShelf.frequency.value = 8000;
    
    lofiHighPass = audioCtx.createBiquadFilter();
    lofiHighPass.type = 'highpass';
    lofiHighPass.frequency.value = 20;
    
    lofiLowPass = audioCtx.createBiquadFilter();
    lofiLowPass.type = 'lowpass';
    lofiLowPass.frequency.value = 20000;

    musicGain = audioCtx.createGain();
    musicGain.gain.value = 1.0;

    source.connect(toneLowShelf);
    toneLowShelf.connect(toneMidPeak);
    toneMidPeak.connect(toneHighShelf);
    toneHighShelf.connect(lofiHighPass);
    lofiHighPass.connect(lofiLowPass);
    lofiLowPass.connect(musicGain);
    musicGain.connect(analyserNode);

    // 2. Procedural Rain Channel: BufferSource -> Rain Gain -> Analyser
    rainGain = audioCtx.createGain();
    rainGain.gain.value = 0.0;
    const rainBuf = createRainBuffer(audioCtx);
    rainSource = audioCtx.createBufferSource();
    rainSource.buffer = rainBuf;
    rainSource.loop = true;
    rainSource.connect(rainGain);
    rainGain.connect(analyserNode);
    rainSource.start();

    // 3. Procedural Vinyl Channel: BufferSource -> Vinyl Gain -> Analyser
    vinylGain = audioCtx.createGain();
    vinylGain.gain.value = 0.0;
    const vinylBuf = createVinylBuffer(audioCtx);
    vinylSource = audioCtx.createBufferSource();
    vinylSource.buffer = vinylBuf;
    vinylSource.loop = true;
    vinylSource.connect(vinylGain);
    vinylGain.connect(analyserNode);
    vinylSource.start();

    // 4. Stereo Binaural Drone: Left & Right Oscillators -> ChannelMerger -> Binaural Gain -> Analyser
    const merger = audioCtx.createChannelMerger(2);
    binauralLeftOsc = audioCtx.createOscillator();
    binauralRightOsc = audioCtx.createOscillator();
    binauralLeftOsc.type = 'sine';
    binauralRightOsc.type = 'sine';
    binauralGain = audioCtx.createGain();
    binauralGain.gain.value = 0.0;

    updateBinauralPreset();

    binauralLeftOsc.connect(merger, 0, 0);  // Left channel
    binauralRightOsc.connect(merger, 0, 1); // Right channel
    merger.connect(binauralGain);
    binauralGain.connect(analyserNode);

    binauralLeftOsc.start();
    binauralRightOsc.start();

    // Master Output Routing
    analyserNode.connect(audioCtx.destination);
    analyserNode.connect(mediaStreamDest);
  }
  if (audioCtx.state === 'suspended') await audioCtx.resume();
}

function updateMixerVolumes() {
  ['music', 'rain', 'vinyl', 'binaural'].forEach(ch => {
    const input = document.getElementById('vol' + ch.charAt(0).toUpperCase() + ch.slice(1));
    const valSpan = document.getElementById('vol' + ch.charAt(0).toUpperCase() + ch.slice(1) + 'Val');
    if (input && valSpan) {
      mixer[ch].vol = parseFloat(input.value) / 100;
      valSpan.textContent = input.value;
    }
  });

  applyMixerGains();
}

function applyMixerGains() {
  if (!audioCtx) return;
  const anySolo = Object.values(mixer).some(m => m.solo);

  const setChannel = (ch, gainNode) => {
    if (!gainNode) return;
    const m = mixer[ch];
    let effective = m.vol;
    if (m.mute || (anySolo && !m.solo)) {
      effective = 0.0;
    }
    gainNode.gain.setTargetAtTime(effective, audioCtx.currentTime, 0.04);

    const strip = document.getElementById('strip' + ch.charAt(0).toUpperCase() + ch.slice(1));
    const muteBtn = document.getElementById('mute' + ch.charAt(0).toUpperCase() + ch.slice(1) + 'Btn');
    const soloBtn = document.getElementById('solo' + ch.charAt(0).toUpperCase() + ch.slice(1) + 'Btn');
    if (strip) {
      strip.classList.toggle('muted', m.mute || (anySolo && !m.solo));
      strip.classList.toggle('soloed', m.solo);
    }
    if (muteBtn) muteBtn.classList.toggle('mute-active', m.mute);
    if (soloBtn) soloBtn.classList.toggle('solo-active', m.solo);
  };

  setChannel('music', musicGain);
  setChannel('rain', rainGain);
  setChannel('vinyl', vinylGain);
  setChannel('binaural', binauralGain);
}

function toggleMute(ch) {
  mixer[ch].mute = !mixer[ch].mute;
  if (mixer[ch].mute) mixer[ch].solo = false;
  applyMixerGains();
}

function toggleSolo(ch) {
  mixer[ch].solo = !mixer[ch].solo;
  if (mixer[ch].solo) mixer[ch].mute = false;
  applyMixerGains();
}

function updateBinauralPreset() {
  const presetKey = document.getElementById('binauralPreset').value;
  const readout = document.getElementById('binauralFreqReadout');
  const p = BINAURAL_PRESETS[presetKey] || BINAURAL_PRESETS.alpha;
  if (readout) readout.textContent = p.label;

  if (binauralLeftOsc && binauralRightOsc && audioCtx) {
    binauralLeftOsc.frequency.setTargetAtTime(p.base, audioCtx.currentTime, 0.08);
    binauralRightOsc.frequency.setTargetAtTime(p.base + p.offset, audioCtx.currentTime, 0.08);
  }
}

function updateToneEq() {
  if (!toneLowShelf) return;
  const w = parseFloat(document.getElementById('toneWarmth').value);
  const a = parseFloat(document.getElementById('toneAir').value);
  const isLofi = document.getElementById('lofiFilter').checked;
  
  document.getElementById('warmthVal').textContent = w;
  document.getElementById('airVal').textContent = a;
  
  toneLowShelf.gain.value = w;
  toneHighShelf.gain.value = a;

  if (isLofi) {
    lofiHighPass.frequency.value = 320;
    lofiLowPass.frequency.value = 4200;
  } else {
    lofiHighPass.frequency.value = 20;
    lofiLowPass.frequency.value = 20000;
  }
}

// Procedural Audio Generators
function createVinylBuffer(ctx) {
  const sr = ctx.sampleRate, dur = 5;
  const buf = ctx.createBuffer(1, sr * dur, sr);
  const data = buf.getChannelData(0);
  for (let i = 0; i < data.length; i++) {
    data[i] = (Math.random() * 2 - 1) * 0.012;
    if (Math.random() < 0.00045) {
      data[i] = (Math.random() * 2 - 1) * 0.18;
    }
  }
  return buf;
}

function createRainBuffer(ctx) {
  const sr = ctx.sampleRate, dur = 5;
  const buf = ctx.createBuffer(1, sr * dur, sr);
  const data = buf.getChannelData(0);
  let lastOut = 0.0;
  for (let i = 0; i < data.length; i++) {
    const white = Math.random() * 2 - 1;
    lastOut = (lastOut + (0.02 * white)) / 1.02;
    data[i] = lastOut * 0.12;
  }
  return buf;
}

// ---- Interactive Waveform Timeline Engine ----
let waveformPeaks = [];
let loopStartSec = 0.0, loopEndSec = 20.0;
let isDraggingLoopHandle = null;

async function buildWaveformFromBlob(blob) {
  const canvas = document.getElementById('waveformCanvas');
  if (!canvas || !blob) return;
  try {
    const ab = await blob.arrayBuffer();
    const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuf = await tempCtx.decodeAudioData(ab);
    const rawData = audioBuf.getChannelData(0);
    const samples = 140;
    const blockSize = Math.floor(rawData.length / samples);
    waveformPeaks = [];
    for (let i = 0; i < samples; i++) {
      let sum = 0;
      for (let j = 0; j < blockSize; j++) {
        sum += Math.abs(rawData[i * blockSize + j]);
      }
      waveformPeaks.push(sum / blockSize);
    }
    const maxVal = Math.max(...waveformPeaks, 0.001);
    waveformPeaks = waveformPeaks.map(p => p / maxVal);
    
    loopStartSec = 0.0;
    loopEndSec = audioBuf.duration;
    updateLoopUi();
    drawTimelineWaveform();
  } catch(e) {
    console.warn('[Tranquilicy] Waveform decode error:', e);
  }
}

function drawTimelineWaveform() {
  const canvas = document.getElementById('waveformCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.offsetWidth || 800;
  const h = canvas.height = 60;

  ctx.clearRect(0, 0, w, h);
  if (!waveformPeaks.length) return;

  const barW = w / waveformPeaks.length;
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, '#F2EFE9');
  grad.addColorStop(0.5, '#D4B97A');
  grad.addColorStop(1, '#A58B58');

  ctx.fillStyle = grad;
  waveformPeaks.forEach((p, i) => {
    const barH = Math.max(3, p * (h - 8));
    const x = i * barW;
    const y = (h - barH) / 2;
    ctx.fillRect(x + 1, y, Math.max(1.5, barW - 1.5), barH);
  });
}

function handleWaveformClick(e) {
  const track = document.getElementById('waveformTrack');
  const player = document.getElementById('player');
  if (!track || !player || !player.duration) return;
  const rect = track.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  player.currentTime = pct * player.duration;
}

function setupWaveformDrag() {
  const track = document.getElementById('waveformTrack');
  const handleA = document.getElementById('loopHandleA');
  const handleB = document.getElementById('loopHandleB');
  if (!track || !handleA || !handleB) return;

  const onPointerDown = (handle, e) => {
    isDraggingLoopHandle = handle;
    handle.setPointerCapture(e.pointerId);
    e.stopPropagation();
    e.preventDefault();
  };

  handleA.addEventListener('pointerdown', e => onPointerDown('a', e));
  handleB.addEventListener('pointerdown', e => onPointerDown('b', e));

  const onPointerMove = e => {
    if (!isDraggingLoopHandle) return;
    const player = document.getElementById('player');
    if (!player || !player.duration) return;
    const rect = track.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const sec = pct * player.duration;

    if (isDraggingLoopHandle === 'a') {
      loopStartSec = Math.min(sec, loopEndSec - 0.5);
    } else if (isDraggingLoopHandle === 'b') {
      loopEndSec = Math.max(sec, loopStartSec + 0.5);
    }
    updateLoopUi();
  };

  const onPointerUp = () => { isDraggingLoopHandle = null; };

  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
}

function updateLoopUi() {
  const player = document.getElementById('player');
  const dur = (player && player.duration) ? player.duration : 20.0;
  const pctA = (loopStartSec / dur) * 100;
  const pctB = (loopEndSec / dur) * 100;

  const handleA = document.getElementById('loopHandleA');
  const handleB = document.getElementById('loopHandleB');
  const highlight = document.getElementById('loopHighlight');
  const tag = document.getElementById('loopTag');

  if (handleA) handleA.style.left = pctA + '%';
  if (handleB) handleB.style.left = pctB + '%';
  if (highlight) {
    highlight.style.left = pctA + '%';
    highlight.style.width = Math.max(0, pctB - pctA) + '%';
  }
  if (tag) {
    const elA = document.getElementById('loopAVal');
    const elB = document.getElementById('loopBVal');
    if (elA) elA.textContent = loopStartSec.toFixed(1) + 's';
    if (elB) elB.textContent = loopEndSec.toFixed(1) + 's';
  }
}

function hookPlayerTimeUpdate() {
  const player = document.getElementById('player');
  const playhead = document.getElementById('waveformPlayhead');
  const timeDisplay = document.getElementById('waveCurrentTime');
  const totalDisplay = document.getElementById('waveTotalTime');

  if (!player) return;

  const fmtTime = s => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec < 10 ? '0' : ''}${sec}`;
  };

  player.addEventListener('timeupdate', () => {
    if (player.duration) {
      const pct = (player.currentTime / player.duration) * 100;
      if (playhead) playhead.style.left = pct + '%';
      if (timeDisplay) timeDisplay.textContent = fmtTime(player.currentTime);
      if (totalDisplay) totalDisplay.textContent = fmtTime(player.duration);

      const isLooping = document.getElementById('playerLoopBtn') && document.getElementById('playerLoopBtn').classList.contains('active');
      if (isLooping && player.currentTime >= loopEndSec) {
        player.currentTime = loopStartSec;
        player.play().catch(() => {});
      }
    }
  });
}

// ---- Session Takes Library Tray ----
const sessionTakes = [];

function addSessionTake(take) {
  sessionTakes.unshift(take);
  const section = document.getElementById('takeLibrarySection');
  const countEl = document.getElementById('takeCount');
  const tray = document.getElementById('takeTray');

  if (section) section.style.display = 'block';
  if (countEl) countEl.textContent = sessionTakes.length;
  if (!tray) return;

  tray.innerHTML = '';
  sessionTakes.forEach((t, idx) => {
    const card = document.createElement('div');
    card.className = 'take-card' + (idx === 0 ? ' active' : '');
    card.innerHTML = `
      <div class="take-card-title">${t.title || 'Take #' + (sessionTakes.length - idx)}</div>
      <div class="take-card-meta">
        <span>⏱ ${Math.round(t.duration)}s</span>
        <span>${t.time}</span>
      </div>
      <div class="take-card-actions">
        <button type="button" class="take-btn-mini" onclick="loadSessionTake(${idx})">▶ Load</button>
        <a class="take-btn-mini" href="${t.url}" download="${(t.title || 'take').replace(/[\/:*?"<>|]/g, '')}.wav">💾 WAV</a>
      </div>
    `;
    tray.appendChild(card);
  });
}

function loadSessionTake(idx) {
  const take = sessionTakes[idx];
  if (!take) return;
  const player = document.getElementById('player');
  const playerWrap = document.getElementById('playerWrap');
  const titleInput = document.getElementById('trackTitle');

  lastAudioUrl = take.url;
  player.src = take.url;
  playerWrap.hidden = false;
  requestAnimationFrame(() => playerWrap.classList.add('ready'));
  player.play().catch(() => {});

  if (titleInput && take.title) titleInput.value = take.title;
  updateDownloadNames();
  buildWaveformFromBlob(take.blob);
  refreshPreview();

  const cards = document.querySelectorAll('.take-card');
  cards.forEach((c, i) => c.classList.toggle('active', i === idx));
}

function updateVidLoopEst() {
  const sel = document.getElementById('vidLoops');
  const span = document.getElementById('vidLoopEstVal');
  const player = document.getElementById('player');
  if (!sel || !span) return;
  const factor = parseInt(sel.value) || 1;
  const dur = (player && player.duration) ? player.duration : 20.0;
  const totalSec = Math.round(dur * factor);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  span.textContent = `${factor}x Loop (~${m ? m + 'm ' : ''}${s}s)`;
}

// ---- Audio Mastering & Download Chain ----
async function downloadMastered() {
  if (!currentJobId) return;
  const btn = document.getElementById('masterBtn');
  const statusEl = document.getElementById('masterStatus');
  const format = document.getElementById('exportFormat').value;
  btn.disabled = true;
  statusEl.style.display = 'block';
  statusEl.textContent = 'Processing master with analog saturation and loudness curve...';

  const params = new URLSearchParams({
    preset: document.getElementById('masterPreset').value,
    width: document.getElementById('stereoWidth').value,
    loop: document.getElementById('seamlessLoop').checked ? 'true' : 'false',
    fade_in: document.getElementById('fadeIn').value,
    fade_out: document.getElementById('fadeOut').value,
    format: format,
    warmth: document.getElementById('masterWarmth').value,
    air: document.getElementById('masterAir').value
  });

  try {
    const res = await fetch(getApiUrl(`/master/${currentJobId}?${params}`));
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Mastering failed with status ' + res.status);
    }
    const blob = await res.blob();
    if (lastMasterUrl) URL.revokeObjectURL(lastMasterUrl);
    lastMasterUrl = URL.createObjectURL(blob);
    lastMasterExt = format;

    setChip('masterChip', lastMasterUrl, buildFilename(format));
    statusEl.textContent = '✓ Master ready (' + format.toUpperCase() + ')';
    flow.mastered = true;
    renderFlow();
  } catch (e) {
    statusEl.textContent = 'Mastering error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

// ---- Video & Visualizer Studio ----
const PALETTES = {
  gold: ['#D4B97A', '#C1A673'],
  ember: ['#E3A868', '#8C5A34'],
  moonlit: ['#DCE6E2', '#7C8F8C'],
  amethyst: ['#C4A1E8', '#6F5299'],
  emerald: ['#9FD4A6', '#4A7A55']
};

const ASPECTS = {
  '16:9': [960, 540],
  '9:16': [540, 960],
  '1:1': [720, 720],
  '4:5': [576, 720]
};

let coverImage = null;
function handleCoverUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => {
      coverImage = img;
      const thumb = document.getElementById('artThumb');
      thumb.src = ev.target.result;
      thumb.style.display = 'block';
      document.getElementById('artStatus').textContent = file.name;
      document.getElementById('removeArtBtn').style.display = 'inline-block';
      document.getElementById('vidBackdrop').value = 'custom';
      refreshPreview();
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

function removeCoverArt() {
  coverImage = null;
  const thumb = document.getElementById('artThumb');
  thumb.src = '';
  thumb.style.display = 'none';
  document.getElementById('artUpload').value = '';
  document.getElementById('artStatus').textContent = 'Click or drop custom album art (JPG/PNG)';
  document.getElementById('removeArtBtn').style.display = 'none';
  if (document.getElementById('vidBackdrop').value === 'custom') {
    document.getElementById('vidBackdrop').value = 'feathers';
  }
  refreshPreview();
}

function sizeExportCanvas() {
  const canvas = document.getElementById('exportCanvas');
  const [w, h] = ASPECTS[document.getElementById('vidAspect').value] || [960, 540];
  canvas.width = w;
  canvas.height = h;
}

let videoFeathers = null;
function ensureVideoFeathers(w, h) {
  if (videoFeathers && videoFeathers.w === w && videoFeathers.h === h) return videoFeathers;
  const count = 18;
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

// Particle Nebula
let stardustField = null;
function ensureStardust(w, h) {
  if (stardustField && stardustField.w === w && stardustField.h === h) return stardustField;
  stardustField = {
    w, h,
    stars: Array.from({ length: 90 }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: 1 + Math.random() * 2.2,
      alpha: 0.2 + Math.random() * 0.7,
      vx: (Math.random() - 0.5) * 6,
      vy: (Math.random() - 0.5) * 6
    }))
  };
  return stardustField;
}

let backdropLastT = null;
function drawBackdrop(ctx, w, h, t, mode, bassRms) {
  ctx.fillStyle = '#090807';
  ctx.fillRect(0, 0, w, h);

  if (mode === 'minimal') return;

  if (mode === 'custom' && coverImage) {
    const blur = parseFloat(document.getElementById('artBlur').value);
    const dim = parseFloat(document.getElementById('artDim').value) / 100;
    ctx.save();
    if (blur > 0) ctx.filter = `blur(${blur}px)`;
    // cover fill with aspect ratio preserving
    const imgRatio = coverImage.width / coverImage.height;
    const canvasRatio = w / h;
    let dw, dh, dx, dy;
    if (imgRatio > canvasRatio) {
      dh = h * 1.08; dw = dh * imgRatio;
      dx = (w - dw) / 2; dy = (h - dh) / 2;
    } else {
      dw = w * 1.08; dh = dw / imgRatio;
      dx = (w - dw) / 2; dy = (h - dh) / 2;
    }
    ctx.drawImage(coverImage, dx, dy, dw, dh);
    ctx.filter = 'none';
    ctx.restore();

    // Dim overlay
    ctx.fillStyle = `rgba(9,8,7,${dim})`;
    ctx.fillRect(0, 0, w, h);
    return;
  }

  if (mode === 'bloom') {
    const cx1 = w * 0.3 + Math.sin(t * 0.3) * w * 0.08, cy1 = h * 0.35 + Math.cos(t * 0.24) * h * 0.06;
    const g1 = ctx.createRadialGradient(cx1, cy1, 0, cx1, cy1, Math.max(w, h) * (0.55 + bassRms * 0.2));
    g1.addColorStop(0, 'rgba(193,166,115,0.34)');
    g1.addColorStop(1, 'rgba(193,166,115,0)');
    ctx.fillStyle = g1; ctx.fillRect(0, 0, w, h);

    const cx2 = w * 0.75 + Math.cos(t * 0.18) * w * 0.07, cy2 = h * 0.7 + Math.sin(t * 0.27) * h * 0.07;
    const g2 = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, Math.max(w, h) * 0.45);
    g2.addColorStop(0, 'rgba(212,185,122,0.22)');
    g2.addColorStop(1, 'rgba(212,185,122,0)');
    ctx.fillStyle = g2; ctx.fillRect(0, 0, w, h);
    return;
  }

  if (mode === 'nebula') {
    const dust = ensureStardust(w, h);
    dust.stars.forEach(s => {
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r * (1 + bassRms * 0.8), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(212,185,122,${s.alpha})`;
      ctx.fill();
    });
    return;
  }

  // Feathers
  const field = ensureVideoFeathers(w, h);
  const dt = backdropLastT === null ? 0 : Math.min(Math.max(t - backdropLastT, 0), 0.1);
  backdropLastT = t;
  for (const p of field.particles) {
    p.y += p.speedY * dt;
    p.rot += p.rotSpeed * dt;
    if (p.y - p.size * 1.3 > h) {
      p.y = -p.size * 1.3;
      p.baseX = Math.random() * w;
      p.phase = Math.random() * Math.PI * 2;
    }
    p.x = p.baseX + Math.sin(t * p.swayFreq * Math.PI * 2 + p.phase) * p.swayAmp;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rot);
    ctx.globalAlpha = p.opacity;
    drawFeatherShape(ctx, p.size, '#D4B97A', '#C1A673', 'rgba(9,8,7,.25)');
    ctx.restore();
  }
}

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

function getBassRms() {
  const data = waveformData('freq');
  let sum = 0;
  const count = Math.min(24, data.length);
  for (let i = 0; i < count; i++) sum += data[i];
  return (sum / (count * 255));
}

// ---- Visualizer Engines ----
function drawWaveform(ctx, w, h, style, palette, t) {
  if (style === 'off') return;
  const [top, bottom] = PALETTES[palette] || PALETTES.gold;
  ctx.globalAlpha = 1;

  if (style === 'bars') {
    const data = waveformData('freq');
    const bars = 48, step = Math.floor(data.length / (bars * 1.5));
    const gap = w / bars, barW = gap * 0.62;
    for (let i = 0; i < bars; i++) {
      const v = data[i * step] / 255;
      const barH = Math.max(3, v * h * 0.44);
      const x = i * gap + (gap - barW) / 2;
      const grad = ctx.createLinearGradient(0, h / 2 - barH, 0, h / 2 + barH);
      grad.addColorStop(0, top);
      grad.addColorStop(1, bottom);
      ctx.fillStyle = grad;
      // Rounded bar
      ctx.beginPath();
      ctx.roundRect(x, h / 2 - barH, barW, barH * 2, [3, 3, 3, 3]);
      ctx.fill();
      // Peak cap
      ctx.fillStyle = '#FFF8E7';
      ctx.fillRect(x, h / 2 - barH - 3, barW, 2);
    }
  } else if (style === 'wave') {
    const data = waveformData('time');
    ctx.beginPath();
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, bottom); grad.addColorStop(0.5, top); grad.addColorStop(1, bottom);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 3.5;
    ctx.shadowColor = top;
    ctx.shadowBlur = 10;
    for (let i = 0; i < data.length; i++) {
      const x = (i / data.length) * w;
      const y = h / 2 + (data[i] / 128 - 1) * h * 0.32;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  } else if (style === 'radial') {
    const data = waveformData('freq');
    const cx = w / 2, cy = h / 2;
    const baseR = Math.min(w, h) * 0.22;
    const bars = 72;
    const step = Math.floor(data.length / (bars * 1.2));
    for (let i = 0; i < bars; i++) {
      const angle = (i / bars) * Math.PI * 2;
      const v = data[i * step] / 255;
      const len = v * Math.min(w, h) * 0.22;
      const x1 = cx + Math.cos(angle) * baseR;
      const y1 = cy + Math.sin(angle) * baseR;
      const x2 = cx + Math.cos(angle) * (baseR + len);
      const y2 = cy + Math.sin(angle) * (baseR + len);
      ctx.strokeStyle = i % 2 === 0 ? top : bottom;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  } else if (style === 'pulse') {
    const data = waveformData('time');
    let sumSq = 0;
    for (let i = 0; i < data.length; i++) { const v = data[i] / 128 - 1; sumSq += v * v; }
    const rms = Math.sqrt(sumSq / data.length);
    const cx = w / 2, cy = h / 2;
    // Sacred lotus concentric rings
    [1, 0.72, 0.48, 0.28].forEach((f, idx) => {
      const r = Math.min(w, h) * (0.16 + rms * 0.5) * f;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = idx === 0 ? top : `${bottom}77`;
      ctx.lineWidth = idx === 0 ? 2.5 : 1.2;
      ctx.stroke();
      // Polygon facets
      const petals = 8;
      ctx.beginPath();
      for (let p = 0; p < petals; p++) {
        const a = (p / petals) * Math.PI * 2 + (t * 0.1 * (idx % 2 === 0 ? 1 : -1));
        const px = cx + Math.cos(a) * r;
        const py = cy + Math.sin(a) * r;
        if (p === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.strokeStyle = `${top}33`;
      ctx.stroke();
    });
  } else if (style === 'stardust') {
    const dust = ensureStardust(w, h);
    const bass = getBassRms();
    dust.stars.forEach(s => {
      s.x += s.vx * (1 + bass * 3);
      s.y += s.vy * (1 + bass * 3);
      if (s.x < 0) s.x = w; if (s.x > w) s.x = 0;
      if (s.y < 0) s.y = h; if (s.y > h) s.y = 0;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r * (1 + bass * 2), 0, Math.PI * 2);
      ctx.fillStyle = s.r > 2 ? top : bottom;
      ctx.fill();
    });
  } else if (style === 'lissajous') {
    const data = waveformData('time');
    const cx = w / 2, cy = h / 2;
    const rx = w * 0.3, ry = h * 0.25;
    ctx.beginPath();
    ctx.strokeStyle = top;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = top;
    ctx.shadowBlur = 8;
    for (let i = 0; i < data.length; i++) {
      const v = (data[i] / 128 - 1);
      const angle = (i / data.length) * Math.PI * 2;
      const x = cx + Math.sin(angle * 2 + t * 0.4) * (rx + v * 50);
      const y = cy + Math.cos(angle * 3 + t * 0.3) * (ry + v * 50);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

// Center Vinyl Disc
function drawVinylDisc(ctx, w, h, t, bass) {
  if (!document.getElementById('showVinylCheck').checked) return;
  const style = document.getElementById('vidStyle').value;
  if (style !== 'radial' && style !== 'pulse' && style !== 'wave') return;

  const cx = w / 2, cy = h / 2;
  const r = Math.min(w, h) * (0.18 + bass * 0.03);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(t * 0.4);

  // Outer black vinyl
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.fillStyle = '#11100E';
  ctx.fill();
  ctx.strokeStyle = 'rgba(193,166,115,0.4)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Grooves
  [0.85, 0.72, 0.6].forEach(f => {
    ctx.beginPath();
    ctx.arc(0, 0, r * f, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    ctx.stroke();
  });

  // Center Cover Art or Gold Emblem
  const centerR = r * 0.46;
  ctx.beginPath();
  ctx.arc(0, 0, centerR, 0, Math.PI * 2);
  ctx.clip();

  if (coverImage) {
    ctx.drawImage(coverImage, -centerR, -centerR, centerR * 2, centerR * 2);
  } else {
    ctx.fillStyle = '#C1A673';
    ctx.fillRect(-centerR, -centerR, centerR * 2, centerR * 2);
    ctx.fillStyle = '#090807';
    ctx.beginPath();
    ctx.arc(0, 0, centerR * 0.22, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}


// ---- Cinematic Video Overlays: Embers, Title Card & Vignette ----
const embers = [];
function initEmbers(count = 45) {
  embers.length = 0;
  for (let i = 0; i < count; i++) {
    embers.push({
      x: Math.random() * 1280,
      y: Math.random() * 720,
      r: Math.random() * 2.2 + 0.8,
      speedY: Math.random() * 0.6 + 0.2,
      sway: Math.random() * 2.0 + 0.8,
      phase: Math.random() * Math.PI * 2,
      alpha: Math.random() * 0.6 + 0.2
    });
  }
}
initEmbers();

function drawEmbers(ctx, w, h, time) {
  ctx.save();
  embers.forEach(e => {
    e.y -= e.speedY;
    e.x += Math.sin(time * 1.5 + e.phase) * (e.sway * 0.25);
    if (e.y < -10) { e.y = h + 10; e.x = Math.random() * w; }
    if (e.x < -10) e.x = w + 10;
    if (e.x > w + 10) e.x = -10;

    const pulse = 0.7 + 0.3 * Math.sin(time * 2 + e.phase);
    ctx.fillStyle = `rgba(212, 185, 122, ${e.alpha * pulse})`;
    ctx.shadowColor = 'rgba(212, 185, 122, 0.7)';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.arc(e.x * (w / 1280), e.y * (h / 720), e.r * (w / 1280), 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawTitleCard(ctx, w, h, elapsedSec) {
  if (elapsedSec > 4.5) return;
  const progress = elapsedSec / 4.5;
  let alpha = 1.0;
  if (progress < 0.2) alpha = progress / 0.2;
  else if (progress > 0.75) alpha = (1.0 - progress) / 0.25;

  const title = (document.getElementById('trackTitle').value || 'Tranquil Soul').trim();
  const artist = (document.getElementById('trackArtist').value || 'Tranquil Soul Music').trim();
  const fontFam = document.getElementById('vidFont').value || 'Cinzel';

  ctx.save();
  ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const centerY = h * 0.42;
  ctx.font = `300 ${Math.max(12, w * 0.016)}px "Space Grotesk", sans-serif`;
  ctx.fillStyle = '#C1A673';
  ctx.letterSpacing = '0.2em';
  ctx.fillText(artist.toUpperCase(), w / 2, centerY - (w * 0.035));

  ctx.font = `600 ${Math.max(20, w * 0.038)}px "${fontFam}", serif`;
  ctx.fillStyle = '#F2EFE9';
  ctx.shadowColor = 'rgba(0,0,0,0.85)';
  ctx.shadowBlur = 16;
  ctx.fillText(title, w / 2, centerY);

  ctx.strokeStyle = '#C1A673';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(w / 2 - (w * 0.08), centerY + (w * 0.03));
  ctx.lineTo(w / 2 + (w * 0.08), centerY + (w * 0.03));
  ctx.stroke();

  ctx.restore();
}

function drawVignetteAndGrain(ctx, w, h) {
  ctx.save();
  const grad = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.35, w / 2, h / 2, Math.max(w, h) * 0.75);
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, 'rgba(9,8,7,0.65)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
  ctx.restore();
}


function drawOverlays(ctx, w, h, t = 0) {
  const embersCheck = document.getElementById('overlayEmbers');
  const titleCardCheck = document.getElementById('overlayTitleCard');
  const vignetteCheck = document.getElementById('overlayVignette');

  if (embersCheck && embersCheck.checked) drawEmbers(ctx, w, h, t);
  if (titleCardCheck && titleCardCheck.checked) drawTitleCard(ctx, w, h, t);
  if (vignetteCheck && vignetteCheck.checked) drawVignetteAndGrain(ctx, w, h);

  const font = document.getElementById('vidFont').value;
  const title = document.getElementById('trackTitle').value.trim() || 'Untitled Chillout Track';
  const artist = document.getElementById('trackArtist').value.trim() || 'Tranquil Soul Music';
  const p = document.getElementById('player');

  // Title
  if (document.getElementById('showTitleCheck').checked) {
    ctx.save();
    ctx.textAlign = 'center';
    ctx.fillStyle = '#F2EFE9';
    ctx.font = `italic 400 ${Math.round(w * 0.048)}px "${font}", serif`;
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 8;
    ctx.fillText(title, w / 2, h * 0.12);
    ctx.restore();
  }

  // Artist
  if (document.getElementById('showArtistCheck').checked) {
    ctx.save();
    ctx.textAlign = 'center';
    ctx.fillStyle = '#D4B97A';
    ctx.font = `500 ${Math.round(w * 0.02)}px Montserrat, sans-serif`;
    ctx.letterSpacing = '0.12em';
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 6;
    ctx.fillText(artist.toUpperCase(), w / 2, h * 0.175);
    ctx.restore();
  }

  // Timecode
  if (document.getElementById('showTimecodeCheck').checked) {
    const cur = p && p.currentTime ? p.currentTime : 0;
    const dur = p && isFinite(p.duration) ? p.duration : 20;
    const fmtTime = s => {
      const m = Math.floor(s / 60);
      const sec = Math.floor(s % 60);
      return `${m < 10 ? '0' : ''}${m}:${sec < 10 ? '0' : ''}${sec}`;
    };
    ctx.save();
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(242,239,233,0.7)';
    ctx.font = `400 ${Math.round(w * 0.02)}px Montserrat, sans-serif`;
    ctx.fillText(`${fmtTime(cur)} / ${fmtTime(dur)}`, w * 0.05, h - h * 0.045);
    ctx.restore();
  }

  // Bottom Progress Scrubber
  if (document.getElementById('showProgressCheck').checked) {
    const cur = p && p.currentTime ? p.currentTime : 0;
    const dur = p && isFinite(p.duration) ? p.duration : 20;
    const frac = dur > 0 ? Math.min(1, cur / dur) : 0;
    ctx.fillStyle = 'rgba(255,255,255,0.1)';
    ctx.fillRect(0, h - 4, w, 4);
    ctx.fillStyle = '#C1A673';
    ctx.fillRect(0, h - 4, w * frac, 4);
  }

  // Watermark
  const wmMode = document.getElementById('vidWatermark').value;
  if (wmMode !== 'none') {
    const pos = document.getElementById('watermarkPos').value;
    let label = 'TRANQUILICY';
    if (wmMode === 'ring') label = '◎';
    if (wmMode === 'custom') label = document.getElementById('customWatermarkText').value.trim() || 'TRANQUILICY';

    ctx.save();
    ctx.globalAlpha = 0.6;
    ctx.fillStyle = '#D4B97A';
    ctx.font = `600 ${Math.round(w * 0.02)}px Montserrat, sans-serif`;
    let wx, wy, align;
    switch (pos) {
      case 'tl': wx = w * 0.05; wy = h * 0.07; align = 'left'; break;
      case 'tr': wx = w * 0.95; wy = h * 0.07; align = 'right'; break;
      case 'bl': wx = w * 0.05; wy = h - h * 0.045; align = 'left'; break;
      case 'bc': wx = w * 0.5; wy = h - h * 0.045; align = 'center'; break;
      default:   wx = w * 0.95; wy = h - h * 0.045; align = 'right'; break;
    }
    ctx.textAlign = align;
    ctx.fillText(label, wx, wy);
    ctx.restore();
  }
}

function drawExportFrame(t) {
  const canvas = document.getElementById('exportCanvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const bass = getBassRms();
  const pulseFactor = (parseFloat(document.getElementById('beatPulse').value) / 100) * 0.08;
  const scale = 1.0 + (bass * pulseFactor);

  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.scale(scale, scale);
  ctx.translate(-w / 2, -h / 2);

  drawBackdrop(ctx, w, h, t, document.getElementById('vidBackdrop').value, bass);
  drawWaveform(ctx, w, h, document.getElementById('vidStyle').value, document.getElementById('vidPalette').value, t);
  drawVinylDisc(ctx, w, h, t, bass);
  ctx.restore();

  drawOverlays(ctx, w, h, t);
}

// Preview Synchronization
['vidStyle', 'vidPalette', 'vidBackdrop', 'vidAspect', 'vidWatermark', 'watermarkPos', 'vidFont'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => {
    if (id === 'vidWatermark') {
      document.getElementById('customWatermarkField').style.display = document.getElementById('vidWatermark').value === 'custom' ? 'block' : 'none';
    }
    refreshPreview();
  });
});
['showTitleCheck', 'showArtistCheck', 'showTimecodeCheck', 'showProgressCheck', 'showVinylCheck'].forEach(id => {
  document.getElementById(id).addEventListener('change', refreshPreview);
});
document.getElementById('trackTitle').addEventListener('input', () => { refreshPreview(); updateDownloadNames(); });
document.getElementById('trackArtist').addEventListener('input', refreshPreview);
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

// Render Video with Web Audio and Stream
async function renderVideo() {
  const player = document.getElementById('player');
  const btn = document.getElementById('renderVidBtn');

  if (!player.src) { setVidStatus('Generate a track first.'); return; }
  if (typeof MediaRecorder === 'undefined') {
    setVidStatus('This browser cannot record video.');
    return;
  }

  btn.disabled = true;
  setChip('downloadVideoBtn', null);
  setVidStatus('Rendering — playing track once...');

  try {
    await renderVideoInner(player, btn);
  } catch (e) {
    setVidStatus('Render failed: ' + e.message);
    btn.disabled = false;
  }
}

async function renderVideoInner(player, btn) {
  await ensureAudioGraph();
  sizeExportCanvas();
  const canvas = document.getElementById('exportCanvas');

  const q = document.getElementById('vidQuality').value;
  const fps = q === 'ultra' ? 60 : 30;
  const bitrate = q === 'ultra' ? 8_000_000 : (q === 'fast' ? 2_500_000 : 5_000_000);

  const canvasStream = canvas.captureStream(fps);
  const combined = new MediaStream([...canvasStream.getVideoTracks(), ...mediaStreamDest.stream.getAudioTracks()]);
  const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus') ? 'video/webm;codecs=vp9,opus' : 'video/webm';
  const recorder = new MediaRecorder(combined, { mimeType, videoBitsPerSecond: bitrate });
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
    canvasStream.getTracks().forEach(t => t.stop());
  }

  player.addEventListener('ended', stopEverything);

  recorder.onstop = () => {
    if (lastVideoUrl) URL.revokeObjectURL(lastVideoUrl);
    const blob = new Blob(chunks, { type: 'video/webm' });
    lastVideoUrl = URL.createObjectURL(blob);
    setChip('downloadVideoBtn', lastVideoUrl, buildFilename('webm'), `Video · ${humanSize(blob.size)}`);
    setVidStatus('Ready in the bar below');
    btn.disabled = false;
    flow.rendered = true;
    renderFlow();
  };

  const lengthMode = document.getElementById('vidLength').value;
  const loopFactor = parseInt(document.getElementById('vidLoops') ? document.getElementById('vidLoops').value : '1') || 1;
  const fullDuration = isFinite(player.duration) && player.duration > 0 ? player.duration : 20;
  let singleLoopSec = fullDuration;
  if (lengthMode === 'loop15') singleLoopSec = Math.min(15, fullDuration);
  if (lengthMode === 'loop30') singleLoopSec = Math.min(30, fullDuration);

  let targetSec = singleLoopSec * loopFactor;
  let currentLoopIdx = 0;

  const onTimeUpdate = () => {
    if (player.currentTime >= singleLoopSec - 0.25) {
      currentLoopIdx++;
      if (currentLoopIdx < loopFactor) {
        player.currentTime = 0;
        player.play().catch(() => {});
        setVidStatus(`Rendering loop ${currentLoopIdx + 1} of ${loopFactor} (Extended Cut)...`);
      }
    }
  };
  player.addEventListener('timeupdate', onTimeUpdate);

  const origStop = stopEverything;
  stopEverything = function() {
    player.removeEventListener('timeupdate', onTimeUpdate);
    origStop();
  };

  player.currentTime = 0;
  recorder.start();
  loop();
  try {
    await player.play();
  } catch (e) {
    stopEverything();
    throw new Error('Could not start playback for recording');
  }
  setTimeout(stopEverything, targetSec * 1000 + 400);
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



// ---- Custom Luxury Gold Cursor ----
(function() {
  const dot = document.getElementById('customCursorDot');
  const ring = document.getElementById('customCursorRing');
  if (!dot || !ring) return;

  let mouseX = -100, mouseY = -100;
  let ringX = -100, ringY = -100;
  let isVisible = false;

  window.addEventListener('mousemove', e => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    if (!isVisible) {
      isVisible = true;
      document.body.classList.remove('cursor-hidden'); document.body.classList.add('has-custom-cursor');
      ringX = mouseX;
      ringY = mouseY;
    }
    dot.style.left = mouseX + 'px';
    dot.style.top = mouseY + 'px';

    const target = e.target;
    const isInteractive = target && target.closest ? target.closest('button, a, input, select, textarea, .knob, .tag-chip, .archetype-btn, .title-pill, .speed-btn, .toggle, .art-drop, #starChart, #exportCanvas') : false;
    ring.classList.toggle('hovering', !!isInteractive);
    const isDragging = document.body.classList.contains('dragging-knob') || document.body.classList.contains('dragging-star');
    ring.classList.toggle('dragging', isDragging);
  });

  document.addEventListener('mouseleave', () => {
    isVisible = false;
    document.body.classList.add('cursor-hidden');
  });

  window.addEventListener('mousedown', () => {
    ring.classList.add('clicking');
    dot.classList.add('clicking');
  });
  window.addEventListener('mouseup', () => {
    ring.classList.remove('clicking');
    dot.classList.remove('clicking');
  });

  function renderCursor() {
    ringX += (mouseX - ringX) * 0.22;
    ringY += (mouseY - ringY) * 0.22;
    ring.style.left = ringX + 'px';
    ring.style.top = ringY + 'px';
    requestAnimationFrame(renderCursor);
  }
  requestAnimationFrame(renderCursor);
})();


// ---- Local GPU API Endpoint Resolver ----
const DEFAULT_GPU_TUNNEL = 'https://clearly-gather-deviation-shorter.trycloudflare.com';

function getApiEndpoint() {
  const stored = (localStorage.getItem('tranquilicy_gpu_endpoint') || '').trim();
  if (stored && stored !== 'null' && stored !== 'undefined') {
    return stored.replace(/\/+$/, '');
  }
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return window.location.port === '8000' ? '' : 'http://127.0.0.1:8000';
  }
  if (window.location.protocol === 'file:') {
    return 'http://127.0.0.1:8000';
  }
  return DEFAULT_GPU_TUNNEL;
}
function getApiUrl(path) {
  const base = getApiEndpoint();
  return base ? (base + (path.startsWith('/') ? path : '/' + path)) : path;
}

async function checkGpuStatus() {
  const badge = document.getElementById('gpuBadge');
  const label = document.getElementById('gpuLabel');
  try {
    const res = await fetch(getApiUrl('/gpu'), { method: 'GET', headers: { 'Accept': 'application/json' } });
    if (res.ok) {
      const data = await res.json();
      badge.className = 'gpu-badge online';
      label.textContent = `${data.gpu_name || 'RTX 3090'} · Local GPU Online`;
      document.getElementById('gpuStatusText').textContent = `Connected: ${data.gpu_name} (${data.vram_allocated_gb}GB / ${data.vram_total_gb}GB VRAM in use)`;
      return true;
    }
  } catch(e) {}
  badge.className = 'gpu-badge offline';
  label.textContent = 'Local GPU Offline (Click to configure)';
  document.getElementById('gpuStatusText').textContent = 'Cannot reach GPU server. Make sure python 09_api_server.py is running.';
  return false;
}

function openGpuModal() {
  document.getElementById('gpuEndpointInput').value = getApiEndpoint();
  document.getElementById('gpuModal').style.display = 'flex';
  
// ---- Quota & Concurrency State Tracker ----
async function refreshQuota() {
  const quotaText = document.getElementById('quotaText');
  if (!quotaText) return;
  try {
    const res = await fetch(getApiUrl('/quota'));
    if (res.ok) {
      const data = await res.json();
      if (data.is_admin) {
        quotaText.textContent = 'Admin · Unlimited';
      } else {
        quotaText.textContent = `${data.generations_remaining}/${data.generations_max} Left Today`;
      }
    }
  } catch(e) {}
}

checkGpuStatus();
refreshQuota();
}

function closeGpuModal() {
  document.getElementById('gpuModal').style.display = 'none';
}

async function saveGpuEndpoint() {
  const val = document.getElementById('gpuEndpointInput').value.trim();
  localStorage.setItem('tranquilicy_gpu_endpoint', val);
  document.getElementById('gpuStatusText').textContent = 'Testing connection...';
  const ok = await checkGpuStatus();
  if (ok) {
    setTimeout(closeGpuModal, 800);
  }
}

// Initial GPU check
checkGpuStatus();
setInterval(checkGpuStatus, 15000);

</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
