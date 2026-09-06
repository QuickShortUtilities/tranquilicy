"""
Local web app for generating chillout music: loads MusicGen once, serves a
browser GUI styled to match the Tranquil Soul Music / Tranquilicy brand.
Supports durations beyond MusicGen's ~30s single-pass limit by chaining
continuation segments together, with a real (not simulated) progress bar
driven by a StoppingCriteria hook that fires on every generation step.

Usage:
    python 09_api_server.py
Then open local server in a browser.
"""
import io
import threading
import time
import os
import json
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse, FileResponse
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

# Every generated track is archived here -- it is the source material for the
# planned collective artwork. Deliberately NOT on C:, which had ~42GB free
# against a growth rate of ~3.4GB/day at full demand. Override with
# TRANQUILICY_OUTPUT_DIR; falls back to a local folder if the drive is missing.
OUTPUT_DIR = Path(os.environ.get("TRANQUILICY_OUTPUT_DIR", r"E:\tranquilicy_outputs"))
if not OUTPUT_DIR.drive or not Path(OUTPUT_DIR.drive + "\\").exists():
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

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
    return {
        "status": "ready",
        "engine": "Tranquil Soul Studio",
        "state": "operational",
        "version": APP_VERSION,
    }


# ---- Security, IP Quotas & GPU Concurrency Queue ---------------------------
# PUBLIC DEMO WEEK settings. gen_lock serialises the GPU, so exactly one track
# renders at a time and every limit below follows from that one fact.
# Measured throughput: ~2.75x realtime (a 25s track took 69s, a 6s track 18s),
# so the default 20s track is roughly 55s of GPU. That is ~65 tracks/hour if
# the queue never runs dry.
#
# Previous conservative values, to restore when the demo comes down:
#   per-IP 2, downloads 3, queue 2, global/day 35
MAX_GENERATIONS_PER_IP = 60
MAX_DOWNLOADS_PER_IP = 60   # exports; matches generations so everything you make can be exported
QUOTA_WINDOW_SEC = 3600.0   # 1 hour
# Queue depth is a wait-time budget, not a capacity dial: at ~55s a track,
# position 6 waits ~5.5 min. Deeper than this and people abandon anyway, so
# they are better served an honest "at capacity" than a queue they will leave.
MAX_QUEUE_WAITING = 15
# Circuit breaker: 400 x ~55s is ~6 GPU-hours/day worst case, which a 3090
# handles comfortably (it throttles itself before anything is at risk) and
# protects against a single script tying it up all day.
MAX_DAILY_GLOBAL_GENERATIONS = 999999

ip_quotas = {}  # ip -> {"generations": int, "downloads": int, "window_start": float, "active_job_id": str | None}

global_generations_today = 0
global_window_start = time.time()
quota_lock = threading.Lock()

STATS_FILE = "stats.json"
def load_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f).get("total_generations", 0)
        except:
            pass
    return 0

def save_stats(total):
    try:
        with open(STATS_FILE, "w") as f:
            json.dump({"total_generations": total}, f)
    except:
        pass

total_generations = load_stats()


def get_client_ip(request: Request) -> str:
    """Resolve the caller's IP, for QUOTA BUCKETING ONLY.

    x-real-ip comes first because the Worker sets it from the
    Cloudflare-validated client IP: CF masks cf-connecting-ip behind the
    Worker's own address, so without this every visitor shares one bucket.

    These are headers, so a determined caller can influence which bucket they
    land in -- the worst case being a fresh quota, which is a nuisance, not a
    breach. Privileges must NOT be decided here: see is_admin_request(), which
    ignores headers entirely. x-forwarded-for is not consulted at all, and
    uvicorn runs with proxy_headers=False so it cannot rewrite client.host.
    """
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"

def is_admin_ip(ip: str) -> bool:
    return ip in ("127.0.0.1", "::1", "localhost") or ip.startswith("192.168.") or ip.startswith("10.")


def is_admin_request(request: Request) -> bool:
    """Decide admin (unlimited GPU) from how the request ARRIVED, never from a
    header value.

    get_client_ip() reads x-real-ip / cf-connecting-ip, which is correct for
    bucketing quotas -- the Worker sets x-real-ip from the Cloudflare-validated
    client IP because CF masks cf-connecting-ip behind the Worker's own address.
    But those are still just headers, so deciding *admin* from them let anyone
    send `X-Real-IP: 127.0.0.1` and claim unlimited generations. (Confirmed
    locally: that header alone flips the identity used for quotas.)

    Any request that reached us through Cloudflare carries cf-connecting-ip,
    which the edge sets and refuses to let clients supply, and cloudflared adds
    it for tunnel traffic. Its ABSENCE is therefore the reliable signal that a
    caller is genuinely local -- and a remote caller cannot strip it.
    """
    if request.headers.get("cf-connecting-ip") or request.headers.get("cf-ray"):
        return False
    host = (request.client.host if request.client else "") or ""
    return is_admin_ip(host)

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

        print(f"[Tranquilicy] Job {job_id[:8]}: duration={duration_sec}s target_len={target_len} total_segments={total_segments}", flush=True)

        full_audio = None
        completed_segments = 0
        max_iterations = total_segments * 3 + 5
        audio_seed_buf = None  # renamed from seed to avoid clobbering the seed int param

        with gen_lock:
            if job_id in jobs:
                jobs[job_id]["started"] = True
            _negative_ctx["text"] = negative_prompt or None
            if seed is not None:
                torch.manual_seed(seed)
            iterations = 0
            while full_audio is None or len(full_audio) < target_len:
                iterations += 1
                if iterations > max_iterations:
                    print(f"[Tranquilicy] Job {job_id[:8]}: max_iterations={max_iterations} hit, stopping with {len(full_audio) if full_audio is not None else 0} samples", flush=True)
                    break
                remaining_samples = target_len - (len(full_audio) if full_audio is not None else 0)
                remaining_tokens = int(np.ceil(remaining_samples / (sr / FRAMES_PER_SEC))) + TOKEN_HEADROOM
                seg_max_tokens = max(MIN_SEGMENT_TOKENS, min(SEGMENT_NEW_TOKENS, remaining_tokens))

                print(f"[Tranquilicy] Job {job_id[:8]}: segment {iterations}/{total_segments} remaining={remaining_samples/sr:.1f}s tokens={seg_max_tokens}", flush=True)

                def on_step(step, _seg=completed_segments, _max=seg_max_tokens):
                    overall = (_seg + min(step / _max, 1.0)) / total_segments
                    jobs[job_id]["progress"] = min(overall, 0.99)

                stopper = StepCounter(on_step, lambda: jobs.get(job_id, {}).get("cancelled", False))
                if full_audio is None:
                    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
                else:
                    audio_seed_buf = full_audio[-seed_len:]
                    inputs = processor(text=[prompt], audio=audio_seed_buf, sampling_rate=sr, padding=True, return_tensors="pt").to(device)
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
                print(f"[Tranquilicy] Job {job_id[:8]}: seg raw={len(seg)} ({len(seg)/sr:.1f}s) seed_len={seed_len}", flush=True)

                if full_audio is None:
                    full_audio = seg
                else:
                    # Strip the conditioning echo from the front of the continuation output.
                    # If seg is shorter than seed_len the model stopped very early -- keep all of it.
                    new_part = seg[seed_len:] if len(seg) > seed_len else seg
                    if len(new_part) == 0:
                        print(f"[Tranquilicy] Job {job_id[:8]}: continuation returned 0 new samples, stopping chain", flush=True)
                        break
                    full_audio = np.concatenate([full_audio, new_part])

                completed_segments += 1
                print(f"[Tranquilicy] Job {job_id[:8]}: total audio now {len(full_audio)/sr:.1f}s / {duration_sec}s", flush=True)
                if len(full_audio) <= before:
                    print(f"[Tranquilicy] Job {job_id[:8]}: no growth in segment {iterations}, stopping chain", flush=True)
                    break

        if full_audio is None or len(full_audio) < sr * 0.5:
            raise RuntimeError("generation produced no usable audio")

        full_audio = full_audio[:target_len]
        print(f"[Tranquilicy] Job {job_id[:8]}: COMPLETE final={len(full_audio)/sr:.1f}s (requested {duration_sec}s)", flush=True)
        buf = io.BytesIO()
        sf.write(buf, full_audio, sr, format="WAV")
        audio_bytes = buf.getvalue()
        jobs[job_id]["audio"] = audio_bytes
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["done"] = True
        
        # Archive every track: this is the source material for the collective
        # artwork, so it is deliberately kept rather than discarded with the job.
        # Written to OUTPUT_DIR (see top of file) rather than the working
        # directory -- at 60 generations/hour this grows ~3.4GB/day, which would
        # fill the system drive within a week and take the server down with it.
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / f"{job_id}.wav").write_bytes(audio_bytes)
        except Exception as e:
            print(f"[warn] could not archive {job_id}: {e}")
    except Exception as e:
        import traceback
        print(f"[Tranquilicy] Job {job_id[:8]}: EXCEPTION {e}\n{traceback.format_exc()}", flush=True)
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["done"] = True



@app.post("/generate")
def generate(req: GenerateRequest, request: Request):
    prune_old_jobs()
    ip = get_client_ip(request)
    admin = is_admin_request(request)

    global global_generations_today, global_window_start, total_generations
    now = time.time()
    if now - global_window_start > QUOTA_WINDOW_SEC:
        global_generations_today = 0
        global_window_start = now

    # 1. Global Daily Circuit Breaker Check
    if not admin and global_generations_today >= MAX_DAILY_GLOBAL_GENERATIONS:
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Today's community demo quota ({MAX_DAILY_GLOBAL_GENERATIONS} tracks) has been reached. Relax and listen to Tranquil Soul Music on Spotify while slots reset!",
                "code": "GLOBAL_DAILY_LIMIT_REACHED"
            }
        )

    # 2. Concurrency queue capacity check
    waiting_jobs = [j for j in jobs.values() if not j["done"] and not j.get("started", False)]
    if not admin and len(waiting_jobs) >= MAX_QUEUE_WAITING:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Studio is currently at peak capacity ({len(waiting_jobs)} creators in queue). Please relax in the lounge!", "code": "QUEUE_FULL"}
        )

    # 3. Per-IP quotas and single active job check
    if not admin:
        q = get_or_create_quota(ip)
        # Check active job
        active_id = q.get("active_job_id")
        if active_id and active_id in jobs and not jobs[active_id]["done"]:
            return JSONResponse(
                status_code=429,
                content={"detail": "You already have a generation running or in queue. Please wait for it to complete.", "code": "JOB_IN_PROGRESS"}
            )

        # Check 2 generations limit for demo
        if q["generations"] >= MAX_GENERATIONS_PER_IP:
            mins_left = max(1, int((QUOTA_WINDOW_SEC - (now - q["window_start"])) / 60))
            return JSONResponse(
                status_code=429,
                content={"detail": f"Demo limit reached ({MAX_GENERATIONS_PER_IP}/{MAX_GENERATIONS_PER_IP} tracks). Resets in ~{mins_left}m.", "code": "QUOTA_EXCEEDED"}
            )
        q["generations"] += 1

    global_generations_today += 1
    total_generations += 1
    save_stats(total_generations)

    # Full duration support: allows 60s, 90s, 120s, 180s with multi-pass continuation chaining
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
        waiting_ahead = [j for j in jobs.values() if not j["done"] and not j.get("started", False) and j["created_at"] < job["created_at"]]
        queue_pos = len(waiting_ahead) + 1
        est_sec = queue_pos * 9

    return {
        "progress": job["progress"],
        "done": job["done"],
        "error": job["error"],
        "in_queue": in_queue,
        "queue_position": queue_pos,
        "estimated_sec": est_sec,
        "started": job.get("started", False)
    }


@app.get("/result/{job_id}")
def result(job_id: str, request: Request):
    job = jobs.get(job_id)
    if job is None or job["audio"] is None:
        return JSONResponse({"error": "not ready"}, status_code=404)
    
    # NOT rationed: the page fetches /result automatically after every
    # generation to load the track into the player, so counting it against the
    # download quota meant that once a visitor passed that limit their tracks
    # generated fine but could never be heard -- while the error told them
    # "audio remains playable in browser", which it was not. The generation
    # quota already bounds how many results can exist. The explicit export
    # (/master) is what carries the download quota.

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




@app.get("/capacity")
def capacity(request: Request):
    ip = get_client_ip(request)
    admin = is_admin_request(request)
    waiting_jobs = [j for j in jobs.values() if not j["done"] and not j.get("started", False)]
    active_jobs = [j for j in jobs.values() if not j["done"]]
    global global_generations_today, global_window_start, total_generations
    now = time.time()
    if now - global_window_start > QUOTA_WINDOW_SEC:
        global_generations_today = 0
        global_window_start = now

    circuit_tripped = not admin and global_generations_today >= MAX_DAILY_GLOBAL_GENERATIONS
    is_full = not admin and (len(waiting_jobs) >= MAX_QUEUE_WAITING or circuit_tripped)
    return {
        "is_full": is_full,
        "active_jobs": len(active_jobs),
        "waiting_jobs": len(waiting_jobs),
        "max_queue": MAX_QUEUE_WAITING,

        "daily_visitor_total": global_generations_today,
        "total_generations": total_generations,
        "daily_visitor_max": MAX_DAILY_GLOBAL_GENERATIONS,

        "circuit_tripped": circuit_tripped,
        "is_admin": admin
    }


@app.get("/lounge/info")
def lounge_info():
    return {
        "artist": "Tranquil Soul Music",
        "tracks": [
            {"id": "earth_pulse", "title": "Earth Pulse", "meta": "Deep Ambient Master · 24-Bit"},
            {"id": "sun_bleached_haze", "title": "Sun-Bleached Haze", "meta": "Warm Sunset Chill · 24-Bit"},
            {"id": "weightless_stillness", "title": "Weightless Stillness", "meta": "Zen Sanctuary Meditation · 24-Bit"}
        ],
        "spotify_url": "https://open.spotify.com/artist/4vAxYA9zh9HHWSVgOHvGrv",
        "spotify_embed": "https://open.spotify.com/embed/artist/4vAxYA9zh9HHWSVgOHvGrv?utm_source=generator&theme=0"
    }


@app.get("/lounge/track")
def lounge_track(track: str = "earth_pulse"):
    allowed = {
        "earth_pulse": "earth_pulse.mp3",
        "sun_bleached_haze": "sun_bleached_haze.mp3",
        "weightless_stillness": "weightless_stillness.mp3",
    }
    filename = allowed.get(track, "earth_pulse.mp3")
    track_path = Path(__file__).resolve().parent.parent / "assets" / "lounge" / filename
    if not track_path.exists():
        return Response(status_code=404, content="Lounge track not found")
    return FileResponse(track_path, media_type="audio/mpeg", headers={"Accept-Ranges": "bytes"})

@app.get("/quota")
def quota(request: Request):
    ip = get_client_ip(request)
    admin = is_admin_request(request)
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
def master(request: Request, job_id: str, preset: str = "streaming", fade_in: float = 0.0,
           fade_out: float = 0.0, seamless: bool = False, width: float = 0.0, fmt: str = "WAV",
           warmth: float = 0.0, air: float = 0.0):
    job = jobs.get(job_id)
    if job is None or job["audio"] is None:
        return JSONResponse({"error": "not ready"}, status_code=404)

    # The export is the thing worth rationing -- unlike /result, it is a
    # deliberate action, and re-encoding is real CPU work.
    if not is_admin_request(request):
        q = get_or_create_quota(get_client_ip(request))
        if q["downloads"] >= MAX_DOWNLOADS_PER_IP:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Export limit reached ({MAX_DOWNLOADS_PER_IP}). "
                                   f"Your tracks are still playable and downloadable as WAV."}
            )
        q["downloads"] += 1

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
    disk_html = Path(__file__).resolve().parent.parent / "index.html"
    if not disk_html.is_file():
        # Previously this silently fell back to a copy of the page embedded in
        # this file. That copy inevitably went stale (it ended up ~17k chars
        # adrift), so a missing index.html served a months-old app that looked
        # fine and quietly lacked half the features. Failing loudly is better.
        return HTMLResponse(
            "<h1>index.html is missing</h1><p>The studio page could not be found at "
            f"{disk_html}. It is the served page and must be present.</p>",
            status_code=500,
        )
    html = (
        disk_html.read_text(encoding="utf-8")
        .replace("__APP_VERSION__", APP_VERSION)
        .replace("__FORMAT_OPTIONS__", format_options)
    )
    # The page is edited and the server restarted constantly during development;
    # without this the browser can serve a cached copy and hide the new build.
    return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})


if __name__ == "__main__":
    # proxy_headers=False: uvicorn otherwise rewrites request.client.host from
    # X-Forwarded-For for connections from 127.0.0.1 -- which is exactly where
    # cloudflared connects from, so a caller-supplied XFF would decide the IP
    # that is_admin_ip() checks. We resolve the client IP explicitly from
    # cf-connecting-ip in get_client_ip(); request.client.host must stay the
    # real socket peer for the local-caller-is-admin fallback to mean anything.
    uvicorn.run(app, host="127.0.0.1", port=8000, proxy_headers=False)
