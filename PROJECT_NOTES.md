# Chillout MusicGen — Project Notes

Last updated: 2026-09-04 · App version 1.7.0

The version number lives in `APP_VERSION` at the top of
`scripts/09_api_server.py` and is rendered into the page footer. Bump it when
shipping changes. The footer deliberately shows only branding and the version —
the model id and device were removed at the user's request.

## Demo week runbook (public deployment)

The live site is a Cloudflare Worker serving `index.html` and proxying API calls
to this machine's GPU over a Cloudflare tunnel.

**If the site says "GPU tunnel unreachable":**

1. Is the local server up? `netstat -ano | findstr :8000` — if not, launch it
   with the WMI command below.
2. Restart the tunnel and get the new hostname:
   `powershell -ExecutionPolicy Bypass -File scripts\start_tunnel.ps1`
3. Put that URL into the Worker variable **GPU_BACKEND**
   (dashboard → Workers & Pages → tranquilicy → Settings → Variables).
   It applies instantly — no git push, no redeploy.

**Why this keeps happening:** it is a *quick* tunnel
(`cloudflared tunnel --url ...`), which is assigned a new random hostname every
single time it starts, and it is not installed as a service so it does not
survive a reboot. `worker.js` only falls back to its hardcoded constant when
`GPU_BACKEND` is unset — so set the variable once and recovery is one field.

**The permanent fix** is a named tunnel: a stable hostname that survives
restarts, so `GPU_BACKEND` never changes again. It needs an interactive login
against a domain on your Cloudflare account, so it cannot be scripted here:

```
cloudflared tunnel login                            # browser; pick your domain
cloudflared tunnel create tranquilicy
cloudflared tunnel route dns tranquilicy gpu.<your-domain>
cloudflared tunnel run tranquilicy                  # then install as a service
```

**Demo limits** live at the top of `09_api_server.py` and are sized for public
traffic on one 3090 (~55s of GPU per default 20s track, serialised by
`gen_lock`): 6 generations per IP/day, 30 downloads, queue depth 6, 400/day
globally. The previous conservative values are recorded in a comment there —
restore them when the demo comes down.

**Admin bypass** (unlimited quota) is granted only to callers whose *real socket
address* is loopback/LAN. `get_client_ip()` trusts only `cf-connecting-ip`
(Cloudflare sets it and rejects client-supplied copies), and uvicorn runs with
`proxy_headers=False` so `X-Forwarded-For` cannot influence it. Verified: an
external caller sending `X-Forwarded-For: 127.0.0.1` stays non-admin.

## Goal

An iOS meditation/chillout music app ("Tranquilicy" / Tranquil Soul Music) that
generates ambient/chillout tracks on demand from a backend AI model, rather
than shipping a fixed library of pre-made tracks.

## Current state: base model, no fine-tuning

The app currently runs on the **unmodified** `facebook/musicgen-medium`
checkpoint from Hugging Face. Five separate LoRA fine-tuning attempts (see
"Fine-tuning attempts" below) all produced worse, more garbled output than
the base model, so fine-tuning was abandoned in favour of shipping the base
model and building the actual product around it.

## Running it

```
C:\Users\Gaming PC\chillout-musicgen\venv\Scripts\python.exe scripts\09_api_server.py
```

Then open **http://localhost:8000**. The model loads once at process start
(takes ~15-30s to load weights onto the 3090), and the server keeps it
resident in VRAM for all subsequent requests.

The process must run **detached** from any interactive terminal/session or
it dies when that session closes. `Start-Process` and Task Scheduler were
both tried and failed on this machine (job-object inheritance / Access
Denied respectively). The only reliable method found was launching via WMI:

```powershell
$venvPy = "C:\Users\Gaming PC\chillout-musicgen\venv\Scripts\python.exe"
$scriptDir = "C:\Users\Gaming PC\chillout-musicgen\scripts"
$cmdLine = "cmd.exe /c `"cd /d `"$scriptDir`" && `"$venvPy`" 09_api_server.py > `"D:\musicgen_data\api_stdout.log`" 2> `"D:\musicgen_data\api_stderr.log`"`""
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmdLine}
```

To restart after editing the script: find the PID on port 8000
(`netstat -ano | findstr :8000`), kill it, relaunch with the command above.

## The GUI (scripts/09_api_server.py)

Single-file FastAPI app: Python backend + one inline HTML/CSS/JS page. Styled
to match the real Tranquil Soul Music brand (colours, fonts, spacing pulled
from github.com/QuickShortUtilities/tranquilsoulmusic).

Features:
- **Prompt box at the top**, with a "Randomise" button that combines mood /
  scene / instrumentation-texture / time-of-day word banks into varied
  chillout prompts (built from scratch — not scraped from anywhere).
- **Six dials** (Atmosphere, Tempo, Warmth, Bass, Melody, Rhythm) with a live
  hexagonal "star chart" SVG visualisation. These shape the *text prompt*
  sent to the model (labelled "shape the sound", not "real DSP" — they don't
  do actual signal processing on the audio).
- **Exclude toggles**: no drums, no vocals, no bass — driven through real
  negative prompting via the guidance model's null branch, *not* by appending
  "no drums" to the prompt (see the negative-prompting section below).
- **Real progress bar**, driven by a `StoppingCriteria` hook that fires on
  every generation step (not a simulated/fake timer).
- **Long-form generation past MusicGen's ~30s single-pass limit**, up to
  300s, by chaining segments: each new segment is seeded with the last 3s of
  audio from the previous one and stitched together.
- **Download button** for the finished WAV.
- **Falling gold feathers** canvas background, in the brand's gold palette.
  (Checked the live site's `site.js`/`style.css` first — no such effect
  exists there to copy, so this was built fresh to match the aesthetic.)

The page is laid out as three equal-height columns presented as one numbered
flow — **01 Generate → 02 Master → 03 Video** — collapsing to 2 columns under
1180px and 1 column under 720px. A step rail above the columns mirrors the
same three states, and survives the responsive stacking (which is why the
progress indicator lives there rather than being drawn between grid columns).

Flow state lives in the `flow` object (`generated` / `mastered` / `rendered`);
`renderFlow()` is the single place that paints rail nodes, step badges and card
states from it. Steps 02 and 03 unlock together but are revealed staggered by
160ms. Starting a new generation resets `mastered`/`rendered`, since those were
completed against the previous track.

### The output bar (1.7.0)

The three columns are **controls only**. Everything that is a *result* lives in
one continuous bar spanning the full width beneath them — segmented boxes above,
one solid strip below:

- Left: the equaliser, a single status line, the always-present progress rail,
  and the audio player. The progress bar is permanently visible (empty at rest),
  which is what gives the bar its continuous underline.
- Right: four download **chips** — WAV, Master, Video, Still. A chip is dimmed
  and `pointer-events: none` until the artefact it points at actually exists,
  and shows format + file size once it does.

Each column's button therefore *produces* rather than downloads: Generate →
Master audio → Render video / Capture still frame. The output always lands as a
chip. This is why `downloadMastered()` and `saveStillFrame()` no longer trigger
a download directly — resist "helpfully" re-adding that, it puts downloads back
in two places.

Starting a new generation clears **all four** chips and revokes their blob URLs,
since every one of them belonged to the previous track. If a regeneration
*fails*, the bar restores the previous track and re-lights the WAV chip (that
blob URL is only revoked on success) rather than going blank.

Safety/correctness details worth knowing if touching this file again:
- `gen_lock` serialises all GPU generation calls — the model/device is not
  safe for concurrent requests.
- Segment loop has an iteration cap (`total_segments * 3 + 5`) so a
  short-returning generation can't spin forever.
- Finished jobs are pruned from the in-memory `jobs` dict after 30 minutes.
- Incoming prompts are clamped to 800 chars server-side.
- **Cancellation** works through the `StepCounter` StoppingCriteria: `POST
  /cancel/{job_id}` sets a flag that the criteria reads on its next generation
  step and returns `True`, which is the *only* point a running
  `model.generate()` can be interrupted. Without this a 180s request holds the
  GPU for minutes with no way out.
- `ensureAudioGraph()` must `await audioCtx.resume()` — a context created
  under browser autoplay policy starts suspended, and routing the player
  through a suspended context silences both playback *and* the recorded audio
  track, producing mute videos.
- The export canvas's on-page display size is deliberately decoupled from its
  internal recording resolution (CSS caps it at 300px tall; `canvas.width/height`
  stay at full export resolution). Don't "fix" this by setting CSS width/height
  to 100% — it would let a 9:16 preview stretch the whole three-column row.
- Starting a new generation must clear the previous track's export state
  (`downloadVideoBtn`, its blob URL, and both status lines). Otherwise the
  video button keeps serving the *old* render under the *new* track's filename.
- The Audio/Video cards use the `inert` attribute, not just
  `.disabled { pointer-events: none }` — pointer-events blocks the mouse but
  leaves every control inside still reachable with Tab.
- `/` is served with `Cache-Control: no-store`, because the edit-restart-refresh
  loop otherwise hands you a cached page and hides the new build.

### The segment-chaining stall (fixed in 1.4.0 — don't reintroduce it)

MusicGen's delay pattern spends the first few steps of each segment filling its
staggered codebooks, so a segment returns slightly **less** audio than
`max_new_tokens * 640` samples implies. Requesting exactly the remaining need
therefore always falls a fraction of a second short, and the loop then tries to
close a ~0.06s gap with a continuation budgeted at ~4 tokens — which emits no
new audio at all, so the loop makes no progress and the safety valve trips.
Net effect: **every short generation (5-6s) failed outright.**

Fixes, all of which need to stay:
- `TOKEN_HEADROOM` (12) added to the requested token count.
- `MIN_SEGMENT_TOKENS` (60) floor, so no segment is ever budgeted a sliver.
- A no-progress guard that breaks if a segment adds zero samples.
- Hitting the iteration cap now *keeps* the audio generated so far instead of
  raising and throwing away a good take.

This was a regression introduced by the earlier "only request the tokens you
actually need" efficiency change — before that, every segment over-generated
and got truncated, which accidentally masked the delay-pattern shortfall.

## Exclude toggles / negative prompting (1.6.0)

The three Exclude toggles used to append "no drums" etc. to the prompt, which
**does not work and can backfire**: MusicGen's T5 text encoder has no concept of
negation, so "no drums" simply feeds it the token *drums*.

They now drive real negative prompting. Classifier-free guidance already runs a
second "unconditional" branch that transformers fills with
`torch.zeros_like(last_hidden_state)` (in
`_prepare_text_encoder_kwargs_for_generation`). We wrap that method on the model
instance and substitute an encoded negative prompt for those zeros, plus a real
attention mask for that half — so guidance actively pushes *away* from the terms.

Why it's a wrapper rather than passing `encoder_outputs` ourselves: `generate()`
skips its encoder prep if `encoder_outputs` is already present, but then the
batch-size inference and the `repeat_interleave(2)` in
`prepare_inputs_for_generation` see a batch of 2 and double it again. Wrapping
leaves every batch-shape assumption untouched and only changes what's *in* the
null branch.

`_negative_ctx` is a module global read by that wrapper. It's assigned inside
`gen_lock` at the start of every run (to `None` when nothing is excluded), so a
value left behind by a failed run can never leak into the next one.

**Measured effectiveness** (`scripts/test_negative.py`, plus paired seed tests):
- Mechanism confirmed firing: same seed twice with no negative is *bit-identical*;
  adding the negative changes the audio. So it is definitely taking effect.
- Adversarial prompt that explicitly asks for drums: percussive energy reduced in
  3/5 paired seeds, mean −0.151.
- Realistic neutral chillout prompt: reduced in 3/4 paired seeds, mean −0.083,
  with large wins (one 0.268 → 0.015).

So it works substantially more often than not, but is **not a hard guarantee** —
the UI hint says as much rather than overpromising. Measurement uses librosa HPSS
percussive-energy ratio; note that single unpaired comparisons are worthless
here, run-to-run variance is enormous (0.04–0.95 for the same settings).

## Seeds

`/generate` accepts an optional `seed`. With it, generation is bit-for-bit
reproducible (verified), which is what makes paired A/B testing possible. Not
exposed in the UI yet — it's API-only.

## Audio export chain

`/master/{job_id}` takes `preset`, `fade_in`, `fade_out`, `seamless`, `width`
and `fmt`. Order is deliberate: **loop → widen → loudness → fades**. Loop runs
on untouched audio; widening happens before the limiter so the limiter catches
any peaks it adds; fades go last so nothing can lift the tails back off silence.
Seamless deliberately *overrides* fades — fading a looping track would
reintroduce the seam it just removed.

Two properties worth preserving:
- **Seamless loop** crossfades the tail over the head with equal-power (sqrt)
  ramps and drops the tail, so repeat playback runs `x[N-L-1] -> x[N-L]`, which
  is sample-contiguous in the source rather than a cut. Equal-power is correct
  because head and tail are uncorrelated material; linear ramps would dip.
- **Stereo widening** is built as `mid ± side`, so summing to mono cancels the
  side exactly and returns the original audio. That mono-compatibility is the
  whole reason it isn't a plain Haas delay, which combs on mono fold.

Formats offered are computed at startup from `sf.available_formats()` — MP3
needs libsndfile >= 1.1 (this machine has 1.2.2, so all four work).

## Tests

- `scripts/test_dsp.py` — DSP unit checks (loop seam continuity, equal-power
  level, mono-sum identity, fade endpoints, every format round-trip). No server
  or GPU needed.
- `scripts/test_e2e.py` — drives the **running** server exactly as the browser
  does: generates a short track, then exercises every export combination.
  This is what caught the chaining stall above; the DSP tests could not have.

## Checking the page without a JS runtime

There's no Node on this machine, so `scratchpad/check_page.py` parses the
inline page out of the server file and verifies: every `getElementById` target
actually exists in the HTML, there are no duplicate ids, and the script's
brackets balance (it understands strings, comments, and regex literals
including `/` inside character classes). Worth re-running after any edit to
the inline JS — it catches typo'd element ids, which otherwise fail silently
at runtime.

## Fine-tuning attempts (abandoned — kept for reference only)

Scripts `01_chunk_audio.py` through `08_generate_long.py`, plus the patched
copy of `musicgen-dreamboothing/dreambooth_musicgen.py`, are the LoRA
training pipeline built and run against progressively larger datasets
(v1 → v4, plus a tiny-overfit sanity check).

**Diagnosis**: a tiny-set overfit test (160 examples) should drive training
loss toward zero if the pipeline is correct. Instead loss plateaued at
~51-54, versus a random-guessing baseline of `ln(2048) ≈ 7.6` — i.e. training
was ~7x *worse* than random guessing, which pointed to corrupted or
misaligned training targets rather than a data/hyperparameter problem.

Root cause found: `apply_audio_decoder()` in `dreambooth_musicgen.py`
manually pre-applied the delay-pattern mask to labels, but
`MusicgenForCausalLM.forward()` already does this internally (per its own
docstring). This is a double-transformation of the training targets. The
fix was applied but made no measurable difference (54.44 → 54.40 loss),
meaning there's likely a second, still-unidentified bug in the label
pipeline. At that point the decision was made to stop debugging the trainer
and ship the base model instead.

Every fine-tuned checkpoint (v1-v4) sounded more garbled than the base model,
getting worse with more epochs/data — the opposite of what should happen if
training were working.

## Training data (for a future fine-tuning attempt, if ever revisited)

- ~5,505 tracks / 182GB consolidated at `D:\Audio 2`, originally the user's
  own AI-generated library (`C:\Users\Gaming PC\Desktop\sg`, ~3,552 tracks)
  plus supplementary CC-licensed downloads.
- CC-licensed supplementary tracks (176 total) sourced from Free Music
  Archive, Internet Archive, ccMixter, SoundCloud, Bandcamp,
  OpenGameArt/itch.io. **Only CC0 / CC-BY / CC-BY-SA** — never NC or ND —
  verified per-track, not inferred from site-wide licensing claims.
- Captioning (`02_caption_chunks.py`) prefers real tags from
  `<title> lyrics.txt` sidecar files when available, falls back to
  tempo/energy auto-analysis + filename hints otherwise. Also flags
  vocal/instrumental status so vocal-heavy tracks can be excluded from
  training an instrumental-only model.

## Gotchas hit and fixed along the way (for anyone touching the pipeline again)

- `datasets` audiofolder split auto-detection silently drops files: any
  filename containing "train" or "val" as a substring *anywhere* (e.g.
  `jungle-train.wav`, `val-davis.wav`) gets misclassified into the wrong
  split, and did so silently — dropped 32,385 files down to 33 examples with
  no error. Fixed by renaming offending files.
- Orphaned excluded-vocal wav files left on disk after being dropped from
  `metadata.csv` crash `datasets` (audiofolder requires strict 1:1 file/row
  coverage) — must delete the file, not just the metadata row.
- `HF_DATASETS_CACHE` must be set inside the WMI-spawned `cmd.exe`'s own
  environment (`set VAR=value&&` in the command line string) — WMI does
  **not** inherit the launching PowerShell session's `$env:` vars. Missed
  this once and it silently filled the C: drive.
- AppleDouble `._*` macOS junk files double file counts if not filtered out.
