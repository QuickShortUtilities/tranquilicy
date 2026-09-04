# Tranquilicy Studio

> **AI Chillout Music Generation & Mastering Suite**

Tranquilicy Studio is an ambient music production studio powered by Meta's `facebook/musicgen-medium` text-to-music model. It provides a browser-based studio interface designed with an obsidian-and-gold aesthetic, featuring deep DSP mastering chains, negative guidance prompt controls, real-time audio visualization, and direct video/artwork export.

---

## Features

### 🎵 AI Generation Engine
* **Long-Form Audio Chaining**: Overcomes MusicGen's ~30s limit by chaining continuation segments together seamlessly.
* **True StoppingCriteria Progress**: Real token-level generation progress reporting hooked directly into PyTorch/Transformers inference.
* **Negative Prompting / Classifier-Free Guidance**: Custom null-branch substitution in CFG pushing generation away from percussive clutter (e.g. drums, beats) rather than simple text prepending.
* **Interactive Dials & Star Chart**: Real-time control of tempo, resonance, duration, and multi-axis mood profiles.

### 🎚️ DSP Mastering Chain
* **Seamless Loops**: Equal-power ($\sqrt{\cdot}$) crossfades merging track tails over heads for click-free ambient loops.
* **Mid-Side Stereo Widening**: Enhances spatial depth while preserving 100% mono cancellation compatibility.
* **Loudness & Limiting**: Clean gain staging and peak limiting across presets (*Streaming*, *Subtle*, *Spacious*).
* **Multiple Export Formats**: Instant rendering to WAV, MP3, FLAC, and OGG.

### 🎨 Video & Artwork Rendering
* **Canvas Video Generator**: Direct browser-rendered WebM video with synchronized audio playback.
* **Waveform Styles**: Real-time Frequency Bars, Time-Domain Waveforms, or Radial Pulse animations.
* **Custom Backdrops & Palettes**: Gold, Obsidian, Amber, and Midnight color themes with customizable aspect ratios (16:9, 9:16, 1:1, 4:5).
* **Still Frame Export**: Instant high-resolution cover artwork snapshot.

---

## Quick Start

### 1. Requirements
* Python 3.10+
* NVIDIA GPU with CUDA recommended (PyTorch)
* `libsndfile` 1.1+ (for MP3 encoding)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/QuickShortUtilities/tranquilicy.git
cd tranquilicy

# Install dependencies
pip install -r requirements.txt
```

### 3. Launching the Studio
```bash
python scripts/09_api_server.py
```
Open **http://localhost:8000** in your browser.

---

## Live Web Interface (GitHub Pages)

The repository root includes `index.html`, which serves the Tranquilicy Studio frontend directly via GitHub Pages:
```
https://quickshortutilities.github.io/tranquilicy/
```
To enable GitHub Pages:
1. Navigate to **Repository Settings** > **Pages**.
2. Select **Source**: `Deploy from a branch`.
3. Choose branch: `main` / folder: `/ (root)`.
4. Click **Save**.

---

## Architecture

```
tranquilicy/
├── index.html              # Standalone Studio web UI (for GitHub Pages & local server)
├── manifest.csv            # Catalog & attribution manifest for reference audio
├── requirements.txt        # Python dependency specifications
├── PROJECT_NOTES.md        # Engineering notes, DSP test logs, and architecture details
├── scripts/
│   ├── 09_api_server.py    # Main FastAPI backend & model inference server
│   ├── test_dsp.py         # DSP audio chain unit test suite
│   ├── test_e2e.py         # End-to-end browser & API integration tests
│   ├── test_negative.py    # Negative guidance classifier verification
│   ├── check_page.py       # Static HTML/JS validation helper
│   ├── progress_util.py    # Generation progress utilities
│   └── ...                 # Audio chunking and dataset preparation scripts
└── musicgen-dreamboothing/ # LoRA training utilities and experiment notes
```

---

## License & Attribution

* MusicGen Medium is licensed under Meta's Research License.
* Reference audio and training manifests comply with Creative Commons (CC0 1.0, CC BY 4.0). Full source links and attributions are cataloged in `manifest.csv`.
