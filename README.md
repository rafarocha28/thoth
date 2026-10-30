# Thoth — video and audio transcription

Command-line tool that transcribes the videos/audios in a folder (using the
**Whisper** model via [faster-whisper](https://github.com/SYSTRAN/faster-whisper))
and generates, for each file:

- a **`.txt`** with the transcript as plain text;
- an **`.srt`** with timestamps (works as subtitles in any player);
- a **`consolidated.txt`** joining every transcript in the folder.

The spoken language is detected automatically, and it runs on an **NVIDIA GPU
when available**, falling back to CPU on its own.

Named after **Thoth** (or Tot), the Egyptian god of writing, language and
measurement — the scribe of the gods and keeper of records.

Tested on Windows 11 with a mid-range NVIDIA GPU (~10x realtime: a 1-hour
video is transcribed in ~6 minutes). It also works on CPU, just slower.

---

## Requirements

| Requirement | Notes |
|---|---|
| **Python 3.10 or newer** | https://www.python.org/downloads/ (on Windows, check "Add Python to PATH") |
| **ffmpeg on PATH** | See install commands per OS below |
| **NVIDIA GPU (optional)** | Windows and Linux only — macOS runs on CPU. Up-to-date driver |

> The first run downloads the chosen model (~3 GB for GPU's large-v3,
> ~500 MB for CPU's small). It is cached, so the following runs start instantly.

### Installing ffmpeg

| OS | Command |
|---|---|
| Windows | `winget install Gyan.FFmpeg` |
| Debian / Ubuntu | `sudo apt install ffmpeg` |
| Fedora / RHEL | enable [RPM Fusion](https://rpmfusion.org), then `sudo dnf install ffmpeg` |
| Arch / Manjaro | `sudo pacman -S ffmpeg` |
| macOS | `brew install ffmpeg` (needs [Homebrew](https://brew.sh)) |

## Installation

Open a terminal **inside this folder**.

**Windows:**

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux and macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **No NVIDIA GPU?** On any OS you can install just the essentials and run on
> CPU (this is the standard path on macOS, which has no CUDA support):
> ```bat
> pip install faster-whisper "av<14"
> ```

> **Linux with NVIDIA GPU:** after `pip install -r requirements.txt`, make the
> CUDA libraries visible to CTranslate2 before running:
> ```bash
> pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cublas, nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cublas.__file__), 'lib') + ':' + os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))")
> ```

## Usage

With the environment activated (`venv\Scripts\activate` on Windows,
`source venv/bin/activate` on Linux/macOS), the simplest way is to point it
at a folder:

```bash
python thoth.py "path/to/My Video Folder"
```

This scans the folder (including subfolders) and saves everything to
`My Video Folder\transcriptions\`: one `.txt` and one `.srt` per video,
plus `consolidated.txt` with everything joined. The subfolder structure is
mirrored inside `transcriptions\`.

More examples:

```bash
# Transcribe a single file
python thoth.py "single video.mp4"

# Force Portuguese instead of auto-detecting the language
python thoth.py "My Folder" --language pt

# Save to a different folder instead of 'transcriptions'
python thoth.py "Course X" --output "my transcripts"

# Top-level folder only, without descending into subfolders
python thoth.py "Course X" --no-subfolders

# No GPU (or just testing): run on CPU
python thoth.py "My Folder" --cpu

# Smaller, faster model (lower quality) for weak machines
python thoth.py "My Folder" --model medium
```

### Options

| Option | Default | Description |
|---|---|---|
| `path` | — (required) | Folder with media files, or a single file |
| `--model` | `large-v3` on GPU, `small` on CPU | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`... |
| `--language` | `auto` (detects) | Spoken language code (`pt`, `en`, `es`...) |
| `--device` | `auto` | `auto` (GPU if available), `cuda` or `cpu` |
| `--cpu` | — | Shorthand for `--device cpu` |
| `--output FOLDER` | `transcriptions` | Folder where the `.txt`/`.srt` files are saved |
| `--no-subfolders` | off | Do not include subfolders in the search |
| `--formats` | common video+audio | Accepted extensions: `mp4,mkv,avi,mov,mp3,m4a,wav...` |
| `--force` | off | Re-transcribe even if the `.txt` already exists |
| `--no-consolidated` | off | Do not generate `consolidated.txt` |

The script is **safe to re-run**: already-transcribed files are skipped
(ideal if a run is interrupted halfway — run it again and it resumes where
it left off).

## Choosing a model

| Model | Size | VRAM (GPU) | Quality | Speed |
|---|---|---|---|---|
| `tiny` | ~75 MB | ~1 GB | Low | Very fast |
| `base` | ~145 MB | ~1 GB | Low+ | Fast |
| `small` | ~490 MB | ~2 GB | Medium | Fast |
| `medium` | ~1.5 GB | ~5 GB | Good | Moderate |
| `large-v3` | ~3 GB | ~5 GB | **Excellent** | Moderate |

On GPUs with 8 GB or more, `large-v3` runs comfortably — it is the default
when a GPU is present. On CPU, the default is `small` (large-v3 on CPU would
take hours per video).

## Troubleshooting

**"ffmpeg not found"**
Install ffmpeg (commands in the table above) and reopen the terminal.

**Uses CPU even with an NVIDIA GPU (Windows/Linux)**
Check the driver first: `nvidia-smi` should list your GPU. Then verify that
torch CUDA is installed:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If it prints `False`, reinstall with `pip install -r requirements.txt`.
On Linux, also make sure the `LD_LIBRARY_PATH` export from the installation
section was applied in the current terminal.

**Error `open() got an unexpected keyword argument 'metadata_errors'`**
A newer PyAV version that is incompatible. Fix it with `pip install "av<14"`.

**Shows "No GPU available - using CPU"**
Installation without torch CUDA; for GPU run `pip install -r requirements.txt`
(plus the `LD_LIBRARY_PATH` step on Linux), or stay on CPU — the script picks
the `small` model on its own in that case.

**Slow/interrupted model download**
The model is downloaded from huggingface.co on the first run and cached in
`%USERPROFILE%\.cache\huggingface` (Windows) or `~/.cache/huggingface`
(Linux/macOS). Just run it again to resume.

**GPU out of memory (`CUDA out of memory`)**
Use a smaller model: `--model medium` or `--model small`.
