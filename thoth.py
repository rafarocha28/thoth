"""
Thoth - video/audio transcription with Whisper (faster-whisper).

Named after the Egyptian god of writing and records, the scribe of the gods.

Point it at a media file or a folder and it transcribes everything,
generating a .txt and an .srt (with timestamps) for each file.
In folder mode, it also creates a consolidated.txt with every transcript.

Examples:
    python thoth.py "My Video Folder"
    python thoth.py "single video.mp4"
    python thoth.py "Course X" --language pt
    python thoth.py "Course X" --model medium --no-subfolders
    python thoth.py "Course X" --cpu
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_FORMATS = "mp4,mkv,avi,mov,m4v,webm,wmv,ts,mp3,m4a,wav,flac,ogg"


def enable_cuda_dlls() -> None:
    """Expose the CUDA/cuDNN DLLs bundled with torch to CTranslate2 (Windows/GPU).

    Only has an effect when torch is installed; ignored on CPU-only setups.
    """
    if os.name != "nt":
        return
    try:
        import torch  # noqa: F401
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
    except ImportError:
        pass


def pick_device(requested: str) -> str:
    """Resolve 'auto' to cuda (if a GPU is available) or cpu, with safe fallback."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return "cuda"
    except ImportError:
        pass
    except Exception as e:
        print(f"[warning] GPU present but unavailable ({e}); using CPU.")
    print("No GPU available - using CPU (slower).")
    return "cpu"


def pick_model(requested: str | None, device: str) -> str:
    """Without --model, use large-v3 on GPU and small on CPU (the only viable one there)."""
    if requested:
        return requested
    return "large-v3" if device == "cuda" else "small"


def format_timestamp(seconds: float) -> str:
    """Convert seconds to the SRT timestamp format (00:00:00,000)."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_audio(path: Path) -> Path:
    """Extract the audio as 16 kHz mono WAV (Whisper's ideal input)."""
    wav = Path(tempfile.gettempdir()) / f"{path.stem}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return wav


def transcribe_file(model, path: Path, dest_base: Path, language: str | None) -> None:
    """Transcribe one media file and write a .txt and an .srt with the same base name."""
    wav = extract_audio(path)
    try:
        segments, info = model.transcribe(
            str(wav),
            language=language,
            vad_filter=True,
        )
        lines = []
        srt_parts = []
        for i, seg in enumerate(segments, 1):
            pct = 100 * seg.end / max(info.duration, 1)
            print(f"\r  {pct:5.1f}%  (segment {i})", end="", flush=True)
            text = seg.text.strip()
            lines.append(text)
            srt_parts.append(
                f"{i}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{text}\n"
            )
        print()
        dest_base.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        dest_base.with_suffix(".srt").write_text("\n".join(srt_parts) + "\n", encoding="utf-8")
    finally:
        wav.unlink(missing_ok=True)


def consolidate(output_dir: Path) -> None:
    """Write consolidated.txt joining every .txt in the output folder."""
    txts = [
        t for t in sorted(output_dir.rglob("*.txt"))
        if t.is_file() and "consolidated" not in t.stem.lower()
    ]
    if not txts:
        return
    parts = [f"# CONSOLIDATED - {output_dir.parent.name}\n"]
    for txt in txts:
        content = txt.read_text(encoding="utf-8").strip()
        parts.append(f"\n\n---\n\n## {txt.stem}\n\n{content}\n")
    dest = output_dir / "consolidated.txt"
    dest.write_text("".join(parts), encoding="utf-8")
    print(f"[consolidated] {len(txts)} transcript(s) -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe videos/audios using faster-whisper "
                    "(one .txt and one .srt per file).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  python thoth.py "My Video Folder"\n'
            '  python thoth.py "single video.mp4"\n'
            '  python thoth.py "Course X" --language pt\n'
            '  python thoth.py "Course X" --model medium --no-subfolders\n'
            '  python thoth.py "Course X" --cpu\n'
        ),
    )
    parser.add_argument("path", help="media file, or folder with media files")
    parser.add_argument("--model", default=None, metavar="MODEL",
                        help="tiny, base, small, medium, large-v3... "
                             "(default: large-v3 on GPU, small on CPU)")
    parser.add_argument("--language", default="auto", metavar="CODE",
                        help="spoken language code (pt, en, es...) or 'auto' to detect "
                             "(default: auto)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="execution device (default: auto - uses GPU when available)")
    parser.add_argument("--cpu", action="store_true",
                        help="shorthand for --device cpu")
    parser.add_argument("--output", default=None, metavar="FOLDER",
                        help="folder to save the .txt/.srt files "
                             "(default: 'transcriptions' inside the given folder)")
    parser.add_argument("--no-subfolders", action="store_true",
                        help="do not descend into subfolders (everything is scanned by default)")
    parser.add_argument("--formats", default=DEFAULT_FORMATS,
                        help=f"accepted extensions, comma-separated (default: {DEFAULT_FORMATS})")
    parser.add_argument("--force", action="store_true",
                        help="re-transcribe even if the .txt already exists")
    parser.add_argument("--no-consolidated", action="store_true",
                        help="do not generate consolidated.txt at the end")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("Error: ffmpeg not found. Install it and make sure it is on the PATH.\n"
                 "Windows: winget install Gyan.FFmpeg")

    path = Path(args.path).resolve()
    if path.is_file():
        base_dir = path.parent
        is_folder = False
        files = [path]
    elif path.is_dir():
        base_dir = path
        is_folder = True
        formats = {f.strip().lower().lstrip(".") for f in args.formats.split(",") if f.strip()}
        glob = base_dir.glob if args.no_subfolders else base_dir.rglob
        files = sorted(
            p for p in glob("*.*")
            if p.is_file() and p.suffix.lower().lstrip(".") in formats
        )
    else:
        sys.exit(f"Error: path not found: {path}")

    if not files:
        sys.exit(f"No media files found in {base_dir}")

    output_dir = Path(args.output).resolve() if args.output else base_dir / "transcriptions"
    output_dir.mkdir(parents=True, exist_ok=True)

    enable_cuda_dlls()
    from faster_whisper import WhisperModel  # imported after configuring the DLLs

    requested_device = "cpu" if args.cpu else args.device
    device = pick_device(requested_device)
    model_name = pick_model(args.model, device)
    compute_type = "float16" if device == "cuda" else "int8"
    language = None if args.language == "auto" else args.language

    print(f"Loading model '{model_name}' ({device}, {compute_type})...")
    print("(the model is downloaded on first run - this may take a while)\n")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    print(f"Found {len(files)} file(s) to transcribe.\n")
    done = skipped = errors = 0
    for media_path in files:
        dest_base = output_dir / media_path.relative_to(base_dir)
        dest_base.parent.mkdir(parents=True, exist_ok=True)
        dest_txt = dest_base.with_suffix(".txt")
        if dest_txt.exists() and not args.force:
            print(f"[already exists] {dest_txt.relative_to(output_dir)}")
            skipped += 1
            continue

        print(f"[transcribing] {media_path.relative_to(base_dir)}")
        try:
            transcribe_file(model, media_path, dest_base, language)
        except subprocess.CalledProcessError:
            print("  [ffmpeg error] check that the file is not corrupted")
            errors += 1
            continue
        except KeyboardInterrupt:
            dest_base.with_suffix(".txt").unlink(missing_ok=True)
            dest_base.with_suffix(".srt").unlink(missing_ok=True)
            raise
        print(f"[ok] {dest_txt}\n")
        done += 1

    print(f"Done: {done} transcribed, {skipped} already existed, {errors} error(s).")

    if is_folder and not args.no_consolidated:
        consolidate(output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
