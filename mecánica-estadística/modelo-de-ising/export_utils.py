from __future__ import annotations

import subprocess
from pathlib import Path


def ensure_public_dir(notebook_dir: Path) -> Path:
    out = notebook_dir / "public"
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_mp4_from_frames(frames_glob: str, output_file: Path, fps: int = 10) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        frames_glob,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_file),
    ]
    subprocess.run(cmd, check=True)
