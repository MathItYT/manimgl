from __future__ import annotations

import platform
import subprocess

from manimlib.utils.directories import get_sound_dir
from manimlib.utils.file_ops import find_file


def get_full_sound_file_path(sound_file_name: str) -> str:
    return find_file(
        sound_file_name,
        directories=[get_sound_dir()],
        extensions=[".wav", ".mp3", ""]
    )


def play_sound(sound_file: str, start_time: float = 0.0) -> subprocess.Popen:
    """Play a sound file from start_time and return its process handle."""
    full_path = get_full_sound_file_path(sound_file)

    if start_time > 0:
        return subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-ss", str(start_time), full_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    system = platform.system()
    if system == "Windows":
        return subprocess.Popen(
            ["powershell", "-c", f"(New-Object Media.SoundPlayer '{full_path}').PlaySync()"],
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    elif system == "Darwin":
        return subprocess.Popen(["afplay", full_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        return subprocess.Popen(["aplay", full_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
