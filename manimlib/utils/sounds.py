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


def play_sound(sound_file: str) -> subprocess.Popen:
    """Play a sound file and return its process handle."""
    full_path = get_full_sound_file_path(sound_file)
    system = platform.system()

    if system == "Windows":
        return subprocess.Popen(
            ["powershell", "-c", f"(New-Object Media.SoundPlayer '{full_path}').PlaySync()"],
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    elif system == "Darwin":
        return subprocess.Popen(
            ["afplay", full_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        return subprocess.Popen(
            ["aplay", full_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

