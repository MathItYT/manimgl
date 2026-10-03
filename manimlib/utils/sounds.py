from __future__ import annotations

import subprocess
import threading
import platform
from typing import Any

from manimlib.utils.directories import get_sound_dir
from manimlib.utils.file_ops import find_file


def get_full_sound_file_path(sound_file_name: str) -> str:
    return find_file(
        sound_file_name,
        directories=[get_sound_dir()],
        extensions=[".wav", ".mp3", ""]
    )


def play_sound(sound_file, start_time: float = 0.0) -> Any:
    """Play a sound file using the system's audio player"""
    full_path = get_full_sound_file_path(sound_file)
    if start_time > 0:
        return subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-ss", str(start_time), full_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    system = platform.system()
    if system == "Windows":
        # Windows
        subprocess.Popen(
            ["powershell", "-c", f"(New-Object Media.SoundPlayer '{full_path}').PlaySync()"],
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    elif system == "Darwin":
        # macOS
        subprocess.Popen(
            ["afplay", full_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        subprocess.Popen(
            ["aplay", full_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


def stop_sound(process: Any) -> None:
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=0.5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
