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
