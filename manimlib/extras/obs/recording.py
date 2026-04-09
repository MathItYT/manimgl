from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from manimlib.logger import log

if TYPE_CHECKING:
    from manimlib.scene.scene import Scene
    from addict import Dict


_HAS_LAUNCHED_OBS_THIS_PROCESS: bool = False


def _normalize_cli_args(args: object) -> list[str]:
    if args is None:
        return []
    if isinstance(args, str):
        # Treat a single string as one arg (avoid naive splitting).
        return [args]
    try:
        return [str(a) for a in args]
    except TypeError:
        return [str(args)]


def _candidate_obs_paths_windows() -> list[Path]:
    candidates: list[Path] = []
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if not base:
            continue
        candidates.extend(
            [
                Path(base) / "obs-studio" / "bin" / "64bit" / "obs64.exe",
                Path(base) / "OBS Studio" / "bin" / "64bit" / "obs64.exe",
            ]
        )
    return candidates


def _find_obs_executable(explicit_path: str | None = None) -> str | None:
    # 1) Explicit path
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return str(p)

    # 2) Env overrides
    for env in ("OBS_STUDIO_PATH", "OBS_PATH"):
        val = os.environ.get(env)
        if val:
            p = Path(val)
            if p.is_file():
                return str(p)

    system = platform.system().lower()

    # 3) Windows common install locations
    if system.startswith("windows"):
        for p in _candidate_obs_paths_windows():
            if p.is_file():
                return str(p)
        for name in ("obs64.exe", "obs64", "obs.exe", "obs"):
            found = shutil.which(name)
            if found:
                return found
        return None

    # 4) macOS
    if system.startswith("darwin"):
        mac_path = Path("/Applications/OBS.app/Contents/MacOS/OBS")
        if mac_path.is_file():
            return str(mac_path)
        found = shutil.which("obs") or shutil.which("obs-studio")
        return found

    # 5) Linux / others
    return shutil.which("obs") or shutil.which("obs-studio")


def _build_obs_command(obs_executable: str, obs_config: "Dict") -> list[str]:
    cmd: list[str] = [obs_executable]

    if bool(obs_config.get("minimize_to_tray", True)):
        cmd.append("--minimize-to-tray")

    cmd.extend(_normalize_cli_args(obs_config.get("extra_args")))

    if bool(obs_config.get("start_recording", True)):
        cmd.append("--startrecording")

    if bool(obs_config.get("start_streaming", False)):
        cmd.append("--startstreaming")

    return cmd


def autostart_obs_recording(scene: "Scene") -> None:
    """Launch OBS Studio and start recording at scene start.

    Controlled via config key `obs.autostart_recording`.

    This uses OBS' CLI flags (no websocket dependency). It is best-effort:
    - If OBS is already running, behaviour depends on OBS' single-instance handling.
    - Starting/stopping recording in an already-running OBS instance generally
      requires OBS WebSocket, which is out of scope for this helper.
    """

    global _HAS_LAUNCHED_OBS_THIS_PROCESS

    obs_config = getattr(getattr(scene, "manim_config", None), "obs", None)
    # Scenes don't currently expose manim_config; fall back to module global.
    if obs_config is None:
        from manimlib.config import manim_config  # local import to avoid cycles

        obs_config = manim_config.get("obs", None)

    if not obs_config or not bool(obs_config.get("autostart_recording", False)):
        return

    # Avoid launching OBS during preruns or quiet passes.
    if bool(scene.file_writer_config.get("quiet", False)):
        return

    if _HAS_LAUNCHED_OBS_THIS_PROCESS:
        return

    obs_executable = _find_obs_executable(str(obs_config.get("executable") or "") or None)
    if not obs_executable:
        log.warning(
            "OBS autostart is enabled, but OBS executable was not found. "
            "Set `obs.executable` in your custom config to the full path to OBS."
        )
        return

    cmd = _build_obs_command(obs_executable, obs_config)

    obs_cwd = None
    try:
        obs_cwd = str(Path(obs_executable).resolve().parent)
    except Exception:
        obs_cwd = None

    startup_wait = float(obs_config.get("startup_wait_seconds", 0.0) or 0.0)

    try:
        popen_kwargs = dict(
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if obs_cwd:
            # OBS relies on relative paths to find data/obs-studio/locale/*.
            # When launched from another working directory, it can fail with
            # "Failed to find locale/en-US.ini".
            popen_kwargs["cwd"] = obs_cwd
        if os.name == "nt":
            creationflags = 0
            # Detach so OBS doesn't die if the parent process exits.
            creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
            creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            popen_kwargs["creationflags"] = creationflags
        else:
            popen_kwargs["start_new_session"] = True

        subprocess.Popen(cmd, **popen_kwargs)
        _HAS_LAUNCHED_OBS_THIS_PROCESS = True

        if startup_wait > 0:
            time.sleep(startup_wait)

        log.info("Launched OBS Studio for automatic recording")

    except Exception:
        log.exception("Failed to launch OBS Studio for automatic recording")
