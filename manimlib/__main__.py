#!/usr/bin/env python
from addict import Dict

from manimlib import __version__
from manimlib.config import manim_config
from manimlib.config import parse_cli
import manimlib.extract_scene
from manimlib.logger import log
from manimlib.utils.cache import clear_cache
from manimlib.window import Window


from IPython.terminal.embed import KillEmbedded


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argparse import Namespace


def _get_presenter_view_options(run_config: Dict) -> Dict | None:
    presenter_config = Dict(manim_config.get("presenter_view", {}))

    is_enabled = bool(
        presenter_config.get("enabled", False)
        or run_config.get("presenter_view", False)
    )
    if not is_enabled:
        return None

    host = (
        run_config.get("presenter_view_host")
        or presenter_config.get("host")
        or "127.0.0.1"
    )
    port = run_config.get("presenter_view_port")
    if port is None:
        port = presenter_config.get("port", 8765)

    try:
        port = int(port)
    except (TypeError, ValueError):
        port = 8765

    open_browser = bool(
        presenter_config.get("open_browser", False)
        or run_config.get("presenter_view_open_browser", False)
    )

    title = str(
        presenter_config.get("title", "ManimGL Presenter View")
    )
    max_notes = presenter_config.get("max_notes", 200)
    try:
        max_notes = int(max_notes)
    except (TypeError, ValueError):
        max_notes = 200

    notes_file = str(
        presenter_config.get("notes_file", "") or ""
    ).strip() or None

    return Dict(
        host=str(host),
        port=port,
        open_browser=open_browser,
        title=title,
        max_notes=max_notes,
        notes_file=notes_file,
    )


def run_scenes():
    """
    Runs the scenes in a loop and detects when a scene reload is requested.
    """
    # Create a new dict to be able to upate without
    # altering global configuration
    scene_config = Dict(manim_config.scene)
    run_config = manim_config.run
    presenter_view_options = _get_presenter_view_options(run_config)
    enable_window = manim_config.window.pop("enabled", True)
    if enable_window:
        window = Window(**manim_config.window)
    else:
        window = None
    scene_config.update(window=window)

    while True:
        try:
            # Blocking call since a scene may init an IPython shell()
            scenes = manimlib.extract_scene.main(scene_config, run_config)
            for scene in scenes:
                if presenter_view_options:
                    try:
                        from manimlib.extras.presenter_view import bind_scene_to_presenter_view

                        bind_scene_to_presenter_view(
                            scene, **presenter_view_options
                        )
                    except Exception:
                        log.exception("Failed to start presenter view")
                scene.run()
            return
        except KillEmbedded:
            # Requested via the `exit_raise` IPython runline magic
            # by means of the reload_scene() command
            pass
        except KeyboardInterrupt:
            break


def main():
    """
    Main entry point for ManimGL.
    """
    print(f"ManimGL \033[32mv{__version__}\033[0m")

    args = parse_cli()
    if args.version and args.file is None:
        return
    if args.clear_cache:
        clear_cache()

    run_scenes()


if __name__ == "__main__":
    main()
