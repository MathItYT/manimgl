"""Browser presenter controls for ManimGL scenes."""

from manimlib.extras.presenter_view.controller import PresenterViewController
from manimlib.extras.presenter_view.controller import bind_scene_to_presenter_view
from manimlib.extras.presenter_view.controller import unbind_scene_from_presenter_view

__all__ = [
    "PresenterViewController",
    "bind_scene_to_presenter_view",
    "unbind_scene_from_presenter_view",
]
