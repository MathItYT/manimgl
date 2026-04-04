"""OBS-related optional integrations for ManimGL."""

from manimlib.extras.obs.virtual_camera import VirtualCameraSink
from manimlib.extras.obs.virtual_camera import bind_scene_to_virtual_camera
from manimlib.extras.obs.virtual_camera import unbind_scene_from_virtual_camera

__all__ = [
    "VirtualCameraSink",
    "bind_scene_to_virtual_camera",
    "unbind_scene_from_virtual_camera",
]