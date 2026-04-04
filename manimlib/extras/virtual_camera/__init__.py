"""Virtual camera optional integrations for ManimGL."""

from manimlib.extras.virtual_camera.sink import VirtualCameraSink
from manimlib.extras.virtual_camera.sink import bind_scene_to_virtual_camera
from manimlib.extras.virtual_camera.sink import unbind_scene_from_virtual_camera

__all__ = [
    "VirtualCameraSink",
    "bind_scene_to_virtual_camera",
    "unbind_scene_from_virtual_camera",
]