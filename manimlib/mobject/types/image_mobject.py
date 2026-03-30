from __future__ import annotations

import cv2
import numpy as np
import moderngl
from PIL import Image
import pathlib

from manimlib.constants import DL, DR, UL, UR
from manimlib.mobject.mobject import Mobject
from manimlib.utils.bezier import inverse_interpolate
from manimlib.utils.images import get_full_raster_image_path
from manimlib.utils.iterables import listify
from manimlib.utils.iterables import resize_with_interpolation

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from typing import Sequence, Tuple
    from manimlib.typing import Vect3


class ImageMobject(Mobject):
    shader_folder: str = "image"
    data_dtype: Sequence[Tuple[str, type, Tuple[int]]] = [
        ("point", np.float32, (3,)),
        ("im_coords", np.float32, (2,)),
        ("opacity", np.float32, (1,)),
    ]
    render_primitive: int = moderngl.TRIANGLES

    def __init__(
        self,
        filename: str | pathlib.Path | np.ndarray,
        height: float = 4.0,
        **kwargs,
    ):
        self.height = height
        self.image_path = (
            get_full_raster_image_path(filename)
            if isinstance(filename, (str, pathlib.Path))
            else (
                filename.tobytes(),
                filename.shape[1],
                filename.shape[0],
            )
        )
        self.image = (
            Image.open(self.image_path)
            if isinstance(filename, str)
            else Image.fromarray(filename)
        )
        super().__init__(
            texture_paths={"Texture": self.image_path}, **kwargs
        )

    def init_data(self) -> None:
        super().init_data(length=6)
        self.data["point"][:] = [UL, DL, UR, DR, UR, DL]
        self.data["im_coords"][:] = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (1, 0),
            (0, 1),
        ]
        self.data["opacity"][:] = self.opacity

    def init_points(self) -> None:
        size = self.image.size
        self.set_width(2 * size[0] / size[1], stretch=True)
        self.set_height(self.height)

    @Mobject.affects_data
    def set_opacity(self, opacity: float, recurse: bool = True):
        self.data["opacity"][:, 0] = resize_with_interpolation(
            np.array(listify(opacity)), self.get_num_points()
        )
        return self

    def set_color(self, color, opacity=None, recurse=None):
        return self

    def point_to_rgb(self, point: Vect3) -> Vect3:
        x0, y0 = self.get_corner(UL)[:2]
        x1, y1 = self.get_corner(DR)[:2]
        x_alpha = inverse_interpolate(x0, x1, point[0])
        y_alpha = inverse_interpolate(y0, y1, point[1])
        if not (0 <= x_alpha <= 1) and (0 <= y_alpha <= 1):
            # TODO, raise smarter exception
            raise Exception(
                "Cannot sample color from outside an image"
            )

        pw, ph = self.image.size
        rgb = self.image.getpixel(
            (
                int((pw - 1) * x_alpha),
                int((ph - 1) * y_alpha),
            )
        )[:3]
        return np.array(rgb) / 255


class VideoMobject(ImageMobject):
    def __init__(
        self,
        iterator: Iterator[np.ndarray],
        fps: float,
        **kwargs,
    ):
        self.iterator = iterator
        self.fps = fps
        self.update_each = 30 / self.fps
        super().__init__(next(iterator), **kwargs)
        self.frame_counter: float = 0.0
        self.passed_frames: int = 0
        self.speed = 1.0

    @staticmethod
    def updater(mob: "VideoMobject", dt: float):
        if mob.iterator is None:
            mob.stop()
            return
        try:
            mob.frame_counter += mob.speed * dt * mob.fps
            frame_delta = round(mob.frame_counter) - mob.passed_frames

            if frame_delta > 0:
                frame = None
                for _ in range(frame_delta):
                    frame = next(mob.iterator)

                mob.passed_frames = round(mob.frame_counter)

                if frame is not None:
                    # REGLA DE ORO (Zero-Copy) aplicada aquí
                    if mob.shader_wrapper is not None:
                        texture = mob.shader_wrapper.textures[0]
                        texture.write(frame)
                    else:
                        mob.texture_paths["Texture"] = frame
        except StopIteration:
            mob.iterator = None

    @classmethod
    def from_video(
        cls,
        video_path_or_camera: str | int,
        flip_horizontal: bool = False,
        **kwargs,
    ):
        cap = cv2.VideoCapture(video_path_or_camera)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # Fallback seguro para webcams

        def frame_reader():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if flip_horizontal:
                    frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                yield frame

        return cls(frame_reader(), fps, **kwargs)

    def play(self):
        self.add_updater(self.updater)
        return self

    def stop(self):
        self.remove_updater(self.updater)
        return self
