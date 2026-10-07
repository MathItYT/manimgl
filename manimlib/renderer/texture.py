from __future__ import annotations

import numpy as np
import wgpu

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from manimlib.renderer.gpu import Gpu


# How an image is read between its pixels, and what wgpu calls each, see Gpu.sampler
FILTER_MODES = {
    "linear": wgpu.FilterMode.linear,
    "nearest": wgpu.FilterMode.nearest,
}


def check_texture_filter(texture_filter: str) -> str:
    """The filter back again, where it is one there is, and otherwise an error saying so."""
    if texture_filter not in FILTER_MODES:
        raise ValueError(
            f"No such texture filter {texture_filter!r}, "
            f"expected one of {', '.join(map(repr, FILTER_MODES))}"
        )
    return texture_filter


class TextureSource(object):
    """
    Inert description of where one of a mobject's images comes from. Realized on the gpu at
    first draw, see Drawing.realize_textures.
    """

    def sharing_key(self) -> Any:
        """Key the realized texture is shared under, or None to never share it."""
        raise NotImplementedError

    def kind(self) -> str:
        """How the image is declared and bound, see shader_source.TEXTURE_KINDS."""
        return "2d"

    def realize(self, gpu: Gpu) -> Texture:
        """Build the gpu texture for this source."""
        raise NotImplementedError

    def copy(self) -> TextureSource:
        """
        The source a copy of the holding mobject draws from. Immutable sources are reused,
        writable ones duplicated.
        """
        return self


class ImageFile(TextureSource):
    """An image file, shared by every mobject naming that path."""

    def __init__(self, path: str):
        self.path = str(path)

    def sharing_key(self) -> str:
        return self.path

    def realize(self, gpu: Gpu) -> Texture:
        return Texture(self, gpu.texture(self.path).create_view())


class LayeredPixels(TextureSource):
    """
    Pixels as a stack of layers, one picked per draw by the shader, which is how a video
    holds its frames.

    With a key the stack is uploaded once and shared by every mobject naming it; without
    one it belongs to the holding mobject, which rewrites it.
    """

    def __init__(self, layers: np.ndarray, key: Any = None):
        self.layers = layers
        self.key = key
        self.version = 0

    def set_layers(self, layers: np.ndarray) -> None:
        self.layers = layers
        self.version += 1

    def sharing_key(self) -> Any:
        return self.key

    def kind(self) -> str:
        return "2d-array"

    def copy(self) -> LayeredPixels:
        # Shared stacks are read alike and reused; unshared ones are rewritten in place,
        # so a copy needs its own
        if self.key is not None:
            return self
        return LayeredPixels(self.layers)

    def realize(self, gpu: Gpu) -> Texture:
        # A keyed stack is uploaded once and every reader wraps that one view; an unkeyed
        # one is the holding mobject's alone, and is rewritten as its layers change
        if self.key is not None:
            return Texture(self, gpu.texture_stack(self.key, self.layers))
        return LayeredTexture(self, gpu)


class Texture(object):
    """A realized texture, held by the drawing that reads it."""

    def __init__(self, source: TextureSource, view: Any):
        self.source = source
        self.view = view

    def accepts(self, source: TextureSource) -> bool:
        """Whether this still matches the mobject's current source."""
        return source is self.source

    def refresh(self) -> None:
        """Upload whatever has changed, before the frame's pass. Static textures: nothing."""
        pass


def stack_view(texture: Any) -> Any:
    """
    A 2d-array view, which must be asked for explicitly: the default view is 2d whatever the
    texture holds, and a shader reading layers rejects it.
    """
    return texture.create_view(dimension=wgpu.TextureViewDimension.d2_array)



class BrowserVideoTexture(Texture):
    """GPU texture updated directly from an HTMLVideoElement."""

    def __init__(self, source: TextureSource, gpu: Gpu):
        self.gpu = gpu
        self.video = source.video.browser_video
        self.width = int(source.video.width)
        self.height = int(source.video.height)
        self.media_time = -1.0
        self.texture = gpu.device.create_texture(
            size=(self.width, self.height, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=(
                wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.RENDER_ATTACHMENT
            ),
        )
        super().__init__(
            source,
            self.texture.create_view(
                dimension=wgpu.TextureViewDimension.d2_array,
                array_layer_count=1,
            ),
        )

    def refresh(self) -> None:
        current_time = float(self.video.currentTime)
        if abs(current_time - self.media_time) <= 1e-6:
            return

        source = {"source": self.video}
        destination = {
            "texture": self.texture,
            "mip_level": 0,
            "origin": (0, 0, 0),
            "color_space": "srgb",
            "premultiplied_alpha": False,
        }
        copy_size = (self.width, self.height, 1)

        try:
            self.gpu.queue.copy_external_image_to_texture(
                source,
                destination,
                copy_size,
            )
        except NotImplementedError:
            # Older browser wheels expose the API on the abstract queue but
            # do not implement the forwarding method yet. Call the JS
            # WebGPU queue directly so the Manim browser bundle remains
            # compatible with those wheels.
            from wgpu.backends.js_webgpu._helpers import to_js_value

            self.gpu.queue._internal.copyExternalImageToTexture(
                to_js_value(source),
                to_js_value(destination),
                to_js_value(copy_size),
            )

        self.media_time = current_time


class LayeredTexture(Texture):
    """A stack the holding mobject rewrites, see LayeredPixels."""

    def __init__(self, source: LayeredPixels, gpu: Gpu):
        self.gpu = gpu
        self.shape = source.layers.shape[:3]
        self.texture = gpu.layer_texture(source.layers)
        self.version = -1
        super().__init__(source, stack_view(self.texture))

    def accepts(self, source: TextureSource) -> bool:
        return source is self.source and source.layers.shape[:3] == self.shape

    def refresh(self) -> None:
        source = self.source
        if source.version == self.version:
            return
        self.version = source.version
        self.gpu.write_layers(self.texture, source.layers)
