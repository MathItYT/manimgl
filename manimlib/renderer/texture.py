from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from manimlib.renderer.gpu import Gpu


class TextureSource(object):
    """
    Inert description of where one of a mobject's images comes from. Realized on the gpu at
    first draw, see Drawing.realize_textures.
    """

    def sharing_key(self) -> Any:
        """Key the realized texture is shared under, or None to never share it."""
        raise NotImplementedError

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
