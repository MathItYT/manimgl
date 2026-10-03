from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache

import wgpu

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


# Stencil bits alongside the depth, which a fill counting windings needs
DEPTH_STENCIL_FORMAT = wgpu.TextureFormat.depth24plus_stencil8
COLOR_FORMAT = wgpu.TextureFormat.rgba8unorm

KEEP = ("keep", "keep", "keep")


# Color channels blend in the usual way, but the alpha channel takes the source's alpha
# whole, so that drawing something half transparent onto an opaque background leaves it
# opaque rather than eating into its alpha.
BLEND = {
    "color": {
        "src_factor": wgpu.BlendFactor.src_alpha,
        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
        "operation": wgpu.BlendOperation.add,
    },
    "alpha": {
        "src_factor": wgpu.BlendFactor.one,
        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
        "operation": wgpu.BlendOperation.add,
    },
}


@dataclass(frozen=True)
class PipelineState:
    """
    The fixed function half of a pipeline.

    depth_test:
        Whether depth testing is enabled. None means that the mobject decides.

    depth_write:
        Whether a fragment which passes the depth/stencil tests writes its
        depth into the depth buffer.

    depth_compare:
        Comparison used when depth testing is enabled.
    """

    depth_test: bool | None = None
    depth_write: bool = True
    depth_compare: str = "less"
    color_write: bool = True

    # What the stencil buffer is compared against.
    stencil_compare: str = "always"

    # What to leave in the stencil buffer when:
    #
    #   1. stencil fails
    #   2. stencil passes but depth fails
    #   3. both tests pass
    #
    # One tuple for front faces and one for back faces.
    stencil_ops: tuple[
        tuple[str, str, str],
        tuple[str, str, str],
    ] = (KEEP, KEEP)

    @lru_cache(maxsize=None)
    def resolved(
        self,
        depth_test: bool,
    ) -> PipelineState:
        """
        Settle the mobject's depth-test choice into the pipeline state.
        """

        if self.depth_test is not None:
            return self

        return replace(
            self,
            depth_test=depth_test,
        )

    def depth_stencil_descriptor(self) -> dict:
        """
        Convert this state into WebGPU's depth/stencil descriptor.
        """

        def face(ops):
            fail, depth_fail, passed = ops

            return {
                "compare": self.stencil_compare,
                "fail_op": fail,
                "depth_fail_op": depth_fail,
                "pass_op": passed,
            }

        front, back = self.stencil_ops

        return {
            "format": DEPTH_STENCIL_FORMAT,

            "depth_write_enabled": self.depth_write,

            # WebGPU still requires a comparison function even when depth
            # testing is disabled. "always" makes the depth test irrelevant.
            "depth_compare": (
                self.depth_compare
                if self.depth_test
                else "always"
            ),

            "stencil_front": face(front),
            "stencil_back": face(back),

            "stencil_read_mask": 0xFF,
            "stencil_write_mask": 0xFF,
        }

    @property
    def color_write_mask(self) -> int:
        return 0xF if self.color_write else 0


DEFAULT = PipelineState()


def build_pipeline(
    device: Any,
    layout: Any,
    module: Any,
    state: PipelineState,
    samples: int,
) -> Any:
    """
    Build one render pipeline.

    The shaders obtain their data directly from the renderer's storage
    buffers, hence the empty vertex-buffer list.
    """

    return device.create_render_pipeline(
        layout=layout,

        vertex={
            "module": module,
            "entry_point": "vs_main",
            "buffers": [],
        },

        fragment={
            "module": module,
            "entry_point": "fs_main",
            "targets": [{
                "format": COLOR_FORMAT,
                "blend": BLEND,
                "write_mask": state.color_write_mask,
            }],
        },

        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
        },

        depth_stencil=state.depth_stencil_descriptor(),

        multisample={
            "count": samples,
        },
    )
