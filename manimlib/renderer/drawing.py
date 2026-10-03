from __future__ import annotations

import numpy as np
import wgpu

from manimlib.renderer.pipeline import DEFAULT
from manimlib.renderer.pipeline import KEEP
from manimlib.renderer.pipeline import PipelineState
from manimlib.renderer.shader_source import MOBJECT_GROUP
from manimlib.renderer.shader_source import RESOURCE_GROUP
from manimlib.renderer.shader_source import read_shader_file

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from manimlib.mobject.mobject import Mobject
    from manimlib.renderer.gpu import RenderPass
    from manimlib.renderer.material import Material, ModuleSpec
    from manimlib.renderer.texture import Texture


class Drawing(object):
    merges = False
    records_between = 0

    @classmethod
    def draws(cls, mobject: Mobject) -> bool:
        return bool(mobject.shader_file)

    @classmethod
    def key(cls, mobject: Mobject) -> tuple:
        return (
            cls,
            mobject.shader_file,
            mobject.data.dtype,
            mobject.uniforms.dtype,
            tuple(
                (name, source.kind())
                for name, source in mobject.textures.items()
            ),
            tuple(
                mobject.shader_code_replacements.items()
            ),
            mobject.verts_per_record,
        )

    @classmethod
    def module_specs(
        cls,
        mobject: Mobject,
    ) -> list[ModuleSpec]:
        return [
            (
                "main",
                mobject.shader_file,
                mobject.shader_code_replacements,
            )
        ]

    def __init__(
        self,
        material: Material,
        mobject: Mobject,
    ):
        self.material = material
        self.mobject = mobject

        self.data = mobject.data
        self.uniforms = mobject.uniforms

        self.replacements = (
            mobject.shader_code_replacements
        )

        self.uniform_offset = -1
        self.data_offset = -1

        self.uniform_version = 0
        self.data_version = 0

        self.data_repeats = 0
        self.repeats_uniforms = False
        self.records = 0

        self.depth_test = False

        self.own_bind_group: Any = None
        self.shared_bind_group: Any = None

        self.textures: dict[str, Texture] = dict()
        self.texture_views: list[Any] = []

        self.sampler: Any = None

        self.invalidated = True

        self.members: list[Drawing] | None = None

    # -------------------------------------------------------------------------
    # Uniforms
    # -------------------------------------------------------------------------

    def write_uniforms(self) -> bool:
        depth_test = self.mobject.depth_test

        self.invalidated = (
            depth_test != self.depth_test
        )

        self.depth_test = depth_test

        buffer = self.material.uniform_buffer

        version = self.uniforms.version
        changed = version != self.uniform_version

        self.uniform_version = version

        offset = buffer.claim(
            self.uniforms.array.nbytes
        )

        moved = (
            offset != self.uniform_offset
        )

        if changed or moved:
            buffer.put(
                offset,
                self.uniforms.bytes,
            )

        self.invalidated = (
            self.invalidated
            or moved
        )

        self.uniform_offset = offset

        return changed

    # -------------------------------------------------------------------------
    # Records
    # -------------------------------------------------------------------------

    def write_records(self) -> None:
        buffer = self.material.data_buffer
        run = self.members

        if run is None:
            records = len(self.data)

            offset = buffer.claim(
                self.data.array.nbytes
            )

            version = self.data.version

            moved = (
                offset != self.data_offset
                or records != self.records
            )

            if version != self.data_version or moved:
                buffer.put(
                    offset,
                    self.data.bytes,
                )

            self.data_version = version
            self.data_offset = offset
            self.data_repeats = 0
            self.records = records

            self.invalidated = (
                self.invalidated
                or moved
            )

            return

        record_size = self.material.record_size

        sizes = [
            len(drawing.data.array)
            for drawing in run
        ]

        records = (
            sum(sizes)
            + self.records_between
            * (len(run) - 1)
        )

        offset = buffer.claim(
            records * record_size
        )

        moved = (
            offset != self.data_offset
            or records != self.records
        )

        at = offset

        between = self.records_between
        last = len(run) - 1

        for index, (drawing, size) in enumerate(
            zip(run, sizes)
        ):
            data = drawing.data
            version = data.version

            repeats = (
                between
                if index != last
                else 0
            )

            if (
                version != drawing.data_version
                or at != drawing.data_offset
                or repeats != drawing.data_repeats
            ):
                buffer.put(
                    at,
                    data.bytes,
                    record_size,
                    repeats,
                )

            drawing.data_version = version
            drawing.data_offset = at
            drawing.data_repeats = repeats

            at += (
                size + between
            ) * record_size

        self.data_offset = offset
        self.records = records

        self.invalidated = (
            self.invalidated
            or moved
        )

    # -------------------------------------------------------------------------
    # Grouping
    # -------------------------------------------------------------------------

    def can_follow(
        self,
        previous: Drawing,
    ) -> bool:
        return (
            self.merges
            and previous.material is self.material
            and self.repeats_uniforms
            and self.depth_test == previous.depth_test
        )

    # -------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------

    def draw(
        self,
        render_pass: RenderPass,
    ) -> None:
        material = self.material

        render_pass.bind(
            MOBJECT_GROUP,
            material.uniform_buffer.bind_group,
            (self.uniform_offset,),
        )

        render_pass.bind(
            RESOURCE_GROUP,
            self.resource_bind_group(),
            (self.data_offset,),
        )

        self.draw_passes(
            render_pass,
        )

    def realize_textures(self) -> None:
        names = self.material.texture_names

        if not names:
            return

        gpu = self.material.gpu
        sources = self.mobject.textures

        remade = False

        for name in names:
            source = sources[name]

            texture = self.textures.get(name)

            if (
                texture is None
                or not texture.accepts(source)
            ):
                texture = source.realize(gpu)
                self.textures[name] = texture
                remade = True

            texture.refresh()

        sampler = gpu.sampler(
            self.mobject.texture_filter
        )

        if (
            remade
            or sampler is not self.sampler
        ):
            self.texture_views = [
                self.textures[name].view
                for name in names
            ]

            self.sampler = sampler
            self.own_bind_group = None

            self.invalidated = True

    def resource_bind_group(self) -> Any:
        shared = self.material.data_buffer

        if not self.texture_views:
            return shared.bind_group

        if (
            self.shared_bind_group
            is not shared.bind_group
            or self.own_bind_group is None
        ):
            self.shared_bind_group = (
                shared.bind_group
            )

            self.own_bind_group = (
                self.material.make_resource_bind_group(
                    self.texture_views,
                    self.sampler,
                )
            )

        return self.own_bind_group

    def draw_passes(
        self,
        render_pass: RenderPass,
    ) -> None:
        self.draw_pass(
            render_pass,
            "main",
            DEFAULT,
            self.material.verts_per_record
            * self.records,
        )

    def draw_pass(
        self,
        render_pass: RenderPass,
        module: str,
        state: PipelineState,
        vertices: int,
        indices: Any = None,
    ) -> None:
        pipeline = self.material.pipeline(
            module,
            state.resolved(
                self.depth_test,
            ),
        )

        render_pass.draw(
            pipeline,
            vertices,
            indices,
        )


class SurfaceDrawing(Drawing):
    """
    Drawing for ordinary 3D surfaces.
    """

    def __init__(
        self,
        material: Material,
        mobject: Mobject,
    ):
        super().__init__(
            material,
            mobject,
        )

        self.order_buffer: Any = None
        self.order_count = 0
        self.ordered = False

    def write_uniforms(self) -> bool:
        changed = super().write_uniforms()

        surface = self.mobject

        was = self.ordered
        count = self.order_count

        sort = (
            surface.sort_to_camera
            or not surface.is_opaque()
        )

        self.ordered = (
            sort
            and self.order_triangles_by_depth()
        )

        self.invalidated = (
            self.invalidated
            or self.ordered != was
            or self.order_count != count
        )

        return changed

    def order_triangles_by_depth(self) -> bool:
        first_vertices, middles = (
            self.mobject.get_triangles()
        )

        if len(first_vertices) == 0:
            return False

        camera_position = (
            self.material.gpu.frame_uniforms[
                "camera_position"
            ]
        )

        offsets = (
            middles
            - np.array(camera_position)
        )

        order = np.argsort(
            -np.einsum(
                "ij,ij->i",
                offsets,
                offsets,
            )
        )

        vertices = (
            first_vertices[
                order,
                np.newaxis,
            ]
            + np.arange(3)
        )

        self.write_order_buffer(
            vertices.astype(
                np.uint32
            ).reshape(-1)
        )

        return True

    def write_order_buffer(
        self,
        indices: np.ndarray,
    ) -> None:
        if (
            self.order_buffer is not None
            and self.order_buffer.size
            != indices.nbytes
        ):
            self.order_buffer.destroy()
            self.order_buffer = None

        if self.order_buffer is None:
            self.order_buffer = (
                self.material.gpu.device.create_buffer(
                    size=indices.nbytes,
                    usage=(
                        wgpu.BufferUsage.INDEX
                        | wgpu.BufferUsage.COPY_DST
                    ),
                )
            )

        self.material.gpu.queue.write_buffer(
            self.order_buffer,
            0,
            indices,
        )

        self.order_count = len(indices)

    def draw_passes(
        self,
        render_pass: RenderPass,
    ) -> None:
        if not self.ordered:
            super().draw_passes(
                render_pass,
            )
            return

        self.draw_pass(
            render_pass,
            "main",
            DEFAULT,
            self.order_count,
            self.order_buffer,
        )


# =============================================================================
# VMobject pipeline states
# =============================================================================

WINDING_COUNT = PipelineState(
    depth_test=False,
    depth_write=False,
    color_write=False,

    stencil_ops=(
        (
            "keep",
            "increment-wrap",
            "increment-wrap",
        ),
        (
            "keep",
            "decrement-wrap",
            "decrement-wrap",
        ),
    ),
)


# Border of the fill.
#
# It participates in the color pass, but NEVER writes depth.
FILL_BORDER = PipelineState(
    depth_write=False,
    depth_compare="less",

    stencil_compare="equal",

    stencil_ops=(
        KEEP,
        KEEP,
    ),
)


# Final fill cover.
#
# Color pass:
#     depth test = less
#     depth write = false
#
# Depth pass:
#     same shader + separate state below
WINDING_COVER = PipelineState(
    depth_write=False,
    depth_compare="less",
    stencil_compare="not-equal",
    stencil_ops=(
        ("keep", "zero", "zero"),
        ("keep", "zero", "zero"),
    ),
)

VMOBJECT_FILL_DEPTH = PipelineState(
    depth_write=True,
    depth_compare="less",
    color_write=False,
    stencil_compare="not-equal",
    stencil_ops=(
        ("keep", "zero", "zero"),
        ("keep", "zero", "zero"),
    ),
)


# Writes the depth of the stroke only.
VMOBJECT_STROKE_DEPTH = PipelineState(
    depth_write=True,
    depth_compare="less",

    color_write=False,
)


# VMobject color passes.
#
# IMPORTANT:
# depth_write is OFF.
#
# Thus fill/border/stroke all compare against the depth that existed before
# this VMobject started drawing. They cannot modify that depth and therefore
# cannot fight with one another.
VMOBJECT_COLOR = PipelineState(
    depth_write=False,
    depth_compare="less",
    color_write=True,
)


class VDrawing(Drawing):
    merges = True
    records_between = 1

    fill_file = "fill.wgsl"
    stroke_file = "stroke.wgsl"

    fill_depth_file = "fill_depth.wgsl"
    stroke_depth_file = "stroke_depth.wgsl"

    fill_verts_per_curve = 6
    stroke_verts_per_curve = 6 * (32 - 1)

    border_declaration = (
        "const IS_FILL_BORDER: bool = false;"
    )

    @classmethod
    def draws(
        cls,
        mobject: Mobject,
    ) -> bool:
        return True

    @classmethod
    def key(
        cls,
        mobject: Mobject,
    ) -> tuple:
        return (
            *super().key(mobject),
            mobject.shader_code_target,
        )

    @classmethod
    def module_specs(
        cls,
        mobject: Mobject,
    ) -> list[ModuleSpec]:

        stroke_source = read_shader_file(
            cls.stroke_file
        )

        declaration = cls.border_declaration

        if declaration not in stroke_source:
            raise ValueError(
                f"The stroke shader no longer declares "
                f"{declaration!r}"
            )

        border = {
            declaration:
                declaration.replace(
                    "false",
                    "true",
                )
        }

        target = mobject.shader_code_target
        replacements = (
            mobject.shader_code_replacements
        )

        nothing: dict[str, str] = dict()

        for_fill = (
            replacements
            if target in (None, "fill")
            else nothing
        )

        for_stroke = (
            replacements
            if target in (None, "stroke")
            else nothing
        )

        return [
            (
                "fill",
                cls.fill_file,
                for_fill,
            ),

            (
                "stroke",
                cls.stroke_file,
                for_stroke,
            ),

            (
                "border",
                cls.stroke_file,
                {
                    **for_stroke,
                    **border,
                },
            ),

            (
                "fill_depth",
                cls.fill_depth_file,
                for_fill,
            ),

            (
                "stroke_depth",
                cls.stroke_depth_file,
                for_stroke,
            ),

            (
                "border_depth",
                cls.stroke_depth_file,
                {
                    **for_stroke,
                    **border,
                },
            ),
        ]

    def __init__(
        self,
        material: Material,
        mobject: Mobject,
    ):
        super().__init__(
            material,
            mobject,
        )

        self.has_fill = False
        self.stroke_behind = False
        self.fill_group: Any = None

    # -------------------------------------------------------------------------
    # Uniforms
    # -------------------------------------------------------------------------

    def write_uniforms(self) -> bool:
        stroke_behind = (
            self.mobject.stroke_behind
        )

        fill_group = (
            self.mobject.fill_group
        )

        changed = super().write_uniforms()

        self.invalidated = (
            self.invalidated
            or stroke_behind != self.stroke_behind
            or fill_group is not self.fill_group
        )

        self.stroke_behind = stroke_behind
        self.fill_group = fill_group

        if changed:
            has_fill = bool(
                self.uniforms["fill_rgba"][3]
                or self.uniforms["fill_rgba_end"][3]
            )

            self.invalidated = (
                self.invalidated
                or has_fill != self.has_fill
            )

            self.has_fill = has_fill

        return changed

    # -------------------------------------------------------------------------
    # Grouping
    # -------------------------------------------------------------------------

    def can_follow(
        self,
        previous: Drawing,
    ) -> bool:
        if not (
            super().can_follow(previous)
            and self.stroke_behind
            == previous.stroke_behind
        ):
            return False

        if not (
            self.has_fill
            or previous.has_fill
        ):
            return True

        return (
            self.fill_group is not None
            and self.fill_group
            is previous.fill_group
        )

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    def get_num_curves(self) -> int:
        return self.records // 2

    def fill_vertices(self) -> int:
        return (
            self.fill_verts_per_curve
            * self.get_num_curves()
        )

    def stroke_vertices(
        self,
        extra_curves: int = 0,
    ) -> int:
        return (
            self.stroke_verts_per_curve
            * (
                self.get_num_curves()
                + extra_curves
            )
        )

    # -------------------------------------------------------------------------
    # COLOR PASSES
    # -------------------------------------------------------------------------

    def draw_fill(
        self,
        render_pass: RenderPass,
    ) -> None:
        if not self.has_fill:
            return

        vertices = self.fill_vertices()

        # Count winding.
        #
        # This never touches depth.
        self.draw_pass(
            render_pass,
            "fill",
            WINDING_COUNT,
            vertices,
        )

        # Draw the border.
        #
        # This also never touches depth.
        self.draw_fill_border(
            render_pass,
        )

        # Draw the actual fill.
        #
        # Depth is tested but NOT written.
        self.draw_pass(
            render_pass,
            "fill",
            WINDING_COVER,
            vertices,
        )

    def draw_fill_border(
        self,
        render_pass: RenderPass,
    ) -> None:
        self.draw_pass(
            render_pass,
            "border",
            FILL_BORDER,
            self.stroke_vertices(
                extra_curves=1,
            ),
        )

    def draw_stroke(
        self,
        render_pass: RenderPass,
    ) -> None:
        # Color only.
        #
        # Crucially depth_write=False.
        self.draw_pass(
            render_pass,
            "stroke",
            VMOBJECT_COLOR,
            self.stroke_vertices(),
        )

    # -------------------------------------------------------------------------
    # DEPTH PASSES
    # -------------------------------------------------------------------------
    def draw_fill_depth(
        self,
        render_pass: RenderPass,
    ) -> None:
        if not self.has_fill:
            return

        vertices = self.fill_vertices()

        self.draw_pass(
            render_pass,
            "fill_depth",
            WINDING_COUNT,
            vertices,
        )

        self.draw_pass(
            render_pass,
            "fill_depth",
            VMOBJECT_FILL_DEPTH,
            vertices,
        )

    def draw_stroke_depth(
        self,
        render_pass: RenderPass,
    ) -> None:
        # The stroke depth shader uses the same generated stroke geometry
        # but discards the anti-alias fringe before depth is written.
        self.draw_pass(
            render_pass,
            "stroke_depth",
            VMOBJECT_STROKE_DEPTH,
            self.stroke_vertices(),
        )

    def draw_depth(
        self,
        render_pass: RenderPass,
    ) -> None:
        """
        Commit this entire VMobject to the depth buffer AFTER all of its
        visible color has been rendered.

        Consequently:

            VMobject A:
                fill ──┐
                border │
                stroke ┘
                       ↓
                   one depth commit

        VMobject B, rendered afterwards, sees A as one depth-coherent
        object.
        """

        if not self.depth_test:
            return

        if self.has_fill:
            self.draw_fill_depth(
                render_pass,
            )

        self.draw_stroke_depth(
            render_pass,
        )

    # -------------------------------------------------------------------------
    # Complete VMobject
    # -------------------------------------------------------------------------

    def draw_passes(
        self,
        render_pass: RenderPass,
    ) -> None:
        # -------------------------------------------------------------
        # PHASE 1:
        #
        # Render ALL color belonging to this VMobject.
        #
        # None of these passes writes depth.
        # -------------------------------------------------------------

        if self.stroke_behind:
            self.draw_stroke(
                render_pass,
            )
            self.draw_fill(
                render_pass,
            )
        else:
            self.draw_fill(
                render_pass,
            )
            self.draw_stroke(
                render_pass,
            )

        # -------------------------------------------------------------
        # PHASE 2:
        #
        # Commit the visible VMobject to the depth buffer.
        #
        # This happens AFTER the entire VMobject has been rendered.
        # -------------------------------------------------------------

        self.draw_depth(
            render_pass,
        )
