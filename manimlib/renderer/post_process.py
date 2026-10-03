"""
manimlib/renderer/post_process.py
=================================
Abstract, effect-agnostic WebGPU post-processing pipeline for Manim.
Manages isolated mobject layer render targets, depth-stencil attachments,
dedicated mask textures, background snapshots, and fullscreen shader dispatching.
"""

from __future__ import annotations

import wgpu
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from manimlib.renderer.gpu import Gpu
    from manimlib.mobject.mobject import Mobject

COLOR_FORMAT = wgpu.TextureFormat.rgba8unorm
DEPTH_STENCIL_FORMAT = wgpu.TextureFormat.depth24plus_stencil8

FULLSCREEN_VERTEX_WGSL = """
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var corners = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0)
    );
    let corner = corners[index];
    var out: VertexOutput;
    out.position = vec4f(corner, 0.0, 1.0);
    out.uv = vec2f(0.5 + 0.5 * corner.x, 0.5 - 0.5 * corner.y);
    return out;
}
"""


class MobjectPostProcessor:
    def __init__(self, gpu: Gpu):
        self.gpu = gpu
        self.device = gpu.device
        self.queue = gpu.queue
        self.renderer: Any = None
        self.camera: Any = None
        self.width = 0
        self.height = 0
        self.samples = getattr(gpu, "samples", 1)
        self.time = 0.0

        # Texturas aisladas resueltas (1 muestra) para fragment shaders
        self.layer_texture: Optional[Any] = None
        self.layer_view: Optional[Any] = None
        self.layer_texture_b: Optional[Any] = None
        self.layer_view_b: Optional[Any] = None

        # Buffers de máscara (1 muestra)
        self.mask_texture: Optional[Any] = None
        self.mask_view: Optional[Any] = None

        # Snapshot del fondo (1 muestra) para refracción
        self.bg_texture: Optional[Any] = None
        self.bg_view: Optional[Any] = None

        # Buffer de acumulación de escena cuando samples > 1
        self.scene_texture: Optional[Any] = None
        self.scene_view: Optional[Any] = None

        # Texturas multisampleadas (samples > 1) para renderizado de mobjects
        self.msaa_layer_texture: Optional[Any] = None
        self.msaa_layer_view: Optional[Any] = None
        self.msaa_mask_texture: Optional[Any] = None
        self.msaa_mask_view: Optional[Any] = None

        self.layer_depth_texture: Optional[Any] = None
        self.layer_depth_view: Optional[Any] = None

        self.sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
        )

        self.single_tex_layout = self.device.create_bind_group_layout(entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": wgpu.TextureSampleType.float},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.filtering},
            },
        ])

    def resize(self, width: int, height: int) -> None:
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return

        current_samples = int(getattr(self.gpu, "samples", 1))
        if (self.width, self.height, self.samples) == (width, height, current_samples):
            return

        self.width = width
        self.height = height
        self.samples = current_samples
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING

        # Capas de post-procesamiento de 1 muestra
        self.layer_texture = self.device.create_texture(
            size=(width, height, 1),
            format=COLOR_FORMAT,
            usage=usage,
        )
        self.layer_view = self.layer_texture.create_view()

        self.layer_texture_b = self.device.create_texture(
            size=(width, height, 1),
            format=COLOR_FORMAT,
            usage=usage,
        )
        self.layer_view_b = self.layer_texture_b.create_view()

        self.mask_texture = self.device.create_texture(
            size=(width, height, 1),
            format=COLOR_FORMAT,
            usage=usage,
        )
        self.mask_view = self.mask_texture.create_view()

        # Background snapshot: resolución simple para muestreo seguro
        self.bg_texture = self.device.create_texture(
            size=(width, height, 1),
            format=COLOR_FORMAT,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.bg_view = self.bg_texture.create_view()

        if self.samples > 1:
            self.scene_texture = self.device.create_texture(
                size=(width, height, 1),
                format=COLOR_FORMAT,
                usage=usage | wgpu.TextureUsage.COPY_SRC,
            )
            self.scene_view = self.scene_texture.create_view()

            self.msaa_layer_texture = self.device.create_texture(
                size=(width, height, 1),
                format=COLOR_FORMAT,
                sample_count=self.samples,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            self.msaa_layer_view = self.msaa_layer_texture.create_view()

            self.msaa_mask_texture = self.device.create_texture(
                size=(width, height, 1),
                format=COLOR_FORMAT,
                sample_count=self.samples,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            self.msaa_mask_view = self.msaa_mask_texture.create_view()

            self.layer_depth_texture = self.device.create_texture(
                size=(width, height, 1),
                format=DEPTH_STENCIL_FORMAT,
                sample_count=self.samples,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            self.layer_depth_view = self.layer_depth_texture.create_view()
        else:
            self.scene_texture = None
            self.scene_view = None
            self.msaa_layer_texture = None
            self.msaa_layer_view = None
            self.msaa_mask_texture = None
            self.msaa_mask_view = None

            self.layer_depth_texture = self.device.create_texture(
                size=(width, height, 1),
                format=DEPTH_STENCIL_FORMAT,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            self.layer_depth_view = self.layer_depth_texture.create_view()

    def copy_target_to_bg(self, encoder: Any, target_view: Any) -> None:
        color_tex = getattr(target_view, "texture", getattr(target_view, "_texture", None))
        if color_tex is not None and self.bg_texture is not None:
            if getattr(color_tex, "sample_count", 1) > 1:
                return
            encoder.copy_texture_to_texture(
                {"texture": color_tex, "mip_level": 0, "origin": (0, 0, 0)},
                {"texture": self.bg_texture, "mip_level": 0, "origin": (0, 0, 0)},
                (self.width, self.height, 1),
            )

    def get_layer_attachments(
        self,
        depth_view: Any = None,
        clear_color: bool = True,
        depth_load_op: wgpu.LoadOp = wgpu.LoadOp.clear,
    ) -> dict:
        color_load = wgpu.LoadOp.clear if clear_color else wgpu.LoadOp.load
        actual_depth_view = depth_view if depth_view is not None else self.layer_depth_view

        if self.samples > 1:
            return {
                "color_attachments": [{
                    "view": self.msaa_layer_view,
                    "resolve_target": self.layer_view,
                    "load_op": color_load,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                }],
                "depth_stencil_attachment": {
                    "view": actual_depth_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": depth_load_op,
                    "depth_store_op": wgpu.StoreOp.store,
                    "stencil_clear_value": 0,
                    "stencil_load_op": depth_load_op,
                    "stencil_store_op": wgpu.StoreOp.store,
                },
            }
        else:
            return {
                "color_attachments": [{
                    "view": self.layer_view,
                    "load_op": color_load,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                }],
                "depth_stencil_attachment": {
                    "view": actual_depth_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": depth_load_op,
                    "depth_store_op": wgpu.StoreOp.store,
                    "stencil_clear_value": 0,
                    "stencil_load_op": depth_load_op,
                    "stencil_store_op": wgpu.StoreOp.store,
                },
            }

    def get_scratch_attachments(
        self,
        depth_view: Any = None,
        clear_color: bool = True,
        depth_load_op: wgpu.LoadOp = wgpu.LoadOp.load,
    ) -> dict:
        color_load = wgpu.LoadOp.clear if clear_color else wgpu.LoadOp.load
        actual_depth_view = depth_view if depth_view is not None else self.layer_depth_view

        if self.samples > 1:
            return {
                "color_attachments": [{
                    "view": self.msaa_layer_view,
                    "resolve_target": self.layer_view_b,
                    "load_op": color_load,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                }],
                "depth_stencil_attachment": {
                    "view": actual_depth_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": depth_load_op,
                    "depth_store_op": wgpu.StoreOp.store,
                    "stencil_clear_value": 0,
                    "stencil_load_op": depth_load_op,
                    "stencil_store_op": wgpu.StoreOp.store,
                },
            }
        else:
            return {
                "color_attachments": [{
                    "view": self.layer_view_b,
                    "load_op": color_load,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                }],
                "depth_stencil_attachment": {
                    "view": actual_depth_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": depth_load_op,
                    "depth_store_op": wgpu.StoreOp.store,
                    "stencil_clear_value": 0,
                    "stencil_load_op": depth_load_op,
                    "stencil_store_op": wgpu.StoreOp.store,
                },
            }

    def create_fullscreen_pipeline(
        self,
        fragment_wgsl: str,
        bind_group_layouts: list,
        blend: Optional[dict] = None,
    ) -> Any:
        vert_mod = self.gpu.module(FULLSCREEN_VERTEX_WGSL)
        frag_mod = self.gpu.module(fragment_wgsl)
        targets = [{"format": COLOR_FORMAT}]
        if blend is not None:
            targets[0]["blend"] = blend

        return self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=bind_group_layouts),
            vertex={"module": vert_mod, "entry_point": "vs_main"},
            fragment={"module": frag_mod, "entry_point": "fs_main", "targets": targets},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

    def blit(self, encoder: Any, src_view: Any, dst_view: Any) -> None:
        if not hasattr(self, "_blit_pipeline"):
            BLIT_WGSL = """
            @group(0) @binding(0) var in_tex: texture_2d<f32>;
            @group(0) @binding(1) var in_smp: sampler;
            struct VertexOutput { @builtin(position) position: vec4f, @location(0) uv: vec2f };
            @fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
                let col = textureSample(in_tex, in_smp, in.uv);
                if (col.a <= 1e-4) { discard; }
                return col;
            }
            """
            PREMULT_BLEND = {
                "color": {"src_factor": wgpu.BlendFactor.one, "dst_factor": wgpu.BlendFactor.one_minus_src_alpha, "operation": wgpu.BlendOperation.add},
                "alpha": {"src_factor": wgpu.BlendFactor.one, "dst_factor": wgpu.BlendFactor.one_minus_src_alpha, "operation": wgpu.BlendOperation.add},
            }
            self._blit_pipeline = self.create_fullscreen_pipeline(BLIT_WGSL, [self.single_tex_layout], blend=PREMULT_BLEND)

        bg = self.device.create_bind_group(
            layout=self.single_tex_layout,
            entries=[{"binding": 0, "resource": src_view}, {"binding": 1, "resource": self.sampler}],
        )
        pass_rp = encoder.begin_render_pass(color_attachments=[{
            "view": dst_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        pass_rp.set_pipeline(self._blit_pipeline)
        pass_rp.set_bind_group(0, bg)
        pass_rp.draw(3)
        pass_rp.end()

    def blit_fullscreen(self, encoder: Any, src_view: Any, dst_view: Any) -> None:
        if not hasattr(self, "_blit_fullscreen_pipeline"):
            BLIT_FULLSCREEN_WGSL = """
            @group(0) @binding(0) var in_tex: texture_2d<f32>;
            @group(0) @binding(1) var in_smp: sampler;
            struct VertexOutput { @builtin(position) position: vec4f, @location(0) uv: vec2f };
            @fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
                return textureSample(in_tex, in_smp, in.uv);
            }
            """
            self._blit_fullscreen_pipeline = self.create_fullscreen_pipeline(
                BLIT_FULLSCREEN_WGSL, [self.single_tex_layout], blend=None
            )

        bg = self.device.create_bind_group(
            layout=self.single_tex_layout,
            entries=[{"binding": 0, "resource": src_view}, {"binding": 1, "resource": self.sampler}],
        )
        pass_rp = encoder.begin_render_pass(color_attachments=[{
            "view": dst_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0.0, 0.0, 0.0, 1.0),
        }])
        pass_rp.set_pipeline(self._blit_fullscreen_pipeline)
        pass_rp.set_bind_group(0, bg)
        pass_rp.draw(3)
        pass_rp.end()

    def process_mobject(self, encoder: Any, target_view: Any, mobject: Mobject) -> None:
        effects = getattr(mobject, "get_vfx_list", lambda: [])()
        if not effects:
            self.blit(encoder, self.layer_view, target_view)
            return

        active_effects = [e for e in effects if getattr(e, "is_active", lambda p: True)(self)]
        if not active_effects:
            self.blit(encoder, self.layer_view, target_view)
            return

        # Cadena de renderizado: Máscara -> Distorsión espacial -> LiquidGlass -> Glow
        mask_effects = [e for e in active_effects if getattr(e, "name", "") == "Mask"]
        glass_effects = [e for e in active_effects if getattr(e, "name", "") == "LiquidGlass"]
        glow_effects = [e for e in active_effects if getattr(e, "name", "") == "Glow"]
        other_effects = [e for e in active_effects if getattr(e, "name", "") not in ("Mask", "LiquidGlass", "Glow")]
        ordered_effects = mask_effects + other_effects + glass_effects + glow_effects

        if len(ordered_effects) == 1:
            ordered_effects[0].apply(self, encoder, self.layer_view, target_view)
            return

        curr_in = self.layer_view
        curr_out = self.layer_view_b
        num_effects = len(ordered_effects)

        for i, effect in enumerate(ordered_effects):
            is_last = (i == num_effects - 1)
            dest_view = target_view if is_last else curr_out

            if not is_last:
                clear_pass = encoder.begin_render_pass(color_attachments=[{
                    "view": dest_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                }])
                clear_pass.end()

            effect.apply(self, encoder, curr_in, dest_view)

            if not is_last:
                if curr_in is self.layer_view:
                    curr_in = self.layer_view_b
                    curr_out = self.layer_view
                else:
                    curr_in = self.layer_view
                    curr_out = self.layer_view_b