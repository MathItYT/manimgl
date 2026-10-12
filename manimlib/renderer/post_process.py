"""
manimlib/renderer/post_process.py
=================================
Pipeline optimizado de post-procesamiento WebGPU con Scissoring automático y Scratch Pooling.
"""

from __future__ import annotations
import wgpu
import numpy as np
from typing import TYPE_CHECKING, Optional, Any
from manimlib.constants import FRAME_WIDTH, FRAME_HEIGHT

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

        # Capas de post-procesamiento ping-pong
        self.layer_texture: Optional[Any] = None
        self.layer_view: Optional[Any] = None
        self.layer_texture_b: Optional[Any] = None
        self.layer_view_b: Optional[Any] = None

        # Scratch Pool compartido para efectos de 2 pases (Glow, Blur, Shadow)
        self.scratch_texture: Optional[Any] = None
        self.scratch_view: Optional[Any] = None

        self.mask_texture: Optional[Any] = None
        self.mask_view: Optional[Any] = None
        self.bg_texture: Optional[Any] = None
        self.bg_view: Optional[Any] = None

        self.scene_texture: Optional[Any] = None
        self.scene_view: Optional[Any] = None
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
        self._bind_group_cache: dict[Any, Any] = {}

    def get_single_tex_bind_group(self, view: Any) -> Any:
        bg = self._bind_group_cache.get(view)
        if bg is None:
            bg = self.device.create_bind_group(
                layout=self.single_tex_layout,
                entries=[
                    {"binding": 0, "resource": view},
                    {"binding": 1, "resource": self.sampler},
                ],
            )
            self._bind_group_cache[view] = bg
        return bg

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
        self._bind_group_cache.clear()

        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING

        self.layer_texture = self.device.create_texture(size=(width, height, 1), format=COLOR_FORMAT, usage=usage)
        self.layer_view = self.layer_texture.create_view()

        self.layer_texture_b = self.device.create_texture(size=(width, height, 1), format=COLOR_FORMAT, usage=usage)
        self.layer_view_b = self.layer_texture_b.create_view()

        # Textura scratch compartida (a media resolución para acelerar pases intermedios)
        sw, sh = max(1, width // 2), max(1, height // 2)
        self.scratch_texture = self.device.create_texture(size=(sw, sh, 1), format=COLOR_FORMAT, usage=usage)
        self.scratch_view = self.scratch_texture.create_view()

        self.mask_texture = self.device.create_texture(size=(width, height, 1), format=COLOR_FORMAT, usage=usage)
        self.mask_view = self.mask_texture.create_view()

        self.bg_texture = self.device.create_texture(
            size=(width, height, 1), format=COLOR_FORMAT,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.bg_view = self.bg_texture.create_view()

        if self.samples > 1:
            self.scene_texture = self.device.create_texture(size=(width, height, 1), format=COLOR_FORMAT, usage=usage | wgpu.TextureUsage.COPY_SRC)
            self.scene_view = self.scene_texture.create_view()
            self.msaa_layer_texture = self.device.create_texture(size=(width, height, 1), format=COLOR_FORMAT, sample_count=self.samples, usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
            self.msaa_layer_view = self.msaa_layer_texture.create_view()
            self.msaa_mask_texture = self.device.create_texture(size=(width, height, 1), format=COLOR_FORMAT, sample_count=self.samples, usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
            self.msaa_mask_view = self.msaa_mask_texture.create_view()
            self.layer_depth_texture = self.device.create_texture(size=(width, height, 1), format=DEPTH_STENCIL_FORMAT, sample_count=self.samples, usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
            self.layer_depth_view = self.layer_depth_texture.create_view()
        else:
            self.scene_texture = None
            self.scene_view = None
            self.msaa_layer_texture = None
            self.msaa_layer_view = None
            self.msaa_mask_texture = None
            self.msaa_mask_view = None
            self.layer_depth_texture = self.device.create_texture(size=(width, height, 1), format=DEPTH_STENCIL_FORMAT, usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
            self.layer_depth_view = self.layer_depth_texture.create_view()

    def clamp_scissor(self, min_x: int, min_y: int, max_x: int, max_y: int, target_w: int, target_h: int) -> tuple[int, int, int, int]:
        """Asegura que el scissor rect cumpla estrictamente las reglas de WebGPU:
        0 <= x < target_w, 0 <= y < target_h, width > 0, height > 0, x + w <= target_w.
        """
        cx = max(0, min(int(min_x), target_w - 1))
        cy = max(0, min(int(min_y), target_h - 1))
        right = max(cx + 1, min(int(max_x), target_w))
        bottom = max(cy + 1, min(int(max_y), target_h))
        return cx, cy, right - cx, bottom - cy

    def get_scratch_scissor(self, scissor: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Escala de forma segura el scissor rect a la resolución del buffer scratch."""
        sx, sy, sw, sh = scissor
        sw_tex = max(1, self.width // 2)
        sh_tex = max(1, self.height // 2)
        rx = sw_tex / max(self.width, 1)
        ry = sh_tex / max(self.height, 1)

        min_x = int(np.floor(sx * rx))
        min_y = int(np.floor(sy * ry))
        max_x = int(np.ceil((sx + sw) * rx))
        max_y = int(np.ceil((sy + sh) * ry))
        return self.clamp_scissor(min_x, min_y, max_x, max_y, sw_tex, sh_tex)

    def get_scissor_rect(self, mobject: Mobject, padding: int = 40) -> tuple[int, int, int, int]:
        """Proyección 3D real idéntica a project_point.wgsl."""
        if self.camera is None or not hasattr(self.camera, "frame"):
            return 0, 0, self.width, self.height

        frame = self.camera.frame

        # 1. Recolectar todos los puntos del mobject y sus submobjects
        pts_list = []
        for sm in mobject.get_family():
            try:
                p = sm.get_points()
                if p is not None and len(p) > 0:
                    pts_list.append(p[:, :3])
            except Exception:
                continue

        if not pts_list:
            return 0, 0, self.width, self.height

        points = np.vstack(pts_list)

        # 2. Agregar las 8 esquinas del bounding box 3D para cubrir grosor y curvatura
        min_xyz = np.min(points, axis=0)
        max_xyz = np.max(points, axis=0)
        bbox_corners = np.array([
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
        ], dtype=np.float64)
        all_points = np.vstack([points, bbox_corners])

        # 3. Determinar si el objeto está fijado al marco (HUD 2D)
        is_fixed = False
        if hasattr(mobject, "uniforms") and "is_fixed_in_frame" in mobject.uniforms.dtype.names:
            is_fixed = bool(mobject.uniforms["is_fixed_in_frame"])
        elif getattr(mobject, "is_fixed_in_frame", False):
            is_fixed = True

        scale = float(frame.get_scale())
        focal_dist = max(float(frame.get_focal_distance()), 1e-6)

        # 4. Transformación de cámara y perspectiva
        if is_fixed:
            cam_points = all_points
            scaled_x = cam_points[:, 0] * (2.0 / FRAME_WIDTH)
            scaled_y = cam_points[:, 1] * (2.0 / FRAME_HEIGHT)
            scaled_z = cam_points[:, 2] / focal_dist
        else:
            view = np.asarray(frame.get_view_matrix(), dtype=np.float64)
            p4 = np.hstack([all_points, np.ones((len(all_points), 1), dtype=np.float64)])
            cam_points = (p4 @ view.T)[:, :3]

            scaled_x = cam_points[:, 0] * (2.0 / FRAME_WIDTH)
            scaled_y = cam_points[:, 1] * (2.0 / FRAME_HEIGHT)
            scaled_z = cam_points[:, 2] * (scale / focal_dist)

        # Divisor de perspectiva según project_point.wgsl: w = 1.0 - scaled.z
        w_clip = 1.0 - scaled_z

        # Si el objeto atraviesa el plano de la cámara o queda detrás, abarcar pantalla completa
        if np.any(w_clip <= 0.05) or not np.all(np.isfinite(w_clip)):
            return 0, 0, self.width, self.height

        # 5. Coordenadas NDC y proyección final a píxeles
        x_ndc = scaled_x / w_clip
        y_ndc = scaled_y / w_clip

        if not np.all(np.isfinite(x_ndc)) or not np.all(np.isfinite(y_ndc)):
            return 0, 0, self.width, self.height

        px = (0.5 + 0.5 * x_ndc) * self.width
        py = (0.5 - 0.5 * y_ndc) * self.height

        min_px = int(np.floor(np.min(px))) - padding
        max_px = int(np.ceil(np.max(px))) + padding
        min_py = int(np.floor(np.min(py))) - padding
        max_py = int(np.ceil(np.max(py))) + padding

        return self.clamp_scissor(min_px, min_py, max_px, max_py, self.width, self.height)

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

    def get_layer_attachments(self, depth_view: Any = None, clear_color: bool = True, depth_load_op: wgpu.LoadOp = wgpu.LoadOp.clear) -> dict:
        color_load = wgpu.LoadOp.clear if clear_color else wgpu.LoadOp.load
        actual_depth = depth_view if depth_view is not None else self.layer_depth_view
        view = self.msaa_layer_view if self.samples > 1 else self.layer_view
        resolve = self.layer_view if self.samples > 1 else None

        att = {
            "color_attachments": [{
                "view": view,
                "resolve_target": resolve,
                "load_op": color_load,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0.0, 0.0, 0.0, 0.0),
            }],
            "depth_stencil_attachment": {
                "view": actual_depth,
                "depth_clear_value": 1.0,
                "depth_load_op": depth_load_op,
                "depth_store_op": wgpu.StoreOp.store,
                "stencil_clear_value": 0,
                "stencil_load_op": depth_load_op,
                "stencil_store_op": wgpu.StoreOp.store,
            },
        }
        return att

    def get_scratch_attachments(self, depth_view: Any = None, clear_color: bool = True, depth_load_op: wgpu.LoadOp = wgpu.LoadOp.load) -> dict:
        color_load = wgpu.LoadOp.clear if clear_color else wgpu.LoadOp.load
        actual_depth = depth_view if depth_view is not None else self.layer_depth_view
        view = self.msaa_layer_view if self.samples > 1 else self.layer_view_b
        resolve = self.layer_view_b if self.samples > 1 else None

        return {
            "color_attachments": [{
                "view": view,
                "resolve_target": resolve,
                "load_op": color_load,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0.0, 0.0, 0.0, 0.0),
            }],
            "depth_stencil_attachment": {
                "view": actual_depth,
                "depth_clear_value": 1.0,
                "depth_load_op": depth_load_op,
                "depth_store_op": wgpu.StoreOp.store,
                "stencil_clear_value": 0,
                "stencil_load_op": depth_load_op,
                "stencil_store_op": wgpu.StoreOp.store,
            },
        }

    def create_fullscreen_pipeline(self, fragment_wgsl: str, bind_group_layouts: list, blend: Optional[dict] = None) -> Any:
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

    def blit(self, encoder: Any, src_view: Any, dst_view: Any, scissor: tuple[int, int, int, int] | None = None) -> None:
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

        bg = self.get_single_tex_bind_group(src_view)
        pass_rp = encoder.begin_render_pass(color_attachments=[{
            "view": dst_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        if scissor is not None:
            pass_rp.set_scissor_rect(*scissor)
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
            self._blit_fullscreen_pipeline = self.create_fullscreen_pipeline(BLIT_FULLSCREEN_WGSL, [self.single_tex_layout], blend=None)

        bg = self.get_single_tex_bind_group(src_view)
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
        active_effects = [e for e in effects if getattr(e, "is_active", lambda p: True)(self)]

        # Margen dinámico que toma en cuenta antialiasing, grosor de trazo, radio y sombras
        pad = 16
        for e in active_effects:
            r = float(getattr(e, "radius", 0.0))
            if r > 0:
                pad = max(pad, int(r * 2.2))
            offset = getattr(e, "offset", None)
            if offset is not None:
                pad = max(pad, int(np.hypot(offset[0], offset[1]) + r * 2.2))

        scissor = self.get_scissor_rect(mobject, padding=pad)

        if not active_effects:
            self.blit(encoder, self.layer_view, target_view, scissor=scissor)
            return

        ordered_effects = sorted(active_effects, key=lambda e: getattr(e, "order", 50))

        if len(ordered_effects) == 1:
            ordered_effects[0].apply(self, encoder, self.layer_view, target_view, scissor=scissor, load_op=wgpu.LoadOp.load)
            return

        curr_in = self.layer_view
        curr_out = self.layer_view_b
        num_effects = len(ordered_effects)

        for i, effect in enumerate(ordered_effects):
            is_last = (i == num_effects - 1)
            dest_view = target_view if is_last else curr_out
            load_op = wgpu.LoadOp.load if is_last else wgpu.LoadOp.clear

            effect.apply(self, encoder, curr_in, dest_view, scissor=scissor, load_op=load_op)

            if not is_last:
                curr_in, curr_out = curr_out, curr_in