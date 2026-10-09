"""
manimlib/utils/vfx_presets.py
=============================
Independent, self-contained Visual Effects presets and custom WGSL post-processing for Manim.
The LiquidGlass preset ports OverShifted/LiquidGlass faithfully; only its surface geometry
is promoted to Manim's real 3D projection path.
"""

from __future__ import annotations

import os
import re
import numpy as np
import wgpu
from typing import Sequence, Callable, Any

from manimlib.mobject.mobject import Mobject
from manimlib.animation.animation import Animation


PREMULTIPLIED_ADDITIVE_BLEND = {
    "color": {
        "src_factor": wgpu.BlendFactor.one,
        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
        "operation": wgpu.BlendOperation.add,
    },
    "alpha": {
        "src_factor": wgpu.BlendFactor.one,
        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
        "operation": wgpu.BlendOperation.add,
    },
}

STANDARD_ALPHA_BLEND = {
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


class PostProcessEffect:
    is_group_effect: bool = False

    def __init__(self, name: str = "PostProcessEffect", allow_multiple: bool = False):
        self.name = name
        self.allow_multiple = allow_multiple
        self.mobject: Mobject | None = None

    def attach(self, mobject: Mobject) -> None:
        self.mobject = mobject

    def is_active(self, processor: Any) -> bool:
        return True

    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        raise NotImplementedError

    def copy(self) -> PostProcessEffect:
        import copy
        return copy.deepcopy(self)


# =====================================================================
# Preset: Fast Separable Gaussian Blur
# =====================================================================

GAUSSIAN_BLUR_PASS1_WGSL = """
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct BlurParams {
    radius: f32,
    _pad0: f32,
    resolution: vec2f,
};
@group(1) @binding(0) var<uniform> params: BlurParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

const SAMPLES: i32 = 16;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = max(params.radius, 0.5);
    let step_px = max(r / f32(SAMPLES), 1.0);
    let actual_radius = step_px * f32(SAMPLES);
    let sigma = max(actual_radius * 0.38, 1.0);
    let two_sigma_sq = 2.0 * sigma * sigma;

    let inv_res_x = 1.0 / params.resolution.x;
    var accum = vec4f(0.0);
    var total_weight = 0.0;

    for (var i = -SAMPLES; i <= SAMPLES; i = i + 1) {
        let offset_px = f32(i) * step_px;
        let weight = exp(-(offset_px * offset_px) / two_sigma_sq);
        let sample_uv = in.uv + vec2f(offset_px * inv_res_x, 0.0);

        let col = textureSample(in_tex, in_smp, sample_uv);
        accum += col * weight;
        total_weight += weight;
    }

    return accum / max(total_weight, 1e-4);
}
"""

GAUSSIAN_BLUR_PASS2_WGSL = """
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct BlurParams {
    radius: f32,
    _pad0: f32,
    resolution: vec2f,
};
@group(1) @binding(0) var<uniform> params: BlurParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

const SAMPLES: i32 = 16;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = max(params.radius, 0.5);
    let step_px = max(r / f32(SAMPLES), 1.0);
    let actual_radius = step_px * f32(SAMPLES);
    let sigma = max(actual_radius * 0.38, 1.0);
    let two_sigma_sq = 2.0 * sigma * sigma;

    let inv_res_y = 1.0 / params.resolution.y;
    var accum = vec4f(0.0);
    var total_weight = 0.0;

    for (var i = -SAMPLES; i <= SAMPLES; i = i + 1) {
        let offset_px = f32(i) * step_px;
        let weight = exp(-(offset_px * offset_px) / two_sigma_sq);
        let sample_uv = in.uv + vec2f(0.0, offset_px * inv_res_y);

        let col = textureSample(in_tex, in_smp, sample_uv);
        accum += col * weight;
        total_weight += weight;
    }

    let final_col = accum / max(total_weight, 1e-4);
    if (final_col.a <= 1e-5 && dot(final_col.rgb, vec3f(1.0)) <= 1e-5) {
        discard;
    }
    return final_col;
}
"""


class GaussianBlur(PostProcessEffect):
    is_group_effect = True

    def __init__(self, radius: float = 12.0, downscale: float = 1.0):
        super().__init__(name="GaussianBlur", allow_multiple=False)
        self.radius = float(radius)
        self.downscale = min(max(float(downscale), 0.1), 1.0)

        self._initialized = False
        self._temp_texture: Any | None = None
        self._temp_view: Any | None = None
        self._temp_size: tuple[int, int] = (0, 0)

    def _ensure_intermediate_texture(self, processor: Any) -> Any:
        w = max(1, int(processor.width * self.downscale))
        h = max(1, int(processor.height * self.downscale))
        if self._temp_texture is None or self._temp_size != (w, h):
            self._temp_texture = processor.device.create_texture(
                size=(w, h, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            self._temp_view = self._temp_texture.create_view()
            self._temp_size = (w, h)
        return self._temp_view

    def _init_gpu(self, processor: Any) -> None:
        device = processor.device
        self.uniform_layout = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])

        self.buf = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.bg_uniform = device.create_bind_group(
            layout=self.uniform_layout,
            entries=[{"binding": 0, "resource": {"buffer": self.buf, "offset": 0, "size": 16}}],
        )

        self.pipeline_p1 = processor.create_fullscreen_pipeline(
            GAUSSIAN_BLUR_PASS1_WGSL,
            [processor.single_tex_layout, self.uniform_layout],
            blend=PREMULTIPLIED_ADDITIVE_BLEND,
        )
        self.pipeline_p2 = processor.create_fullscreen_pipeline(
            GAUSSIAN_BLUR_PASS2_WGSL,
            [processor.single_tex_layout, self.uniform_layout],
            blend=PREMULTIPLIED_ADDITIVE_BLEND,
        )
        self._initialized = True

    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        if not self._initialized:
            self._init_gpu(processor)

        data = np.array([
            self.radius,
            0.0,
            float(processor.width),
            float(processor.height),
        ], dtype=np.float32)
        processor.queue.write_buffer(self.buf, 0, data)

        intermediate_view = self._ensure_intermediate_texture(processor)

        # Pase 1: Horizontal -> intermediate_view
        bg_pass1_tex = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": input_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )
        p1 = encoder.begin_render_pass(color_attachments=[{
            "view": intermediate_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0.0, 0.0, 0.0, 0.0),
        }])
        p1.set_pipeline(self.pipeline_p1)
        p1.set_bind_group(0, bg_pass1_tex)
        p1.set_bind_group(1, self.bg_uniform)
        p1.draw(3)
        p1.end()

        # Pase 2: Vertical -> target_view
        bg_pass2_tex = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": intermediate_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )
        p2 = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        p2.set_pipeline(self.pipeline_p2)
        p2.set_bind_group(0, bg_pass2_tex)
        p2.set_bind_group(1, self.bg_uniform)
        p2.draw(3)
        p2.end()

    def copy(self) -> GaussianBlur:
        new_obj = GaussianBlur(radius=self.radius, downscale=self.downscale)
        new_obj.mobject = self.mobject
        return new_obj

    def __deepcopy__(self, memo: dict) -> GaussianBlur:
        return self.copy()


# =====================================================================
# Preset: Film Grain
# =====================================================================

GRAIN_WGSL = """
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct GrainParams {
    intensity: f32,
    speed: f32,
    time: f32,
    colored: f32,
    resolution: vec2f,
    _pad: vec2f,
};
@group(1) @binding(0) var<uniform> params: GrainParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

fn hash21(p: vec2f) -> f32 {
    var p3 = fract(vec3f(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let col = textureSample(in_tex, in_smp, in.uv);
    if (col.a <= 1e-5) {
        discard;
    }

    let t = floor(params.time * max(params.speed, 1.0));
    let pixel_coord = in.position.xy;

    var noise: vec3f;
    if (params.colored > 0.5) {
        let n_r = hash21(pixel_coord + vec2f(t * 13.17, 1.13)) + hash21(pixel_coord + vec2f(t * 7.53, 5.71)) - 1.0;
        let n_g = hash21(pixel_coord + vec2f(t * 17.41, 3.29)) + hash21(pixel_coord + vec2f(t * 9.87, 8.43)) - 1.0;
        let n_b = hash21(pixel_coord + vec2f(t * 23.63, 7.81)) + hash21(pixel_coord + vec2f(t * 11.23, 2.91)) - 1.0;
        noise = vec3f(n_r, n_g, n_b);
    } else {
        let n = hash21(pixel_coord + vec2f(t * 17.13, 3.71)) + hash21(pixel_coord + vec2f(t * 7.89, 11.23)) - 1.0;
        noise = vec3f(n);
    }

    // Composición premultiplicada: modula la energía por el alfa existente
    let grain_rgb = col.rgb + noise * (params.intensity * col.a);
    let final_rgb = max(vec3f(0.0), grain_rgb);

    return vec4f(final_rgb, col.a);
}
"""


class Grain(PostProcessEffect):
    is_group_effect = False

    def __init__(self, intensity: float = 0.08, speed: float = 24.0, colored: bool = False):
        super().__init__(name="Grain", allow_multiple=False)
        self.intensity = float(intensity)
        self.speed = float(speed)
        self.colored = bool(colored)
        self._initialized = False

    def _init_gpu(self, processor: Any) -> None:
        device = processor.device
        self.uniform_layout = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])
        self.buf = device.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.bg_uniform = device.create_bind_group(
            layout=self.uniform_layout,
            entries=[{"binding": 0, "resource": {"buffer": self.buf, "offset": 0, "size": 32}}],
        )
        self.pipeline = processor.create_fullscreen_pipeline(
            GRAIN_WGSL,
            [processor.single_tex_layout, self.uniform_layout],
            blend=PREMULTIPLIED_ADDITIVE_BLEND,
        )
        self._initialized = True

    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        if not self._initialized:
            self._init_gpu(processor)

        data = np.array([
            self.intensity,
            self.speed,
            processor.time,
            1.0 if self.colored else 0.0,
            float(processor.width),
            float(processor.height),
            0.0,
            0.0,
        ], dtype=np.float32)
        processor.queue.write_buffer(self.buf, 0, data)

        tex_bg = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": input_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )

        p = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        p.set_pipeline(self.pipeline)
        p.set_bind_group(0, tex_bg)
        p.set_bind_group(1, self.bg_uniform)
        p.draw(3)
        p.end()

    def copy(self) -> Grain:
        new_obj = Grain(intensity=self.intensity, speed=self.speed, colored=self.colored)
        new_obj.mobject = self.mobject
        return new_obj

    def __deepcopy__(self, memo: dict) -> Grain:
        return self.copy()


# =====================================================================
# Preset: Vignette
# =====================================================================

VIGNETTE_WGSL = """
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct VignetteParams {
    center: vec2f,
    resolution: vec2f,
    radius: f32,
    softness: f32,
    intensity: f32,
    keep_circular: f32,
    color: vec4f,
};
@group(1) @binding(0) var<uniform> params: VignetteParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let col = textureSample(in_tex, in_smp, in.uv);
    if (col.a <= 1e-5) {
        discard;
    }

    var delta = in.uv - params.center;
    if (params.keep_circular > 0.5) {
        let aspect = params.resolution.x / max(params.resolution.y, 1.0);
        delta.x *= aspect;
    }

    let dist = length(delta);
    let edge0 = params.radius;
    let edge1 = max(params.radius + params.softness, edge0 + 1e-4);
    let v_factor = smoothstep(edge0, edge1, dist) * params.intensity;

    // Tinta hacia el color de la viñeta respetando el canal alfa premultiplicado
    let target_tint = params.color.rgb * col.a;
    let final_rgb = mix(col.rgb, target_tint, clamp(v_factor, 0.0, 1.0));

    return vec4f(final_rgb, col.a);
}
"""


class Vignette(PostProcessEffect):
    is_group_effect = True

    def __init__(
        self,
        radius: float = 0.5,
        softness: float = 0.45,
        intensity: float = 0.7,
        center: Sequence[float] = (0.5, 0.5),
        color: Sequence[float] | str | None = None,
        keep_circular: bool = True,
    ):
        super().__init__(name="Vignette", allow_multiple=False)
        self.radius = float(radius)
        self.softness = float(softness)
        self.intensity = float(intensity)
        self.center = (float(center[0]), float(center[1]))
        self.color = color
        self.keep_circular = bool(keep_circular)
        self._initialized = False

    def _get_color(self) -> tuple[float, float, float, float]:
        if self.color is None:
            return (0.0, 0.0, 0.0, 1.0)
        if isinstance(self.color, str):
            try:
                from manimlib.utils.color import color_to_rgba
                return tuple(color_to_rgba(self.color))
            except Exception:
                pass
        if isinstance(self.color, (list, tuple, np.ndarray)):
            if len(self.color) == 3:
                return (float(self.color[0]), float(self.color[1]), float(self.color[2]), 1.0)
            elif len(self.color) >= 4:
                return (
                    float(self.color[0]),
                    float(self.color[1]),
                    float(self.color[2]),
                    float(self.color[3]),
                )
        return (0.0, 0.0, 0.0, 1.0)

    def _init_gpu(self, processor: Any) -> None:
        device = processor.device
        self.uniform_layout = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])
        self.buf = device.create_buffer(size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.bg_uniform = device.create_bind_group(
            layout=self.uniform_layout,
            entries=[{"binding": 0, "resource": {"buffer": self.buf, "offset": 0, "size": 48}}],
        )
        self.pipeline = processor.create_fullscreen_pipeline(
            VIGNETTE_WGSL,
            [processor.single_tex_layout, self.uniform_layout],
            blend=PREMULTIPLIED_ADDITIVE_BLEND,
        )
        self._initialized = True

    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        if not self._initialized:
            self._init_gpu(processor)

        cr, cg, cb, ca = self._get_color()
        data = np.array([
            self.center[0],
            self.center[1],
            float(processor.width),
            float(processor.height),
            self.radius,
            self.softness,
            self.intensity,
            1.0 if self.keep_circular else 0.0,
            cr,
            cg,
            cb,
            ca,
        ], dtype=np.float32)
        processor.queue.write_buffer(self.buf, 0, data)

        tex_bg = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": input_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )

        p = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        p.set_pipeline(self.pipeline)
        p.set_bind_group(0, tex_bg)
        p.set_bind_group(1, self.bg_uniform)
        p.draw(3)
        p.end()

    def copy(self) -> Vignette:
        new_obj = Vignette(
            radius=self.radius,
            softness=self.softness,
            intensity=self.intensity,
            center=self.center,
            color=self.color,
            keep_circular=self.keep_circular,
        )
        new_obj.mobject = self.mobject
        return new_obj

    def __deepcopy__(self, memo: dict) -> Vignette:
        return self.copy()

# =====================================================================
# Preset: Alpha & Luminance Masking (Chained Multi-Masking)
# =====================================================================

MASK_WGSL = """
@group(0) @binding(0) var mob_tex: texture_2d<f32>;
@group(0) @binding(1) var mob_smp: sampler;

@group(1) @binding(0) var mask_tex: texture_2d<f32>;
@group(1) @binding(1) var mask_smp: sampler;

struct MaskParams {
    invert: f32,
    use_luminance: f32,
    mode: f32,
    _pad: f32,
};
@group(2) @binding(0) var<uniform> params: MaskParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let color = textureSample(mob_tex, mob_smp, in.uv);
    let mask_val = textureSample(mask_tex, mask_smp, in.uv);

    var m: f32;
    if (params.use_luminance > 0.5) {
        m = dot(mask_val.rgb, vec3f(0.2126, 0.7152, 0.0722)) * mask_val.a;
    } else {
        m = mask_val.a;
    }

    if (params.invert > 0.5) {
        m = 1.0 - m;
    }

    var factor = m;
    if (params.mode > 0.5) {
        factor = 1.0 - m;
    }

    let final_rgb = color.rgb * factor;
    let final_a = color.a * factor;

    if (final_a <= 1e-5 && dot(final_rgb, vec3f(1.0)) <= 1e-5) {
        discard;
    }

    return vec4f(final_rgb, final_a);
}
"""


class Mask(PostProcessEffect):
    is_group_effect = True

    def __init__(
        self,
        mask: Mobject | Sequence[Mobject],
        invert: bool = False,
        use_luminance: bool = False,
        mode: str = "intersect",
    ):
        super().__init__(name="Mask", allow_multiple=True)
        if isinstance(mask, (list, tuple)):
            from manimlib.mobject.mobject import Group
            self.mask = Group(*mask)
        else:
            self.mask = mask
        self.invert = bool(invert)
        self.use_luminance = bool(use_luminance)
        self.mode = mode.lower()
        self.mask.set_hidden(True)
        self._ensure_solid_stencil(self.mask)
        self._initialized = False

    def _ensure_solid_stencil(self, mob: Mobject) -> None:
        from manimlib.constants import WHITE
        for sm in mob.get_family():
            if hasattr(sm, "get_fill_opacity") and sm.get_fill_opacity() == 0.0:
                sm.set_fill(color=WHITE, opacity=1.0)

    def _init_gpu(self, processor: Any) -> None:
        device = processor.device
        self.uniform_layout = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])
        self.buf = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.bg_uniform = device.create_bind_group(
            layout=self.uniform_layout,
            entries=[{"binding": 0, "resource": {"buffer": self.buf, "offset": 0, "size": 16}}],
        )
        self.pipeline = processor.create_fullscreen_pipeline(
            MASK_WGSL,
            [processor.single_tex_layout, processor.single_tex_layout, self.uniform_layout],
            blend=PREMULTIPLIED_ADDITIVE_BLEND,
        )
        self._initialized = True

    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        if not self._initialized:
            self._init_gpu(processor)

        self._ensure_solid_stencil(self.mask)

        renderer = getattr(processor, "renderer", None)
        if renderer is not None:
            renderer.render_mobject_to_view(encoder, self.mask, processor.mask_view)

        mode_val = 1.0 if self.mode in ("subtract", "difference") else 0.0
        data = np.array([
            1.0 if self.invert else 0.0,
            1.0 if self.use_luminance else 0.0,
            mode_val,
            0.0,
        ], dtype=np.float32)
        processor.queue.write_buffer(self.buf, 0, data)

        bg_mob = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": input_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )
        bg_mask = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": processor.mask_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )

        p = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        p.set_pipeline(self.pipeline)
        p.set_bind_group(0, bg_mob)
        p.set_bind_group(1, bg_mask)
        p.set_bind_group(2, self.bg_uniform)
        p.draw(3)
        p.end()

    def copy(self) -> Mask:
        new_obj = Mask(
            mask=self.mask,
            invert=self.invert,
            use_luminance=self.use_luminance,
            mode=self.mode,
        )
        new_obj.mobject = self.mobject
        return new_obj

    def __deepcopy__(self, memo: dict) -> Mask:
        return self.copy()


# =====================================================================
# Preset: Fast Separable Half-Resolution Gaussian Glow
# =====================================================================

GLOW_PASS1_WGSL = """
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct GlowParams {
    radius: f32,
    intensity: f32,
    threshold: f32,
    softness: f32,
    resolution: vec2f,
    _pad0: f32,
    _pad1: f32,
    tint: vec4f,
};
@group(1) @binding(0) var<uniform> params: GlowParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

fn extract_glow(color: vec4f, threshold: f32, softness: f32) -> vec4f {
    if (color.a <= 1e-4) {
        return vec4f(0.0);
    }
    if (threshold <= 0.0) {
        return color;
    }
    let lum = dot(color.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let knee = max(threshold * softness, 1e-4);
    var soft = lum - threshold + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = (soft * soft) / (4.0 * knee);
    let factor = max(lum - threshold, soft) / max(lum, 1e-4);
    let scale = max(factor, 0.0);
    return vec4f(color.rgb * scale, color.a * scale);
}

const SAMPLES: i32 = 16;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = max(params.radius, 0.5);
    let step_px = max(r / f32(SAMPLES), 1.0);
    let actual_radius = step_px * f32(SAMPLES);
    let sigma = max(actual_radius * 0.38, 1.0);
    let two_sigma_sq = 2.0 * sigma * sigma;

    let inv_res_x = 1.0 / params.resolution.x;

    var accum = vec4f(0.0);
    var total_weight = 0.0;

    for (var i = -SAMPLES; i <= SAMPLES; i = i + 1) {
        let offset_px = f32(i) * step_px;
        let weight = exp(-(offset_px * offset_px) / two_sigma_sq);
        let sample_uv = in.uv + vec2f(offset_px * inv_res_x, 0.0);

        var sample_col = textureSample(in_tex, in_smp, sample_uv);
        sample_col = extract_glow(sample_col, params.threshold, params.softness);

        accum = accum + sample_col * weight;
        total_weight = total_weight + weight;
    }

    return accum / max(total_weight, 1e-4);
}
"""

GLOW_PASS2_WGSL = """
@group(0) @binding(0) var blur_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct GlowParams {
    radius: f32,
    intensity: f32,
    threshold: f32,
    softness: f32,
    resolution: vec2f,
    _pad0: f32,
    _pad1: f32,
    tint: vec4f,
};
@group(1) @binding(0) var<uniform> params: GlowParams;

@group(2) @binding(0) var orig_tex: texture_2d<f32>;
@group(2) @binding(1) var orig_smp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

const SAMPLES: i32 = 16;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = max(params.radius, 0.5);
    let step_px = max(r / f32(SAMPLES), 1.0);
    let actual_radius = step_px * f32(SAMPLES);
    let sigma = max(actual_radius * 0.38, 1.0);
    let two_sigma_sq = 2.0 * sigma * sigma;

    let inv_res_y = 1.0 / params.resolution.y;

    var accum = vec4f(0.0);
    var total_weight = 0.0;

    for (var i = -SAMPLES; i <= SAMPLES; i = i + 1) {
        let offset_px = f32(i) * step_px;
        let weight = exp(-(offset_px * offset_px) / two_sigma_sq);
        let sample_uv = in.uv + vec2f(0.0, offset_px * inv_res_y);

        let sample_col = textureSample(blur_tex, in_smp, sample_uv);
        accum = accum + sample_col * weight;
        total_weight = total_weight + weight;
    }

    let blur_col = accum / max(total_weight, 1e-4);
    let orig = textureSample(orig_tex, orig_smp, in.uv);

    let halo_rgb = blur_col.rgb * params.intensity * params.tint.rgb;
    let final_rgb = orig.rgb + halo_rgb;
    let final_a = orig.a;

    if (final_a <= 1e-5 && dot(final_rgb, vec3f(1.0)) <= 1e-5) {
        discard;
    }

    return vec4f(final_rgb, final_a);
}
"""


class Glow(PostProcessEffect):
    is_group_effect = True

    def __init__(
        self,
        radius: float = 30.0,
        intensity: float = 1.5,
        threshold: float = 0.0,
        softness: float = 0.5,
        color: Sequence[float] | str | None = None,
    ):
        super().__init__(name="Glow", allow_multiple=False)
        self.radius = float(radius)
        self.intensity = float(intensity)
        self.threshold = float(threshold)
        self.softness = float(softness)
        self.color = color

        self._initialized = False
        self._temp_texture: Any | None = None
        self._temp_view: Any | None = None
        self._temp_size: tuple[int, int] = (0, 0)

    def _get_tint(self) -> tuple[float, float, float, float]:
        if self.color is None:
            return (1.0, 1.0, 1.0, 1.0)
        if isinstance(self.color, str):
            try:
                from manimlib.utils.color import color_to_rgba
                return tuple(color_to_rgba(self.color))
            except Exception:
                pass
        if isinstance(self.color, (list, tuple, np.ndarray)):
            if len(self.color) == 3:
                return (float(self.color[0]), float(self.color[1]), float(self.color[2]), 1.0)
            elif len(self.color) >= 4:
                return (float(self.color[0]), float(self.color[1]), float(self.color[2]), float(self.color[3]))
        return (1.0, 1.0, 1.0, 1.0)

    def _ensure_intermediate_texture(self, processor: Any) -> Any:
        w = max(1, int(processor.width) // 2)
        h = max(1, int(processor.height) // 2)
        if self._temp_texture is None or self._temp_size != (w, h):
            self._temp_texture = processor.device.create_texture(
                size=(w, h, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            self._temp_view = self._temp_texture.create_view()
            self._temp_size = (w, h)
        return self._temp_view

    def _init_gpu(self, processor: Any) -> None:
        device = processor.device
        self.uniform_layout = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])

        self.buf = device.create_buffer(size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.bg_uniform = device.create_bind_group(
            layout=self.uniform_layout,
            entries=[{"binding": 0, "resource": {"buffer": self.buf, "offset": 0, "size": 48}}],
        )

        self.pipeline_p1 = processor.create_fullscreen_pipeline(
            GLOW_PASS1_WGSL,
            [processor.single_tex_layout, self.uniform_layout],
            blend=PREMULTIPLIED_ADDITIVE_BLEND,
        )

        self.pipeline_p2 = processor.create_fullscreen_pipeline(
            GLOW_PASS2_WGSL,
            [processor.single_tex_layout, self.uniform_layout, processor.single_tex_layout],
            blend=PREMULTIPLIED_ADDITIVE_BLEND,
        )
        self._initialized = True

    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        if not self._initialized:
            self._init_gpu(processor)

        tint_r, tint_g, tint_b, tint_a = self._get_tint()
        data = np.array([
            self.radius,
            self.intensity,
            self.threshold,
            self.softness,
            float(processor.width),
            float(processor.height),
            0.0,
            0.0,
            tint_r,
            tint_g,
            tint_b,
            tint_a,
        ], dtype=np.float32)
        processor.queue.write_buffer(self.buf, 0, data)

        intermediate_view = self._ensure_intermediate_texture(processor)

        bg_pass1_tex = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": input_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )

        p1 = encoder.begin_render_pass(color_attachments=[{
            "view": intermediate_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0.0, 0.0, 0.0, 0.0),
        }])
        p1.set_pipeline(self.pipeline_p1)
        p1.set_bind_group(0, bg_pass1_tex)
        p1.set_bind_group(1, self.bg_uniform)
        p1.draw(3)
        p1.end()

        bg_pass2_blur = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": intermediate_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )
        bg_pass2_orig = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": input_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )

        p2 = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        p2.set_pipeline(self.pipeline_p2)
        p2.set_bind_group(0, bg_pass2_blur)
        p2.set_bind_group(1, self.bg_uniform)
        p2.set_bind_group(2, bg_pass2_orig)
        p2.draw(3)
        p2.end()

    def copy(self) -> Glow:
        new_obj = Glow(
            radius=self.radius,
            intensity=self.intensity,
            threshold=self.threshold,
            softness=self.softness,
            color=self.color,
        )
        new_obj.mobject = self.mobject
        return new_obj

    def __deepcopy__(self, memo: dict) -> Glow:
        return self.copy()


# =====================================================================
# Preset: Cyberpunk Glitch
# =====================================================================

GLITCH_WGSL = """
@group(0) @binding(0) var layer_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct GlitchParams {
    intensity: f32,
    speed: f32,
    time: f32,
    slice_height: f32,
    resolution: vec2f,
    _pad0: f32,
    _pad1: f32,
};
@group(1) @binding(0) var<uniform> params: GlitchParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

fn hash12(p: vec2f) -> f32 {
    var p3 = fract(vec3f(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2f) -> vec2f {
    var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    if (params.intensity <= 0.0) {
        let col = textureSample(layer_tex, in_smp, in.uv);
        if (col.a <= 1e-4) { discard; }
        return col;
    }

    let rate = max(params.speed * 2.2, 16.0);
    let t_step = floor(params.time * rate);
    let t_sub = floor(params.time * rate * 2.0);

    let burst_rnd = hash12(vec2f(t_step, 19.83));
    let is_active = burst_rnd < clamp(params.intensity * 0.85 + 0.15, 0.25, 0.98);

    var uv = in.uv;

    if (is_active) {
        let chunk_size = max(params.slice_height * 2.5, 36.0);
        let chunk_id = floor(in.position.y / chunk_size);
        let chunk_rnd = hash12(vec2f(chunk_id, t_step));

        if (chunk_rnd > (1.0 - params.intensity * 0.6)) {
            let shift_x = (hash12(vec2f(chunk_id * 7.13, t_step * 3.41)) - 0.5) * params.intensity * 0.08;
            uv.x += shift_x;
        }

        let scan_size = max(params.slice_height * 0.5, 6.0);
        let scan_id = floor(in.position.y / scan_size);
        let scan_rnd = hash12(vec2f(scan_id, t_sub));

        if (scan_rnd > (1.0 - params.intensity * 0.35)) {
            let micro_x = (hash12(vec2f(scan_id * 11.2, t_sub)) - 0.5) * params.intensity * 0.03;
            uv.x += micro_x;
        }

        let v_trigger = hash12(vec2f(t_step, 87.12));
        if (v_trigger > (1.0 - params.intensity * 0.45)) {
            let shift_y = (hash12(vec2f(t_step * 5.2, 13.7)) - 0.5) * params.intensity * 0.025;
            uv.y += shift_y;
        }
    }

    let base = textureSample(layer_tex, in_smp, uv);
    var final_rgb = base.rgb;
    var final_a = base.a;

    if (is_active) {
        let c_rnd = hash22(vec2f(t_step, 45.19));
        let chroma_mag = params.intensity * 0.022;
        let offset = vec2f((c_rnd.x - 0.5) * chroma_mag, (c_rnd.y - 0.5) * chroma_mag * 0.35);

        let sample_r = textureSample(layer_tex, in_smp, uv + offset);
        let sample_c = textureSample(layer_tex, in_smp, uv - offset);

        let red_fringe = vec3f(1.0, 0.08, 0.25) * sample_r.a;
        let cyan_fringe = vec3f(0.05, 0.85, 1.0) * sample_c.a;
        let fringe_a = max(sample_r.a, sample_c.a) * 0.8;

        final_rgb = max(final_rgb, max(red_fringe, cyan_fringe) * 0.85);
        final_a = max(final_a, fringe_a);
    }

    if (final_a <= 1e-4) {
        discard;
    }

    return vec4f(final_rgb, final_a);
}
"""


class Glitch(PostProcessEffect):
    def __init__(self, intensity: float = 0.6, speed: float = 8.0, slice_height: float = 16.0):
        super().__init__(name="Glitch", allow_multiple=False)
        self.intensity = float(intensity)
        self.speed = float(speed)
        self.slice_height = float(slice_height)
        self._initialized = False

    def _init_gpu(self, processor: Any) -> None:
        device = processor.device
        self.uniform_layout = device.create_bind_group_layout(entries=[{
            "binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])
        self.buf = device.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.bg = device.create_bind_group(
            layout=self.uniform_layout,
            entries=[{"binding": 0, "resource": {"buffer": self.buf, "offset": 0, "size": 32}}],
        )
        self.pipeline = processor.create_fullscreen_pipeline(
            GLITCH_WGSL, [processor.single_tex_layout, self.uniform_layout], blend=PREMULTIPLIED_ADDITIVE_BLEND
        )
        self._initialized = True

    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        if not self._initialized:
            self._init_gpu(processor)

        data = np.array([
            self.intensity,
            self.speed,
            processor.time,
            self.slice_height,
            float(processor.width),
            float(processor.height),
            0.0,
            0.0,
        ], dtype=np.float32)
        processor.queue.write_buffer(self.buf, 0, data)

        tex_bg = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[{"binding": 0, "resource": input_view}, {"binding": 1, "resource": processor.sampler}],
        )

        p = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        p.set_pipeline(self.pipeline)
        p.set_bind_group(0, tex_bg)
        p.set_bind_group(1, self.bg)
        p.draw(3)
        p.end()

    def copy(self) -> Glitch:
        new_obj = Glitch(
            intensity=self.intensity,
            speed=self.speed,
            slice_height=self.slice_height,
        )
        new_obj.mobject = self.mobject
        return new_obj

    def __deepcopy__(self, memo: dict) -> Glitch:
        return self.copy()


# =====================================================================
# Preset: Trajectory-Aware Motion Blur
# =====================================================================

MOTION_BLUR_WGSL = """
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct Params {
    direction: vec2f,
    strength: f32,
    _pad: f32,
};

@group(1) @binding(0) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

const SAMPLES: i32 = 16;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let dir = params.direction * params.strength;

    if (length(dir) <= 1e-5) {
        let col = textureSample(
            in_tex,
            in_smp,
            in.uv
        );

        if (col.a <= 1e-4) {
            discard;
        }

        return col;
    }

    var accum = vec4f(0.0);

    let step = 1.0 / f32(SAMPLES - 1);

    for (var i = 0; i < SAMPLES; i = i + 1) {
        let t = f32(i) * step - 0.5;

        accum += textureSample(
            in_tex,
            in_smp,
            in.uv + dir * t
        );
    }

    let col = accum / f32(SAMPLES);

    if (col.a <= 1e-4) {
        discard;
    }

    return col;
}
"""


class MotionBlur(PostProcessEffect):
    """
    Motion blur direccional con tracking 3D.

    `direction` admite:

        (x, y)
        (x, y, z)

    Cuando `auto_direction=True`, la posición 3D del Mobject se proyecta
    mediante la cámara y se compara contra la posición proyectada del
    frame anterior.

    Por tanto funcionan:

        RIGHT / LEFT
        UP / DOWN
        IN / OUT

    y combinaciones arbitrarias de ellas.

    IMPORTANTE:

    Este efecto sigue siendo un motion blur uniforme de pantalla.
    No es todavía un velocity buffer por píxel.
    """

    def __init__(
        self,
        direction: Sequence[float] | None = None,
        strength: float = 1.0,
        auto_direction: bool | None = None,
        max_blur: float = 0.25,
        mobject: Mobject | None = None,
    ):
        super().__init__(
            name="MotionBlur",
            allow_multiple=False,
        )

        # --------------------------------------------------------------
        # MANUAL 3D DIRECTION
        # --------------------------------------------------------------

        self.manual_direction: (
            tuple[float, float, float] | None
        ) = None

        if direction is not None:
            vec = np.asarray(
                direction,
                dtype=np.float64,
            ).reshape(-1)

            if vec.size < 2:
                raise ValueError(
                    "MotionBlur direction must contain "
                    "at least 2 components."
                )

            if vec.size == 2:
                vec = np.array(
                    [
                        vec[0],
                        vec[1],
                        0.0,
                    ],
                    dtype=np.float64,
                )
            else:
                vec = vec[:3]

            norm = float(np.linalg.norm(vec))

            if norm > 1e-12:
                vec /= norm
            else:
                vec[:] = (
                    1.0,
                    0.0,
                    0.0,
                )

            self.manual_direction = (
                float(vec[0]),
                float(vec[1]),
                float(vec[2]),
            )

        # --------------------------------------------------------------
        # MODE
        # --------------------------------------------------------------

        if auto_direction is None:
            self.auto_direction = direction is None
        else:
            self.auto_direction = bool(
                auto_direction
            )

        self.strength = float(strength)
        self.max_blur = float(max_blur)

        self.mobject = mobject

        # --------------------------------------------------------------
        # FRAME STATE
        # --------------------------------------------------------------

        # Position projected during the previous rendered frame.
        self._prev_projected_pos: (
            np.ndarray | None
        ) = None

        # Time corresponding to _prev_projected_pos.
        self._last_motion_time: (
            float | None
        ) = None

        # Result calculated for the current frame.
        self._cached_dir: tuple[
            float,
            float,
        ] = (
            1.0,
            0.0,
        )

        self._cached_strength = 0.0

        # Suaviza la entrada/salida del blur para evitar picos al iniciar
        # movimiento después de varios frames estáticos.
        self._smoothed_strength = 0.0
        self._strength_smoothing = 0.28
        self._strength_release = 0.45
        self._was_moving = False

        self._initialized = False

    # ==================================================================
    # ATTACH
    # ==================================================================

    def attach(
        self,
        mobject: Mobject,
    ) -> None:
        super().attach(mobject)

        self.mobject = mobject

        self._prev_projected_pos = None
        self._last_motion_time = None

        self._cached_dir = (
            1.0,
            0.0,
        )

        self._cached_strength = 0.0
        self._smoothed_strength = 0.0
        self._was_moving = False

    # ==================================================================
    # WORLD POSITION
    # ==================================================================

    def _get_world_center(
        self,
        processor: Any,
    ) -> np.ndarray | None:
        """
        Obtiene el centro 3D del Mobject en coordenadas de mundo.

        No aplica ninguna transformación de cámara aquí.
        """

        if self.mobject is None:
            return None

        try:
            if hasattr(
                self.mobject,
                "refresh_bounding_box",
            ):
                self.mobject.refresh_bounding_box(
                    recurse_down=True,
                )

            points = []

            for sm in self.mobject.get_family():
                try:
                    pts = sm.get_points()
                except Exception:
                    continue

                if pts is None:
                    continue

                if len(pts) == 0:
                    continue

                pts = np.asarray(
                    pts,
                    dtype=np.float64,
                )

                if pts.ndim != 2:
                    continue

                if pts.shape[1] < 2:
                    continue

                points.append(pts)

            if points:
                all_points = np.vstack(points)

                if all_points.shape[1] >= 3:
                    center = np.mean(
                        all_points[:, :3],
                        axis=0,
                    )
                else:
                    center = np.array(
                        [
                            np.mean(
                                all_points[:, 0]
                            ),
                            np.mean(
                                all_points[:, 1]
                            ),
                            0.0,
                        ],
                        dtype=np.float64,
                    )
            else:
                center = np.asarray(
                    self.mobject.get_center(),
                    dtype=np.float64,
                )

                if center.size < 3:
                    center = np.pad(
                        center,
                        (
                            0,
                            3 - center.size,
                        ),
                    )

                center = center[:3]

            return center.astype(
                np.float64,
                copy=False,
            )

        except Exception:
            return None

    # ==================================================================
    # CAMERA
    # ==================================================================

    def _get_camera(
        self,
        processor: Any,
    ) -> Any | None:
        return getattr(
            processor,
            "camera",
            None,
        )

    def _get_camera_frame(
        self,
        processor: Any,
    ) -> Any | None:
        camera = self._get_camera(
            processor
        )

        if camera is None:
            return None

        return getattr(
            camera,
            "frame",
            None,
        )

    # ==================================================================
    # 3D -> 2D PROJECTION
    # ==================================================================

    def _project_world_point(
        self,
        processor: Any,
        point: np.ndarray,
    ) -> np.ndarray:
        """
        Proyecta un punto 3D de mundo a coordenadas 2D de cámara.

        El punto se transforma primero mediante la view matrix de
        CameraFrame y después se aplica la perspectiva.

        MUY IMPORTANTE:

        No hacemos simplemente:

            view @ point -> xy

        porque eso ignora la perspectiva.

        Tampoco proyectamos un vector desde el origen. Para motion
        tracking se proyectan posiciones absolutas.
        """

        frame = self._get_camera_frame(
            processor
        )

        point = np.asarray(
            point,
            dtype=np.float64,
        ).reshape(-1)

        if point.size < 3:
            point = np.pad(
                point,
                (
                    0,
                    3 - point.size,
                ),
            )

        point = point[:3]

        if frame is None:
            return point[:2].copy()

        # --------------------------------------------------------------
        # VIEW MATRIX
        # --------------------------------------------------------------

        try:
            view = np.asarray(
                frame.get_view_matrix(),
                dtype=np.float64,
            )

            if view.shape != (4, 4):
                raise ValueError(
                    "Invalid camera view matrix."
                )

            p = np.array(
                [
                    point[0],
                    point[1],
                    point[2],
                    1.0,
                ],
                dtype=np.float64,
            )

            # CameraFrame.to_fixed_frame_point() applies the view
            # matrix as p @ view.T, i.e. as view @ p for column vectors.
            # Using view.T here transposes the camera transform and breaks
            # screen-space motion tracking when the camera moves.
            camera_point = view @ p

        except Exception:
            # ----------------------------------------------------------
            # FALLBACK
            # ----------------------------------------------------------

            center = np.zeros(
                3,
                dtype=np.float64,
            )

            try:
                center = np.asarray(
                    frame.get_center(),
                    dtype=np.float64,
                ).reshape(-1)

                if center.size < 3:
                    center = np.pad(
                        center,
                        (
                            0,
                            3 - center.size,
                        ),
                    )

                center = center[:3]

            except Exception:
                pass

            return (
                point[:2] - center[:2]
            )

        # --------------------------------------------------------------
        # CAMERA PARAMETERS
        # --------------------------------------------------------------

        try:
            focal_distance = float(
                frame.get_focal_distance()
            )
        except Exception:
            focal_distance = 20.0

        try:
            scale = float(
                frame.get_scale()
            )
        except Exception:
            scale = 1.0

        focal_distance = max(
            abs(focal_distance),
            1e-8,
        )

        scale = max(
            abs(scale),
            1e-8,
        )

        # --------------------------------------------------------------
        # PERSPECTIVE
        # --------------------------------------------------------------

        #
        # CameraFrame's perspective convention places the camera
        # relative to focal_distance. The projected position is:
        #
        #        xy * focal / (focal - z)
        #
        # where z is camera-space depth.
        #
        z = float(
            camera_point[2]
        )

        denominator = (
            focal_distance - z
        )

        # Avoid singularity at the camera plane.
        if abs(denominator) < 1e-8:
            denominator = (
                1e-8
                if denominator >= 0.0
                else -1e-8
            )

        perspective = (
            focal_distance
            / denominator
        )

        projected = (
            camera_point[:2]
            * perspective
            / scale
        )

        return projected

    # ==================================================================
    # FRAME DIMENSIONS
    # ==================================================================

    def _get_frame_dimensions(
        self,
        processor: Any,
    ) -> tuple[float, float]:
        """
        Dimensions del frame en unidades de escena.
        """

        fw = 14.222222222222221
        fh = 8.0

        camera = self._get_camera(
            processor
        )

        if camera is None:
            return (
                fw,
                fh,
            )

        frame = getattr(
            camera,
            "frame",
            None,
        )

        if frame is not None:
            try:
                if hasattr(
                    frame,
                    "get_width",
                ):
                    fw = float(
                        frame.get_width()
                    )
            except Exception:
                pass

            try:
                if hasattr(
                    frame,
                    "get_height",
                ):
                    fh = float(
                        frame.get_height()
                    )
            except Exception:
                pass

        else:
            try:
                if (
                    hasattr(
                        camera,
                        "get_frame_width",
                    )
                    and hasattr(
                        camera,
                        "get_frame_height",
                    )
                ):
                    fw = float(
                        camera.get_frame_width()
                    )

                    fh = float(
                        camera.get_frame_height()
                    )
            except Exception:
                pass

        return (
            max(fw, 1e-8),
            max(fh, 1e-8),
        )

    # ==================================================================
    # GPU INITIALIZATION
    # ==================================================================

    def _init_gpu(
        self,
        processor: Any,
    ) -> None:
        device = processor.device

        self.uniform_layout = (
            device.create_bind_group_layout(
                entries=[
                    {
                        "binding": 0,
                        "visibility": (
                            wgpu.ShaderStage.FRAGMENT
                        ),
                        "buffer": {
                            "type": (
                                wgpu.BufferBindingType.uniform
                            ),
                        },
                    },
                ],
            )
        )

        self.buf = device.create_buffer(
            size=16,
            usage=(
                wgpu.BufferUsage.UNIFORM
                | wgpu.BufferUsage.COPY_DST
            ),
        )

        self.bg = device.create_bind_group(
            layout=self.uniform_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.buf,
                        "offset": 0,
                        "size": 16,
                    },
                },
            ],
        )

        self.pipeline = (
            processor.create_fullscreen_pipeline(
                MOTION_BLUR_WGSL,
                [
                    processor.single_tex_layout,
                    self.uniform_layout,
                ],
                blend=PREMULTIPLIED_ADDITIVE_BLEND,
            )
        )

        self._initialized = True

    # ==================================================================
    # MOTION UPDATE
    # ==================================================================

    def _update_velocity(
        self,
        processor: Any,
    ) -> tuple[float, float, float]:
        """
        Calcula el movimiento correspondiente al frame actual.

        Esta función es deliberadamente idempotente por frame.

        Puede ser llamada por:

            is_active()
            apply()

        sin consumir dos veces el mismo frame.
        """

        curr_time = getattr(
            processor,
            "time",
            None,
        )

        # ==============================================================
        # SAME FRAME CACHE
        # ==============================================================

        if (
            curr_time is not None
            and self._last_motion_time == curr_time
        ):
            return (
                self._cached_dir[0],
                self._cached_dir[1],
                self._cached_strength,
            )

        # ==============================================================
        # MANUAL MODE
        # ==============================================================

        if (
            not self.auto_direction
            or self.mobject is None
        ):
            if self.manual_direction is None:
                result = (
                    1.0,
                    0.0,
                    self.strength,
                )

                self._cached_dir = (
                    result[0],
                    result[1],
                )

                self._cached_strength = (
                    result[2]
                )

                self._last_motion_time = (
                    curr_time
                )

                return result

            dx, dy, dz = (
                self.manual_direction
            )

            # ----------------------------------------------------------
            # Project the 3D direction around the actual object position.
            #
            # This is essential for perspective.
            # ----------------------------------------------------------

            origin = self._get_world_center(
                processor
            )

            if origin is None:
                origin = np.zeros(
                    3,
                    dtype=np.float64,
                )

            direction3 = np.array(
                [
                    dx,
                    dy,
                    dz,
                ],
                dtype=np.float64,
            )

            norm = float(
                np.linalg.norm(direction3)
            )

            if norm > 1e-8:
                direction3 /= norm

            # Small local displacement.
            #
            # We don't use project(direction3) - project(0), because
            # that would incorrectly assume the object is at the
            # camera origin.
            epsilon = 1.0

            p0 = self._project_world_point(
                processor,
                origin,
            )

            p1 = self._project_world_point(
                processor,
                origin + direction3 * epsilon,
            )

            screen_delta = (
                p1 - p0
            )

            fw, fh = (
                self._get_frame_dimensions(
                    processor
                )
            )

            uv_delta = np.array(
                [
                    screen_delta[0] / fw,
                    -screen_delta[1] / fh,
                ],
                dtype=np.float64,
            )

            magnitude = float(
                np.linalg.norm(
                    uv_delta
                )
            )

            if magnitude > 1e-8:
                screen_direction = (
                    uv_delta / magnitude
                )

                result = (
                    float(
                        screen_direction[0]
                    ),
                    float(
                        screen_direction[1]
                    ),
                    min(
                        magnitude
                        * self.strength,
                        self.max_blur,
                    ),
                )

            else:
                result = (
                    1.0,
                    0.0,
                    0.0,
                )

            self._cached_dir = (
                result[0],
                result[1],
            )

            self._cached_strength = (
                result[2]
            )

            self._last_motion_time = (
                curr_time
            )

            return result

        # ==============================================================
        # AUTOMATIC 3D TRACKING
        # ==============================================================

        world_pos = self._get_world_center(
            processor
        )

        if world_pos is None:
            self._last_motion_time = (
                curr_time
            )

            self._cached_strength = 0.0

            return (
                self._cached_dir[0],
                self._cached_dir[1],
                0.0,
            )

        current_projected = (
            self._project_world_point(
                processor,
                world_pos,
            )
        )

        # ==============================================================
        # FIRST FRAME
        # ==============================================================

        if self._prev_projected_pos is None:
            self._prev_projected_pos = (
                current_projected.copy()
            )

            self._last_motion_time = (
                curr_time
            )

            self._cached_dir = (
                1.0,
                0.0,
            )

            self._cached_strength = 0.0

            return (
                1.0,
                0.0,
                0.0,
            )

        # ==============================================================
        # SCREEN-SPACE DISPLACEMENT
        # ==============================================================

        delta = (
            current_projected
            - self._prev_projected_pos
        )

        fw, fh = (
            self._get_frame_dimensions(
                processor
            )
        )

        # Camera coordinates:
        #
        #   +X -> right
        #   +Y -> up
        #
        # UV coordinates:
        #
        #   +U -> right
        #   +V -> down
        #
        du = float(delta[0]) / fw
        dv = -float(delta[1]) / fh

        displacement = np.array(
            [
                du,
                dv,
            ],
            dtype=np.float64,
        )

        magnitude = float(
            np.linalg.norm(
                displacement
            )
        )

        # ==============================================================
        # MOVING
        # ==============================================================

        if magnitude > 1e-8:
            direction = (
                displacement / magnitude
            )

            dir_x = float(
                direction[0]
            )

            dir_y = float(
                direction[1]
            )

            active_strength = min(
                magnitude * self.strength,
                self.max_blur,
            )

        # ==============================================================
        # NOT MOVING
        # ==============================================================

        else:
            dir_x = self._cached_dir[0]
            dir_y = self._cached_dir[1]
            active_strength = 0.0

        # ==============================================================
        # SMOOTH STRENGTH
        # ==============================================================
        #
        # A sudden transition from rest to motion can produce a large
        # one-frame displacement. Ease the blur in and out instead of
        # feeding that value directly to the shader.
        #
        # Suppress the onset impulse: a large first displacement after
        # a static frame must not immediately create a full-strength blur.
        moving_now = magnitude > 1e-8
        if moving_now and not self._was_moving:
            active_strength = min(
                active_strength,
                self.max_blur * 0.15,
            )

        # Use separate attack/release rates. The onset is capped above,
        # then sustained motion can ramp up normally; stopping fades out.
        alpha = (
            self._strength_smoothing
            if active_strength > self._smoothed_strength
            else self._strength_release
        )
        self._smoothed_strength += alpha * (
            active_strength - self._smoothed_strength
        )
        active_strength = min(
            max(self._smoothed_strength, 0.0),
            self.max_blur,
        )
        self._was_moving = moving_now

        # ==============================================================
        # COMMIT FRAME
        # ==============================================================

        #
        # THIS IS THE ONLY PLACE where the previous position advances.
        #
        self._prev_projected_pos = (
            current_projected.copy()
        )

        self._last_motion_time = (
            curr_time
        )

        self._cached_dir = (
            dir_x,
            dir_y,
        )

        self._cached_strength = (
            active_strength
        )

        return (
            dir_x,
            dir_y,
            active_strength,
        )

    # ==================================================================
    # ACTIVE
    # ==================================================================

    def is_active(
        self,
        processor: Any,
    ) -> bool:
        """
        Consulta el estado del frame sin consumirlo dos veces.
        """

        _, _, strength = (
            self._update_velocity(
                processor
            )
        )

        return strength > 1e-4

    # ==================================================================
    # APPLY
    # ==================================================================

    def apply(
        self,
        processor: Any,
        encoder: Any,
        input_view: Any,
        target_view: Any,
    ) -> None:

        if not self._initialized:
            self._init_gpu(
                processor
            )

        dir_x, dir_y, strength = (
            self._update_velocity(
                processor
            )
        )

        # --------------------------------------------------------------
        # UNIFORM
        # --------------------------------------------------------------

        processor.queue.write_buffer(
            self.buf,
            0,
            np.array(
                [
                    dir_x,
                    dir_y,
                    strength,
                    0.0,
                ],
                dtype=np.float32,
            ),
        )

        # --------------------------------------------------------------
        # INPUT TEXTURE
        # --------------------------------------------------------------

        tex_bg = (
            processor.device.create_bind_group(
                layout=processor.single_tex_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": input_view,
                    },
                    {
                        "binding": 1,
                        "resource": processor.sampler,
                    },
                ],
            )
        )

        # --------------------------------------------------------------
        # FULLSCREEN PASS
        # --------------------------------------------------------------

        p = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target_view,
                    "load_op": wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                },
            ],
        )

        p.set_pipeline(
            self.pipeline
        )

        p.set_bind_group(
            0,
            tex_bg,
        )

        p.set_bind_group(
            1,
            self.bg,
        )

        p.draw(3)

        p.end()

    # ==================================================================
    # COPY
    # ==================================================================

    def copy(self) -> MotionBlur:
        new_obj = MotionBlur(
            direction=self.manual_direction,
            strength=self.strength,
            auto_direction=self.auto_direction,
            max_blur=self.max_blur,
            mobject=self.mobject,
        )

        # Nunca copiar el historial temporal.
        #
        # El nuevo efecto empieza con un frame de referencia nuevo.
        new_obj._prev_projected_pos = None
        new_obj._last_motion_time = None

        new_obj._cached_dir = (
            1.0,
            0.0,
        )

        new_obj._cached_strength = 0.0

        return new_obj

    # ==================================================================
    # DEEPCOPY
    # ==================================================================

    def __deepcopy__(
        self,
        memo: dict,
    ) -> MotionBlur:
        return self.copy()

# =====================================================================
# Preset: Dynamic Zoom Blur
# =====================================================================

ZOOM_BLUR_WGSL = """
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct ZoomBlurParams {
    center: vec2f,
    strength: f32,
    decay: f32,
    t_start: f32,
    t_end: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(1) @binding(0) var<uniform> params: ZoomBlurParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

const SAMPLES: i32 = 32;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    if (abs(params.strength) <= 1e-5) {
        let col = textureSample(in_tex, in_smp, in.uv);
        if (col.a <= 1e-4) {
            discard;
        }
        return col;
    }

    let ray = in.uv - params.center;
    var accum = vec4f(0.0);
    var total_weight = 0.0;

    for (var i = 0; i < SAMPLES; i = i + 1) {
        let progress = f32(i) / f32(SAMPLES - 1);
        let t = mix(params.t_start, params.t_end, progress);
        let scale = 1.0 + t * params.strength;
        let sample_uv = params.center + ray * scale;

        let in_bounds = step(0.0, sample_uv.x) * step(sample_uv.x, 1.0) *
                        step(0.0, sample_uv.y) * step(sample_uv.y, 1.0);

        let weight = exp(-4.0 * t * t) * pow(clamp(params.decay, 0.01, 1.0), abs(t) * f32(SAMPLES)) * in_bounds;
        let sample_col = textureSample(in_tex, in_smp, sample_uv);

        accum += sample_col * weight;
        total_weight += weight;
    }

    let col = accum / max(total_weight, 1e-4);
    if (col.a <= 1e-4) {
        discard;
    }
    return col;
}
"""


class ZoomBlur(PostProcessEffect):
    def __init__(
        self,
        strength: float = 1.0,
        center: Sequence[float] | None = None,
        uv_center: Sequence[float] | None = None,
        decay: float = 0.98,
        mode: str = "auto",
        auto_scale: bool = True,
        max_blur: float = 0.45,
        mobject: Mobject | None = None,
    ):
        super().__init__(name="ZoomBlur", allow_multiple=False)
        self.strength = float(strength)
        self.center = center
        self.uv_center = uv_center
        self.decay = float(decay)
        self.mode = mode.lower()
        self.auto_scale = bool(auto_scale)
        self.max_blur = float(max_blur)
        self.mobject = mobject

        self._prev_scale: float | None = None
        self._last_time: float | None = None
        self._cached_strength: float = 0.0
        self._cached_t_range: tuple[float, float] = (-0.5, 0.5)
        self._initialized = False

    def attach(self, mobject: Mobject) -> None:
        super().attach(mobject)
        self._prev_scale = None
        self._last_time = None

    def _get_frame_dimensions(self, processor: Any) -> tuple[float, float]:
        fw, fh = 14.222222222222221, 8.0
        cam = getattr(processor, "camera", None)
        if cam is not None:
            frame = getattr(cam, "frame", None)
            if frame is not None:
                if hasattr(frame, "get_width"):
                    fw = float(frame.get_width())
                if hasattr(frame, "get_height"):
                    fh = float(frame.get_height())
            elif hasattr(cam, "get_frame_width") and hasattr(cam, "get_frame_height"):
                fw = float(cam.get_frame_width())
                fh = float(cam.get_frame_height())
        return max(fw, 1e-4), max(fh, 1e-4)

    def _get_mobject_scale_uv(self, processor: Any) -> float | None:
        if self.mobject is None:
            return None
        try:
            w = float(self.mobject.get_width())
            h = float(self.mobject.get_height())
            fw, fh = self._get_frame_dimensions(processor)
            w_uv = w / fw
            h_uv = h / fh
            diag = float(np.hypot(w_uv, h_uv))
            return max(diag, 1e-5)
        except Exception:
            return None

    def _point_to_uv(self, processor: Any, point: Sequence[float]) -> tuple[float, float]:
        pt = np.array(point[:3], dtype=np.float64)
        cam = getattr(processor, "camera", None)
        if cam is not None:
            frame = getattr(cam, "frame", None)
            if frame is not None and hasattr(frame, "get_center"):
                pt = pt - np.array(frame.get_center(), dtype=np.float64)
        fw, fh = self._get_frame_dimensions(processor)
        u = 0.5 + float(pt[0]) / fw
        v = 0.5 - float(pt[1]) / fh
        return u, v

    def _get_focal_uv(self, processor: Any) -> tuple[float, float]:
        if self.uv_center is not None:
            return float(self.uv_center[0]), float(self.uv_center[1])

        if self.center is not None:
            return self._point_to_uv(processor, self.center)

        if self.mobject is not None:
            try:
                center_world = self.mobject.get_center()
                return self._point_to_uv(processor, center_world)
            except Exception:
                pass

        return 0.5, 0.5

    def _init_gpu(self, processor: Any) -> None:
        device = processor.device
        self.uniform_layout = device.create_bind_group_layout(entries=[{
            "binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])
        self.buf = device.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.bg = device.create_bind_group(
            layout=self.uniform_layout,
            entries=[{"binding": 0, "resource": {"buffer": self.buf, "offset": 0, "size": 32}}],
        )
        self.pipeline = processor.create_fullscreen_pipeline(
            ZOOM_BLUR_WGSL, [processor.single_tex_layout, self.uniform_layout], blend=PREMULTIPLIED_ADDITIVE_BLEND
        )
        self._initialized = True

    def _update_scale_strength(self, processor: Any) -> tuple[float, tuple[float, float]]:
        active_strength = self.strength
        t_start, t_end = -0.5, 0.5

        if self.auto_scale and self.mobject is not None:
            curr_scale = self._get_mobject_scale_uv(processor)
            curr_time = getattr(processor, "time", None)

            if curr_time is not None and self._last_time == curr_time:
                return self._cached_strength, self._cached_t_range

            if self._prev_scale is not None and curr_scale is not None:
                delta_s = curr_scale - self._prev_scale
                rel_rate = abs(delta_s) / max(self._prev_scale, 1e-4)

                if rel_rate > 1e-5:
                    active_strength = min(rel_rate * self.strength * 10.0, self.max_blur)
                    if self.mode == "auto":
                        t_start, t_end = (0.0, 1.0) if delta_s >= 0 else (-1.0, 0.0)
                    elif self.mode == "outward":
                        t_start, t_end = 0.0, 1.0
                    elif self.mode == "inward":
                        t_start, t_end = -1.0, 0.0
                    else:
                        t_start, t_end = -0.5, 0.5
                else:
                    active_strength = 0.0
            else:
                active_strength = 0.0

            if curr_scale is not None:
                self._prev_scale = curr_scale
            self._last_time = curr_time
            self._cached_strength = active_strength
            self._cached_t_range = (t_start, t_end)
        else:
            if self.mode == "outward":
                t_start, t_end = 0.0, 1.0
            elif self.mode == "inward":
                t_start, t_end = -1.0, 0.0
            else:
                t_start, t_end = -0.5, 0.5

        return active_strength, (t_start, t_end)

    def is_active(self, processor: Any) -> bool:
        strength, _ = self._update_scale_strength(processor)
        return strength > 1e-4

    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        if not self._initialized:
            self._init_gpu(processor)

        cx, cy = self._get_focal_uv(processor)
        active_strength, (t_start, t_end) = self._update_scale_strength(processor)

        data = np.array([
            cx, cy,
            active_strength,
            self.decay,
            t_start, t_end,
            0.0, 0.0,
        ], dtype=np.float32)
        processor.queue.write_buffer(self.buf, 0, data)

        tex_bg = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[{"binding": 0, "resource": input_view}, {"binding": 1, "resource": processor.sampler}],
        )

        p = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        p.set_pipeline(self.pipeline)
        p.set_bind_group(0, tex_bg)
        p.set_bind_group(1, self.bg)
        p.draw(3)
        p.end()

    def copy(self) -> ZoomBlur:
        new_obj = ZoomBlur(
            strength=self.strength,
            center=self.center,
            uv_center=self.uv_center,
            decay=self.decay,
            mode=self.mode,
            auto_scale=self.auto_scale,
            max_blur=self.max_blur,
            mobject=self.mobject,
        )
        return new_obj

    def __deepcopy__(self, memo: dict) -> ZoomBlur:
        return self.copy()


# =====================================================================
# Preset: Chromatic Aberration
# =====================================================================

CA_WGSL = """
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct Params {
    offset: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(1) @binding(0) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let shift = (in.uv - vec2f(0.5)) * params.offset;
    let base = textureSample(in_tex, in_smp, in.uv);
    let sample_r = textureSample(in_tex, in_smp, in.uv + shift);
    let sample_b = textureSample(in_tex, in_smp, in.uv - shift);

    let lum_r = dot(sample_r.rgb, vec3f(0.299, 0.587, 0.114));
    let lum_b = dot(sample_b.rgb, vec3f(0.299, 0.587, 0.114));

    let red_fringe = vec3f(1.0, 0.1, 0.2) * max(lum_r, sample_r.r) * sample_r.a * 0.7;
    let blue_fringe = vec3f(0.0, 0.6, 1.0) * max(lum_b, max(sample_b.g, sample_b.b)) * sample_b.a * 0.7;

    let final_rgb = max(base.rgb, max(red_fringe, blue_fringe));
    let final_a = max(base.a, max(sample_r.a * 0.7, sample_b.a * 0.7));

    if (final_a <= 1e-4) {
        discard;
    }
    return vec4f(final_rgb, final_a);
}
"""


class ChromaticAberration(PostProcessEffect):
    def __init__(self, offset: float = 0.02):
        super().__init__(name="ChromaticAberration", allow_multiple=False)
        self.offset = float(offset)
        self._initialized = False

    def _init_gpu(self, processor: Any) -> None:
        device = processor.device
        self.uniform_layout = device.create_bind_group_layout(entries=[{
            "binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])
        self.buf = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.bg = device.create_bind_group(
            layout=self.uniform_layout,
            entries=[{"binding": 0, "resource": {"buffer": self.buf, "offset": 0, "size": 16}}],
        )
        self.pipeline = processor.create_fullscreen_pipeline(
            CA_WGSL, [processor.single_tex_layout, self.uniform_layout], blend=PREMULTIPLIED_ADDITIVE_BLEND
        )
        self._initialized = True

    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        if not self._initialized:
            self._init_gpu(processor)

        processor.queue.write_buffer(self.buf, 0, np.array([self.offset, 0.0, 0.0, 0.0], dtype=np.float32))
        tex_bg = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[{"binding": 0, "resource": input_view}, {"binding": 1, "resource": processor.sampler}],
        )
        p = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        p.set_pipeline(self.pipeline)
        p.set_bind_group(0, tex_bg)
        p.set_bind_group(1, self.bg)
        p.draw(3)
        p.end()

    def copy(self) -> ChromaticAberration:
        new_obj = ChromaticAberration(offset=self.offset)
        new_obj.mobject = self.mobject
        return new_obj

    def __deepcopy__(self, memo: dict) -> ChromaticAberration:
        return self.copy()


# =====================================================================
# Preset: Custom User-Defined WGSL Effect
# =====================================================================

CUSTOM_POST_WGSL_TEMPLATE = """
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var color = textureSample(in_tex, in_smp, in.uv);
    if (color.a <= 1e-4) {
        discard;
    }
    {USER_CODE}
    return color;
}
"""


class CustomVFX(PostProcessEffect):
    def __init__(self, fragment_code: str, name: str = "CustomVFX", allow_multiple: bool = False):
        super().__init__(name=name, allow_multiple=allow_multiple)
        self.fragment_code = fragment_code
        self._initialized = False

    def _init_gpu(self, processor: Any) -> None:
        shader = CUSTOM_POST_WGSL_TEMPLATE.replace("{USER_CODE}", self.fragment_code)
        self.pipeline = processor.create_fullscreen_pipeline(
            shader, [processor.single_tex_layout], blend=PREMULTIPLIED_ADDITIVE_BLEND
        )
        self._initialized = True

    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        if not self._initialized:
            self._init_gpu(processor)

        tex_bg = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[{"binding": 0, "resource": input_view}, {"binding": 1, "resource": processor.sampler}],
        )
        p = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        p.set_pipeline(self.pipeline)
        p.set_bind_group(0, tex_bg)
        p.draw(3)
        p.end()

    def copy(self) -> CustomVFX:
        new_obj = CustomVFX(fragment_code=self.fragment_code, name=self.name, allow_multiple=self.allow_multiple)
        new_obj.mobject = self.mobject
        return new_obj

    def __deepcopy__(self, memo: dict) -> CustomVFX:
        return self.copy()


VFX = PostProcessEffect
CustomPostProcess = CustomVFX
GlowEffect = Glow
RadialBlur = ZoomBlur
RadialBlurEffect = ZoomBlur


# ---------------------------------------------------------------------
# Extensión dinámica de Mobject
# ---------------------------------------------------------------------

def set_hidden(self: Mobject, hidden: bool = True, recurse: bool = True) -> Mobject:
    targets = self.get_family(recurse=recurse)
    for mob in targets:
        mob.hidden = bool(hidden)
    return self


def is_hidden(self: Mobject) -> bool:
    return getattr(self, "hidden", False)


def add_vfx(self: Mobject, effect: PostProcessEffect | str, recurse: bool = False) -> Mobject:
    if isinstance(effect, str):
        effect = CustomVFX(fragment_code=effect)

    if not hasattr(self, "_vfx_effects"):
        self._vfx_effects = []

    eff = effect.copy()
    eff._source_effect = effect  # Guarda referencia al objeto original
    effect._target_effect = eff
    eff.attach(self)

    if not getattr(eff, "allow_multiple", False):
        existing = [e for e in self._vfx_effects if e.name == eff.name]
        if existing:
            self._vfx_effects.remove(existing[0])

    self._vfx_effects.append(eff)

    is_group = getattr(eff, "is_group_effect", False) or (eff.name in ("Glow", "Mask", "LiquidGlass", "DropShadow", "GaussianBlur", "Vignette"))

    for mob in self.get_family():
        if is_group:
            mob._vfx_group = self
        else:
            mob._vfx_local = self
            if not hasattr(mob, "_vfx_group"):
                mob._vfx_group = self

    return self


def remove_vfx(self: Mobject, effect_type_or_name: Any, recurse: bool = False) -> Mobject:
    if hasattr(self, "_vfx_effects"):
        if isinstance(effect_type_or_name, str):
            self._vfx_effects = [e for e in self._vfx_effects if e.name != effect_type_or_name]
        elif isinstance(effect_type_or_name, type):
            self._vfx_effects = [e for e in self._vfx_effects if not isinstance(e, effect_type_or_name)]
        else:
            # Comprueba la instancia exacta, la referencia original guardada o el nombre del efecto
            target_name = getattr(effect_type_or_name, "name", None)
            self._vfx_effects = [
                e for e in self._vfx_effects
                if e is not effect_type_or_name 
                and getattr(e, "_source_effect", None) is not effect_type_or_name
                and (target_name is None or e.name != target_name)
            ]

        if not any(getattr(e, "is_group_effect", False) or e.name in ("Glow", "Mask", "LiquidGlass", "DropShadow") for e in self._vfx_effects):
            for mob in self.get_family():
                if hasattr(mob, "_vfx_group") and mob._vfx_group is self:
                    delattr(mob, "_vfx_group")
                if hasattr(mob, "_vfx_local") and mob._vfx_local is self:
                    delattr(mob, "_vfx_local")
    return self


def clear_vfx(self: Mobject, recurse: bool = False) -> Mobject:
    for mob in self.get_family():
        if hasattr(mob, "_vfx_effects"):
            mob._vfx_effects.clear()
        if hasattr(mob, "_vfx_group"):
            delattr(mob, "_vfx_group")
        if hasattr(mob, "_vfx_local"):
            delattr(mob, "_vfx_local")
    return self


def has_vfx(self: Mobject) -> bool:
    return bool(getattr(self, "_vfx_effects", None))


def get_vfx_list(self: Mobject) -> list[PostProcessEffect]:
    return getattr(self, "_vfx_effects", [])


def set_mask(
    self: Mobject,
    mask: Mobject | Sequence[Mobject],
    invert: bool = False,
    use_luminance: bool = False,
    mode: str = "intersect",
) -> Mobject:
    self.remove_mask()
    self.add_mask(mask, invert=invert, use_luminance=use_luminance, mode=mode)
    return self


def add_mask(
    self: Mobject,
    mask: Mobject | Sequence[Mobject],
    invert: bool = False,
    use_luminance: bool = False,
    mode: str = "intersect",
) -> Mobject:
    if isinstance(mask, (list, tuple)):
        from manimlib.mobject.mobject import Group
        mask = Group(*mask)
    mask.set_hidden(True)
    self.add_vfx(Mask(mask, invert=invert, use_luminance=use_luminance, mode=mode))
    return self


def remove_mask(self: Mobject, mask: Mobject | None = None) -> Mobject:
    if hasattr(self, "_vfx_effects"):
        masks_to_remove = [
            eff for eff in self._vfx_effects
            if eff.name == "Mask" and (mask is None or getattr(eff, "mask", None) is mask)
        ]
        for eff in masks_to_remove:
            if hasattr(eff, "mask"):
                eff.mask.set_hidden(False)
            self._vfx_effects.remove(eff)

        has_other_group_effects = any(getattr(e, "name", "") in ("Glow", "Mask", "LiquidGlass") for e in self._vfx_effects)
        if not has_other_group_effects:
            for mob in self.get_family():
                if hasattr(mob, "_vfx_group") and mob._vfx_group is self:
                    delattr(mob, "_vfx_group")
    return self


# =====================================================================
# Preset: Liquid Glass (port de OverShifted/LiquidGlass, MIT)
# =====================================================================
#
# The blur is fullscreen, just like the blur stage of the original OpenGL
# implementation. The actual glass surface is a scene-space quad, so its
# vertices pass through Manim's real project_point() path.

LG_BLUR_WGSL = """
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct BlurParams {
    offset: vec2f,
    _pad: vec2f,
};
@group(1) @binding(0) var<uniform> params: BlurParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let st = params.offset;
    var c = textureSampleLevel(in_tex, in_smp, in.uv, 0.0) * 0.1964825501511404;
    c += (textureSampleLevel(in_tex, in_smp, in.uv + st * 1.411764705882353, 0.0)
        + textureSampleLevel(in_tex, in_smp, in.uv - st * 1.411764705882353, 0.0)) * 0.2969069646728344;
    c += (textureSampleLevel(in_tex, in_smp, in.uv + st * 3.2941176470588234, 0.0)
        + textureSampleLevel(in_tex, in_smp, in.uv - st * 3.2941176470588234, 0.0)) * 0.09447039785044732;
    c += (textureSampleLevel(in_tex, in_smp, in.uv + st * 5.176470588235294, 0.0)
        + textureSampleLevel(in_tex, in_smp, in.uv - st * 5.176470588235294, 0.0)) * 0.010381362401148057;
    return vec4f(c.rgb, 1.0);
}
"""

# project_point.wgsl is inserted instead of reimplementing projection math.
# It is the exact shader file supplied for this renderer and imports
# frame_units.wgsl itself. The custom mob struct below deliberately exposes the
# members that project_point() reads.
LIQUID_GLASS_PROJECTED_WGSL = """
#INSERT frame_uniforms.wgsl

struct GlassMobjectUniforms {
    // First four vec4s are the world-space quad corners.
    corners: array<vec4f, 4>,
    is_fixed_in_frame: f32,
    clip_plane0: vec4f,
    clip_plane1: vec4f,
    clip_plane2: vec4f,
    clip_plane3: vec4f,
};
@group(1) @binding(0) var<uniform> mob: GlassMobjectUniforms;

#INSERT project_point.wgsl

struct GlassParams {
    power: f32,
    f_power: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    noise: f32,
    glow_weight: f32,
    glow_bias: f32,
    glow_edge0: f32,
    glow_edge1: f32,
    opacity: f32,
    layer_mix: f32,
    clip_layer: f32,
    resolution: vec2f,
};
@group(2) @binding(0) var<uniform> params: GlassParams;

@group(3) @binding(0) var mob_tex: texture_2d<f32>;
@group(3) @binding(1) var mob_smp: sampler;

@group(4) @binding(0) var blur_tex: texture_2d<f32>;
@group(4) @binding(1) var blur_smp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) local_uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    // Two triangles over the four scene-space corners.
    var corner_index = array<u32, 6>(
        0u, 1u, 2u,
        0u, 2u, 3u,
    );
    var local_uv = array<vec2f, 6>(
        vec2f(0.0, 0.0),
        vec2f(1.0, 0.0),
        vec2f(1.0, 1.0),
        vec2f(0.0, 0.0),
        vec2f(1.0, 1.0),
        vec2f(0.0, 1.0),
    );

    let projection = project_point(mob.corners[corner_index[index]].xyz);
    var out: VertexOutput;
    out.position = projection.position;
    out.local_uv = local_uv[index];
    return out;
}

fn sd_superellipse(p: vec2f, n: f32, r: f32) -> f32 {
    let ap = max(abs(p), vec2f(1e-6));
    let num = pow(ap.x, n) + pow(ap.y, n) - pow(r, n);
    let den = n * sqrt(pow(ap.x, 2.0 * n - 2.0) + pow(ap.y, 2.0 * n - 2.0)) + 0.00001;
    return num / den;
}

fn f_curve(x: f32) -> f32 {
    return 1.0 - params.b * pow(params.c * 2.718281828459045, -params.d * x - params.a);
}

fn rand(co: vec2f) -> f32 {
    return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let screen_uv = clamp(
        in.position.xy / max(params.resolution, vec2f(1.0)),
        vec2f(0.0),
        vec2f(1.0),
    );

    // Original OverShifted coordinate system. Only the surface projection is
    // different: the glass quad itself is projected through Manim's 3D camera.
    let p = in.local_uv * 2.0 - vec2f(1.0);
    let d = sd_superellipse(p, params.power, 1.0);
    if (d > 0.0) {
        discard;
    }

    let dist = -d;
    let f = pow(f_curve(dist), params.f_power);
    let sample_p = p * f;
    let local_delta = (sample_p - p) * 0.5;

    // Map the original local displacement through the projected surface.
    let du_dx = dpdx(in.local_uv.x);
    let du_dy = dpdy(in.local_uv.x);
    let dv_dx = dpdx(in.local_uv.y);
    let dv_dy = dpdy(in.local_uv.y);
    let det = du_dx * dv_dy - du_dy * dv_dx;

    var sample_uv = screen_uv;
    if (abs(det) > 1e-8) {
        let delta_px = vec2f(
            (dv_dy * local_delta.x - du_dy * local_delta.y) / det,
            (-dv_dx * local_delta.x + du_dx * local_delta.y) / det,
        );
        sample_uv = clamp(
            screen_uv + delta_px / max(params.resolution, vec2f(1.0)),
            vec2f(0.0),
            vec2f(1.0),
        );
    }

    let bg = textureSampleLevel(blur_tex, blur_smp, sample_uv, 0.0).rgb;
    let n = rand(in.position.xy * 1e-3) - 0.5;

    let den = params.glow_edge1 - params.glow_edge0;
    let edge_den = select(den, 1e-5, abs(den) < 1e-5);
    let t = clamp((dist - params.glow_edge0) / edge_den, 0.0, 1.0);
    let edge = t * t * (3.0 - 2.0 * t);
    let glow = sin(atan2(p.y, p.x) - 0.5);
    let mul = glow * params.glow_weight * edge + 1.0 + params.glow_bias;

    let glass = (bg + vec3f(n * params.noise)) * mul;

    // Manim integration only; disabled by default, so there is no additional
    // Fresnel/specular/dome material beyond the original OverShifted shader.
    let layer = textureSampleLevel(mob_tex, mob_smp, screen_uv, 0.0);
    let ga = params.opacity;
    let lay = layer * params.layer_mix;
    let out_rgb = glass * ga * (1.0 - lay.a) + lay.rgb;
    let out_a = ga * (1.0 - lay.a) + lay.a;
    return vec4f(out_rgb, out_a);
}"""


class VFXAnimation(Animation):
    def __init__(
        self,
        mobject: Mobject,
        effect: PostProcessEffect,
        update_params_func: Callable[[float], dict],
        **kwargs,
    ):
        super().__init__(mobject, **kwargs)
        self.effect = effect
        self.update_params_func = update_params_func

    def interpolate_mobject(self, alpha: float):
        alpha = self.rate_func(alpha)
        params = self.update_params_func(alpha)

        # El efecto entregado por el usuario puede ser el objeto original
        # o directamente una copia renderizable.
        source = getattr(self.effect, "_target_effect", None)

        targets = [self.effect]

        if source is not None and source is not self.effect:
            targets.append(source)

        for attached in getattr(self.effect, "_attached_effects", []):
            if attached not in targets:
                targets.append(attached)

        if source is not None:
            for attached in getattr(source, "_attached_effects", []):
                if attached not in targets:
                    targets.append(attached)

        for target in targets:
            for key, value in params.items():
                setattr(target, key, value)


class LiquidGlass(PostProcessEffect):
    """
    Liquid glass equivalente al port visual de OverShifted/LiquidGlass.

    La superficie se trata como un panel 3D: sus cuatro esquinas viven en
    coordenadas de escena, pasan por project_point.wgsl y solo después se
    rasterizan. Así una rotación 3D de la cámara cambia realmente la forma
    proyectada del vidrio, en vez de mover un rectángulo 2D de UV. El mobject
    original aporta geometría/máscara, pero su fill/stroke no se compone salvo
    que show_mobject=True.
    """

    def __init__(
        self,
        power: float = 3.0,
        refraction_power: float = 1.0,
        a: float = 0.7,
        b: float = 2.3,
        c: float = 5.2,
        d: float = 6.9,
        blur_radius: float = 2.0,
        blur_iterations: int = 1,
        blur_downscale: float = 0.5,
        noise: float = 0.06,
        glow_weight: float = 0.25,
        glow_bias: float = 0.0,
        glow_edge0: float = 0.5,
        glow_edge1: float = -0.5,
        opacity: float = 1.0,
        show_mobject: bool = False,
        clip_to_layer: bool = False,
    ):
        super().__init__(name="LiquidGlass", allow_multiple=False)
        self.power = max(float(power), 1.001)
        self.refraction_power = float(refraction_power)
        self.a = float(a)
        self.b = float(b)
        self.c = max(float(c), 0.1)
        self.d = float(d)
        self.blur_radius = float(blur_radius)
        self.blur_iterations = max(int(blur_iterations), 0)
        self.blur_downscale = min(max(float(blur_downscale), 0.1), 1.0)
        self.noise = float(noise)
        self.glow_weight = float(glow_weight)
        self.glow_bias = float(glow_bias)
        self.glow_edge0 = float(glow_edge0)
        self.glow_edge1 = float(glow_edge1)
        self.opacity = float(opacity)
        self.show_mobject = bool(show_mobject)
        self.clip_to_layer = bool(clip_to_layer)

        self._initialized = False
        self._blur_size: tuple[int, int] = (0, 0)
        self._blur_tex_a: Any | None = None
        self._blur_view_a: Any | None = None
        self._blur_tex_b: Any | None = None
        self._blur_view_b: Any | None = None
        self._last_u: np.ndarray | None = None
        self._last_normal: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _get_world_points(self) -> np.ndarray | None:
        mob = self.mobject
        if mob is None:
            return None
        try:
            if hasattr(mob, "refresh_bounding_box"):
                mob.refresh_bounding_box(recurse_down=True)

            if hasattr(mob, "get_all_points"):
                pts = np.asarray(mob.get_all_points(), dtype=np.float64)
            else:
                pts_list = [
                    np.asarray(sm.get_points(), dtype=np.float64)
                    for sm in mob.get_family()
                    if len(sm.get_points()) > 0
                ]
                if not pts_list:
                    return None
                pts = np.vstack(pts_list)

            if pts.ndim != 2 or pts.shape[1] < 3:
                return None
            pts = pts[:, :3]
            pts = pts[np.isfinite(pts).all(axis=1)]
            return pts if len(pts) >= 2 else None
        except Exception:
            return None

    @staticmethod
    def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= 1e-9:
            return fallback.astype(np.float64)
        return np.asarray(v / n, dtype=np.float64)

    def _get_quad_world(self) -> np.ndarray:
        """Build an oriented scene-space rectangle around the mobject."""
        pts = self._get_world_points()
        if pts is None:
            mob = self.mobject
            if mob is None:
                center = np.zeros(3, dtype=np.float64)
                hw, hh = 1.0, 1.0
            else:
                try:
                    center = np.asarray(mob.get_center(), dtype=np.float64)
                    hw = max(float(mob.get_width()) * 0.5, 1e-4)
                    hh = max(float(mob.get_height()) * 0.5, 1e-4)
                except Exception:
                    center = np.zeros(3, dtype=np.float64)
                    hw, hh = 1.0, 1.0
            return np.array([
                center + [-hw, -hh, 0.0],
                center + [ hw, -hh, 0.0],
                center + [ hw,  hh, 0.0],
                center + [-hw,  hh, 0.0],
            ], dtype=np.float64)

        center = pts.mean(axis=0)
        centered = pts - center

        # Estimate the object's plane normal by PCA. If the object is planar,
        # the smallest-variance eigenvector is its normal, independently of the
        # camera orientation.
        try:
            cov = np.dot(centered.T, centered) / max(len(centered), 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)
            normal = self._normalize(
                eigvecs[:, order[0]],
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
            )
        except Exception:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Prefer an actual consecutive path direction when available. This is
        # more stable than PCA for squares, whose two largest eigenvalues match.
        u = None
        if len(pts) >= 2:
            deltas = np.diff(pts, axis=0)
            lengths = np.linalg.norm(deltas, axis=1)
            for idx in np.argsort(lengths)[::-1]:
                candidate = deltas[int(idx)]
                candidate = candidate - normal * float(np.dot(candidate, normal))
                if np.linalg.norm(candidate) > 1e-7:
                    u = self._normalize(candidate, np.array([1.0, 0.0, 0.0]))
                    break

        if u is None:
            u = self._normalize(
                eigvecs[:, order[-1]] if 'eigvecs' in locals() else np.array([1.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            )
            u = self._normalize(u - normal * float(np.dot(u, normal)), np.array([1.0, 0.0, 0.0]))

        # Keep orientation continuous across frames so glow direction does not
        # jump when an eigenvector changes sign numerically.
        if self._last_normal is not None and float(np.dot(normal, self._last_normal)) < 0.0:
            normal = -normal
        if self._last_u is not None and float(np.dot(u, self._last_u)) < 0.0:
            u = -u

        v = self._normalize(
            np.cross(normal, u),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
        # Re-orthogonalize in case of numerical drift.
        u = self._normalize(np.cross(v, normal), u)

        self._last_normal = normal.copy()
        self._last_u = u.copy()

        ext_u = centered @ u
        ext_v = centered @ v
        min_u, max_u = float(ext_u.min()), float(ext_u.max())
        min_v, max_v = float(ext_v.min()), float(ext_v.max())

        center = center + u * (0.5 * (min_u + max_u)) + v * (0.5 * (min_v + max_v))
        hu = max(0.5 * (max_u - min_u), 1e-4)
        hv = max(0.5 * (max_v - min_v), 1e-4)

        return np.array([
            center - u * hu - v * hv,
            center + u * hu - v * hv,
            center + u * hu + v * hv,
            center - u * hu + v * hv,
        ], dtype=np.float64)

    def _get_fixed_in_frame(self, mob: Mobject) -> float:
        try:
            return float(mob.uniforms["is_fixed_in_frame"])
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # GPU resources
    # ------------------------------------------------------------------
    def _init_gpu(self, processor: Any) -> None:
        device = processor.device
        gpu = processor.gpu

        self.uniform_layout = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])

        self.geometry_layout = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.VERTEX,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])

        def make_uniform(size: int) -> tuple[Any, Any]:
            buf = device.create_buffer(
                size=size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            bg = device.create_bind_group(
                layout=self.uniform_layout,
                entries=[{
                    "binding": 0,
                    "resource": {"buffer": buf, "offset": 0, "size": size},
                }],
            )
            return buf, bg

        self.buf_h0, self.bg_h0 = make_uniform(16)
        self.buf_h, self.bg_h = make_uniform(16)
        self.buf_v, self.bg_v = make_uniform(16)
        self.buf_zero, self.bg_zero = make_uniform(16)

        # 4 x vec4 corners (64) + fixed/pad (16) + 4 clip planes (64) = 144.
        self.buf_geometry = device.create_buffer(
            size=144,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.bg_geometry = device.create_bind_group(
            layout=self.geometry_layout,
            entries=[{
                "binding": 0,
                "resource": {"buffer": self.buf_geometry, "offset": 0, "size": 144},
            }],
        )

        self.buf_glass, self.bg_glass = make_uniform(64)
        processor.queue.write_buffer(self.buf_zero, 0, np.zeros(4, dtype=np.float32))

        self.pipeline_blur = processor.create_fullscreen_pipeline(
            LG_BLUR_WGSL,
            [processor.single_tex_layout, self.uniform_layout],
        )

        # Gpu.module() compiles WGSL verbatim; unlike Material shader loading, it
        # does not expand Manim's #INSERT directives. Resolve the shader source
        # through the same insert loader used by the normal WebGPU renderer.
        from manimlib.renderer.shader_source import read_shader_file
        shader_code = LIQUID_GLASS_PROJECTED_WGSL
        for marker in re.findall(r"^#INSERT .*\.wgsl$", shader_code, flags=re.MULTILINE):
            inserted = read_shader_file(
                os.path.join("inserts", marker.replace("#INSERT ", ""))
            )
            if inserted is None:
                raise RuntimeError(f"Unable to resolve WGSL insertion: {marker}")
            shader_code = shader_code.replace(marker, inserted)
        shader_module = gpu.module(shader_code)
        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[
            gpu.frame_layout,
            self.geometry_layout,
            self.uniform_layout,
            processor.single_tex_layout,
            processor.single_tex_layout,
        ])
        self.pipeline_glass = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{
                    "format": wgpu.TextureFormat.rgba8unorm,
                    "blend": PREMULTIPLIED_ADDITIVE_BLEND,
                }],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.none,
            },
        )
        self._initialized = True

    def _ensure_blur_textures(self, processor: Any) -> tuple[Any, Any]:
        w = max(1, int(processor.width * self.blur_downscale))
        h = max(1, int(processor.height * self.blur_downscale))
        if self._blur_tex_a is None or self._blur_size != (w, h):
            usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
            self._blur_tex_a = processor.device.create_texture(
                size=(w, h, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=usage,
            )
            self._blur_tex_b = processor.device.create_texture(
                size=(w, h, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=usage,
            )
            self._blur_view_a = self._blur_tex_a.create_view()
            self._blur_view_b = self._blur_tex_b.create_view()
            self._blur_size = (w, h)
        return self._blur_view_a, self._blur_view_b

    def _blur_pass(self, processor: Any, encoder: Any, src_view: Any, dst_view: Any, uniform_bg: Any) -> None:
        src_bg = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": src_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )
        p = encoder.begin_render_pass(color_attachments=[{
            "view": dst_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0.0, 0.0, 0.0, 1.0),
        }])
        p.set_pipeline(self.pipeline_blur)
        p.set_bind_group(0, src_bg)
        p.set_bind_group(1, uniform_bg)
        p.draw(3)
        p.end()

    # ------------------------------------------------------------------
    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        if processor.bg_view is None or self.mobject is None:
            return
        if not self._initialized:
            self._init_gpu(processor)

        view_a, view_b = self._ensure_blur_textures(processor)
        full_w = max(float(processor.width), 1.0)
        full_h = max(float(processor.height), 1.0)
        blur_w = max(float(self._blur_size[0]), 1.0)
        blur_h = max(float(self._blur_size[1]), 1.0)
        r = self.blur_radius
        q = processor.queue

        q.write_buffer(self.buf_h0, 0, np.array([r / full_w, 0.0, 0.0, 0.0], dtype=np.float32))
        q.write_buffer(self.buf_h, 0, np.array([r / blur_w, 0.0, 0.0, 0.0], dtype=np.float32))
        q.write_buffer(self.buf_v, 0, np.array([0.0, r / blur_h, 0.0, 0.0], dtype=np.float32))

        if self.blur_iterations == 0:
            self._blur_pass(processor, encoder, processor.bg_view, view_b, self.bg_zero)
        else:
            for i in range(self.blur_iterations):
                src = processor.bg_view if i == 0 else view_b
                self._blur_pass(
                    processor,
                    encoder,
                    src,
                    view_a,
                    self.bg_h0 if i == 0 else self.bg_h,
                )
                self._blur_pass(processor, encoder, view_a, view_b, self.bg_v)

        quad = self._get_quad_world().astype(np.float32)
        geom_data = np.zeros(36, dtype=np.float32)
        geom_data[:16] = np.concatenate(
            [quad, np.zeros((4, 1), dtype=np.float32)],
            axis=1,
        ).reshape(-1)
        geom_data[16] = self._get_fixed_in_frame(self.mobject)
        # Clip planes are intentionally disabled for the synthetic glass quad.
        # project_point() still receives the fields it expects; all-zero planes
        # evaluate to "keep everything".
        q.write_buffer(self.buf_geometry, 0, geom_data)

        glass_data = np.array([
            self.power,
            self.refraction_power,
            self.a,
            self.b,
            self.c,
            self.d,
            self.noise,
            self.glow_weight,
            self.glow_bias,
            self.glow_edge0,
            self.glow_edge1,
            self.opacity,
            1.0 if self.show_mobject else 0.0,
            1.0 if self.clip_to_layer else 0.0,
            float(processor.width),
            float(processor.height),
        ], dtype=np.float32)
        q.write_buffer(self.buf_glass, 0, glass_data)

        bg_mob = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": input_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )
        bg_blur = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": view_b},
                {"binding": 1, "resource": processor.sampler},
            ],
        )

        gpu = processor.gpu
        p = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        p.set_pipeline(self.pipeline_glass)
        p.set_bind_group(0, gpu.frame_bind_group)
        p.set_bind_group(1, self.bg_geometry)
        p.set_bind_group(2, self.bg_glass)
        p.set_bind_group(3, bg_mob)
        p.set_bind_group(4, bg_blur)
        p.draw(6)
        p.end()

    def _params(self) -> dict:
        return dict(
            power=self.power,
            refraction_power=self.refraction_power,
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
            blur_radius=self.blur_radius,
            blur_iterations=self.blur_iterations,
            blur_downscale=self.blur_downscale,
            noise=self.noise,
            glow_weight=self.glow_weight,
            glow_bias=self.glow_bias,
            glow_edge0=self.glow_edge0,
            glow_edge1=self.glow_edge1,
            opacity=self.opacity,
            show_mobject=self.show_mobject,
            clip_to_layer=self.clip_to_layer,
        )

    def copy(self) -> LiquidGlass:
        new_obj = LiquidGlass(**self._params())
        new_obj.mobject = self.mobject
        return new_obj

    def __deepcopy__(self, memo: dict) -> LiquidGlass:
        return self.copy()


def set_liquid_glass(self: Mobject, **kwargs: Any) -> Mobject:
    """mob.set_liquid_glass(power=3, blur_radius=2, ...) -> mob"""
    self.add_vfx(LiquidGlass(**kwargs))
    return self


Mobject.hidden = False
Mobject.set_hidden = set_hidden
Mobject.is_hidden = is_hidden
Mobject.add_vfx = add_vfx
Mobject.remove_vfx = remove_vfx
Mobject.clear_vfx = clear_vfx
Mobject.has_vfx = has_vfx
Mobject.get_vfx_list = get_vfx_list
Mobject.set_mask = set_mask
Mobject.add_mask = add_mask
Mobject.remove_mask = remove_mask
Mobject.set_liquid_glass = set_liquid_glass

DROP_SHADOW_PASS1_WGSL = """
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct DropShadowParams {
    offset: vec2f,
    resolution: vec2f,
    radius: f32,
    opacity: f32,
    shadow_only: f32,
    _pad: f32,
    color: vec4f,
};
@group(1) @binding(0) var<uniform> params: DropShadowParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

const SAMPLES: i32 = 16;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = max(params.radius, 0.5);
    let step_px = max(r / f32(SAMPLES), 1.0);
    let actual_radius = step_px * f32(SAMPLES);
    let sigma = max(actual_radius * 0.38, 1.0);
    let two_sigma_sq = 2.0 * sigma * sigma;

    let inv_res_x = 1.0 / params.resolution.x;
    // En Manim +Y es hacia arriba; en UV +V es hacia abajo.
    let shadow_center_uv = in.uv - vec2f(
        params.offset.x / params.resolution.x,
        -params.offset.y / params.resolution.y
    );

    var accum_a = 0.0;
    var total_weight = 0.0;

    for (var i = -SAMPLES; i <= SAMPLES; i = i + 1) {
        let offset_px = f32(i) * step_px;
        let weight = exp(-(offset_px * offset_px) / two_sigma_sq);
        let sample_uv = shadow_center_uv + vec2f(offset_px * inv_res_x, 0.0);

        let sample_col = textureSample(in_tex, in_smp, sample_uv);
        accum_a += sample_col.a * weight;
        total_weight += weight;
    }

    let blurred_a = accum_a / max(total_weight, 1e-4);
    return vec4f(blurred_a, blurred_a, blurred_a, blurred_a);
}
"""

DROP_SHADOW_PASS2_WGSL = """
@group(0) @binding(0) var blur_tex: texture_2d<f32>;
@group(0) @binding(1) var in_smp: sampler;

struct DropShadowParams {
    offset: vec2f,
    resolution: vec2f,
    radius: f32,
    opacity: f32,
    shadow_only: f32,
    _pad: f32,
    color: vec4f,
};
@group(1) @binding(0) var<uniform> params: DropShadowParams;

@group(2) @binding(0) var orig_tex: texture_2d<f32>;
@group(2) @binding(1) var orig_smp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

const SAMPLES: i32 = 16;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = max(params.radius, 0.5);
    let step_px = max(r / f32(SAMPLES), 1.0);
    let actual_radius = step_px * f32(SAMPLES);
    let sigma = max(actual_radius * 0.38, 1.0);
    let two_sigma_sq = 2.0 * sigma * sigma;

    let inv_res_y = 1.0 / params.resolution.y;

    var accum_a = 0.0;
    var total_weight = 0.0;

    for (var i = -SAMPLES; i <= SAMPLES; i = i + 1) {
        let offset_px = f32(i) * step_px;
        let weight = exp(-(offset_px * offset_px) / two_sigma_sq);
        let sample_uv = in.uv + vec2f(0.0, offset_px * inv_res_y);

        let s = textureSample(blur_tex, in_smp, sample_uv);
        accum_a += s.r * weight;
        total_weight += weight;
    }

    let blurred_a = accum_a / max(total_weight, 1e-4);
    let shadow_a = clamp(blurred_a * params.opacity * params.color.a, 0.0, 1.0);
    let shadow_rgb = params.color.rgb * shadow_a;

    let show_orig = 1.0 - params.shadow_only;
    let orig = textureSample(orig_tex, orig_smp, in.uv) * show_orig;

    // Composición Porter-Duff 'Over' premultiplicada: Original sobre Sombra
    let final_rgb = orig.rgb + shadow_rgb * (1.0 - orig.a);
    let final_a = orig.a + shadow_a * (1.0 - orig.a);

    if (final_a <= 1e-5 && dot(final_rgb, vec3f(1.0)) <= 1e-5) {
        discard;
    }

    return vec4f(final_rgb, final_a);
}
"""


class DropShadow(PostProcessEffect):
    is_group_effect = True

    def __init__(
        self,
        offset: Sequence[float] = (6.0, -6.0),
        radius: float = 12.0,
        opacity: float = 0.75,
        color: Sequence[float] | str | None = None,
        shadow_only: bool = False,
    ):
        super().__init__(name="DropShadow", allow_multiple=False)
        self.offset = (float(offset[0]), float(offset[1]))
        self.radius = float(radius)
        self.opacity = float(opacity)
        self.color = color
        self.shadow_only = bool(shadow_only)

        self._initialized = False
        self._temp_texture: Any | None = None
        self._temp_view: Any | None = None
        self._temp_size: tuple[int, int] = (0, 0)

    def _get_color(self) -> tuple[float, float, float, float]:
        if self.color is None:
            return (0.0, 0.0, 0.0, 1.0)
        if isinstance(self.color, str):
            try:
                from manimlib.utils.color import color_to_rgba
                return tuple(color_to_rgba(self.color))
            except Exception:
                pass
        if isinstance(self.color, (list, tuple, np.ndarray)):
            if len(self.color) == 3:
                return (float(self.color[0]), float(self.color[1]), float(self.color[2]), 1.0)
            elif len(self.color) >= 4:
                return (
                    float(self.color[0]),
                    float(self.color[1]),
                    float(self.color[2]),
                    float(self.color[3]),
                )
        return (0.0, 0.0, 0.0, 1.0)

    def _ensure_intermediate_texture(self, processor: Any) -> Any:
        w = max(1, int(processor.width))
        h = max(1, int(processor.height))
        if self._temp_texture is None or self._temp_size != (w, h):
            self._temp_texture = processor.device.create_texture(
                size=(w, h, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            self._temp_view = self._temp_texture.create_view()
            self._temp_size = (w, h)
        return self._temp_view

    def _init_gpu(self, processor: Any) -> None:
        device = processor.device
        self.uniform_layout = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])

        self.buf = device.create_buffer(size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.bg_uniform = device.create_bind_group(
            layout=self.uniform_layout,
            entries=[{"binding": 0, "resource": {"buffer": self.buf, "offset": 0, "size": 48}}],
        )

        self.pipeline_p1 = processor.create_fullscreen_pipeline(
            DROP_SHADOW_PASS1_WGSL,
            [processor.single_tex_layout, self.uniform_layout],
            blend=PREMULTIPLIED_ADDITIVE_BLEND,
        )

        self.pipeline_p2 = processor.create_fullscreen_pipeline(
            DROP_SHADOW_PASS2_WGSL,
            [processor.single_tex_layout, self.uniform_layout, processor.single_tex_layout],
            blend=PREMULTIPLIED_ADDITIVE_BLEND,
        )
        self._initialized = True

    def apply(self, processor: Any, encoder: Any, input_view: Any, target_view: Any) -> None:
        if not self._initialized:
            self._init_gpu(processor)

        c_r, c_g, c_b, c_a = self._get_color()
        data = np.array([
            self.offset[0],
            self.offset[1],
            float(processor.width),
            float(processor.height),
            self.radius,
            self.opacity,
            1.0 if self.shadow_only else 0.0,
            0.0,
            c_r,
            c_g,
            c_b,
            c_a,
        ], dtype=np.float32)
        processor.queue.write_buffer(self.buf, 0, data)

        intermediate_view = self._ensure_intermediate_texture(processor)

        bg_pass1_tex = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": input_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )

        p1 = encoder.begin_render_pass(color_attachments=[{
            "view": intermediate_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0.0, 0.0, 0.0, 0.0),
        }])
        p1.set_pipeline(self.pipeline_p1)
        p1.set_bind_group(0, bg_pass1_tex)
        p1.set_bind_group(1, self.bg_uniform)
        p1.draw(3)
        p1.end()

        bg_pass2_blur = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": intermediate_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )
        bg_pass2_orig = processor.device.create_bind_group(
            layout=processor.single_tex_layout,
            entries=[
                {"binding": 0, "resource": input_view},
                {"binding": 1, "resource": processor.sampler},
            ],
        )

        p2 = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        p2.set_pipeline(self.pipeline_p2)
        p2.set_bind_group(0, bg_pass2_blur)
        p2.set_bind_group(1, self.bg_uniform)
        p2.set_bind_group(2, bg_pass2_orig)
        p2.draw(3)
        p2.end()

    def copy(self) -> DropShadow:
        new_obj = DropShadow(
            offset=self.offset,
            radius=self.radius,
            opacity=self.opacity,
            color=self.color,
            shadow_only=self.shadow_only,
        )
        new_obj.mobject = self.mobject
        return new_obj

    def __deepcopy__(self, memo: dict) -> DropShadow:
        return self.copy()


def set_drop_shadow(
    self: Mobject,
    offset: Sequence[float] = (6.0, -6.0),
    radius: float = 12.0,
    opacity: float = 0.75,
    color: Sequence[float] | str | None = None,
    shadow_only: bool = False,
) -> Mobject:
    self.add_vfx(DropShadow(
        offset=offset,
        radius=radius,
        opacity=opacity,
        color=color,
        shadow_only=shadow_only,
    ))
    return self


def set_gaussian_blur(self: Mobject, radius: float = 12.0, downscale: float = 1.0) -> Mobject:
    self.add_vfx(GaussianBlur(radius=radius, downscale=downscale))
    return self

def set_grain(self: Mobject, intensity: float = 0.08, speed: float = 24.0, colored: bool = False) -> Mobject:
    self.add_vfx(Grain(intensity=intensity, speed=speed, colored=colored))
    return self

def set_vignette(
    self: Mobject,
    radius: float = 0.5,
    softness: float = 0.45,
    intensity: float = 0.7,
    center: Sequence[float] = (0.5, 0.5),
    color: Sequence[float] | str | None = None,
    keep_circular: bool = True,
) -> Mobject:
    self.add_vfx(Vignette(
        radius=radius,
        softness=softness,
        intensity=intensity,
        center=center,
        color=color,
        keep_circular=keep_circular,
    ))
    return self

# Aliases
Blur = GaussianBlur
FilmGrain = Grain

Mobject.set_gaussian_blur = set_gaussian_blur
Mobject.set_grain = set_grain
Mobject.set_vignette = set_vignette
Shadow = DropShadow
ShadowEffect = DropShadow
Mobject.set_drop_shadow = set_drop_shadow
Mobject.hidden = False
Mobject.set_hidden = set_hidden
Mobject.is_hidden = is_hidden
Mobject.add_vfx = add_vfx
Mobject.remove_vfx = remove_vfx
Mobject.clear_vfx = clear_vfx
Mobject.has_vfx = has_vfx
Mobject.get_vfx_list = get_vfx_list
Mobject.set_mask = set_mask
Mobject.add_mask = add_mask
Mobject.remove_mask = remove_mask
Mobject.set_liquid_glass = set_liquid_glass