from __future__ import annotations
import os
import numpy as np
from typing import Callable, Any

from manimlib.constants import FRAME_WIDTH, FRAME_HEIGHT, ORIGIN
from manimlib.mobject.mobject import Mobject
from manimlib.renderer.drawing import Drawing
from manimlib.renderer.uniform_block import (
    COMMON_UNIFORMS,
    uniform_block_dtype,
    Uniforms,
)
from manimlib.utils.directories import get_shader_dir


def infer_uniform_size(val: Any) -> int:
    if callable(val):
        return infer_uniform_size(val(0.0))
    if isinstance(val, (int, float, np.floating, np.integer)):
        return 1
    if isinstance(val, (list, tuple, np.ndarray)):
        length = len(val)
        if length in (2, 3, 4, 16):
            return length
        raise ValueError(f"Dimensión vectorial {length} no admitida en std140.")
    raise TypeError(f"Tipo no reconocido para uniforme: {type(val)}")


def sanitize_uniform_value(val: Any, size: int) -> Any:
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val)
    return np.array(val, dtype=np.float32)


SHADER_MOBJECT_TEMPLATE = """
#INSERT frame_uniforms.wgsl
#INSERT mobject_uniforms.wgsl
#INSERT read_data.wgsl
#INSERT project_point.wgsl
#INSERT quad_corners.wgsl
#INSERT clip_test.wgsl

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) clip_distances: vec4f,
    @location(1) uv: vec2f,
    @location(2) point: vec3f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var out: VertexOutput;
    if (index >= VERTS_PER_QUAD) {
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    let corner = quad_corner(index);
    let point = read_vec3(corner, DATA_OFFSET_point);
    let projection = project_point(point);
    out.position = projection.position;
    out.clip_distances = projection.clip_distances;
    out.point = point;

    // Coordenadas UV normalizadas [-1.0, 1.0]
    var uvs = array<vec2f, 4>(
        vec2f(-1.0, 1.0),
        vec2f(-1.0, -1.0),
        vec2f(1.0, 1.0),
        vec2f(1.0, -1.0)
    );
    out.uv = uvs[corner];
    return out;
}

///// USER_WGSL_CODE /////
"""


class ShaderMobject(Mobject):
    def __init__(
        self,
        shader_code: str,
        width: float = FRAME_WIDTH,
        height: float = FRAME_HEIGHT,
        center: np.ndarray = ORIGIN,
        is_fixed_in_frame: bool = True,
        z_index: int = -10,  # Ubicar al fondo por defecto
        **kwargs,
    ):
        # 1. Llamar primero a super().__init__ para inicializar la estructura base
        super().__init__(z_index=z_index)

        self.verts_per_record = 2
        self.drawing_class = Drawing
        self.depth_test = False
        self.internal_time = 0.0
        self.dynamic_uniforms: dict[str, Callable[[float], Any]] = {}
        self.static_uniforms: dict[str, Any] = {}

        # 2. Desglosar uniformes de **kwargs
        custom_members = []
        for key, value in kwargs.items():
            size = infer_uniform_size(value)
            custom_members.append((key, size))
            if callable(value):
                self.dynamic_uniforms[key] = value
            else:
                self.static_uniforms[key] = sanitize_uniform_value(value, size)

        # 3. Construir y asignar el bloque std140 DEFINITIVO (sin riesgo de sobreescritura)
        self.uniform_dtype = uniform_block_dtype(
            *COMMON_UNIFORMS,
            *custom_members,
        )
        self.uniforms = Uniforms(self.uniform_dtype)

        # 4. Asignar parámetros comunes del motor
        self.uniforms["is_fixed_in_frame"] = 1.0 if is_fixed_in_frame else 0.0
        self.uniforms["shading"] = np.zeros(3, dtype=np.float32)
        for i in range(4):
            self.uniforms[f"clip_plane{i}"] = np.zeros(4, dtype=np.float32)

        # 5. Cargar valores iniciales de los uniformes
        for key, val in self.static_uniforms.items():
            self.uniforms[key] = val

        for key, func in self.dynamic_uniforms.items():
            size = infer_uniform_size(func)
            self.uniforms[key] = sanitize_uniform_value(func(0.0), size)

        # 6. Escribir el sombreador dentro del directorio oficial de shaders
        self.shader_file = self._write_shader_file(shader_code)

        # 7. Definir los 4 puntos del quad (UL, DL, UR, DR)
        half_w = width / 2.0
        half_h = height / 2.0
        cx, cy, cz = center
        corners = np.array([
            [cx - half_w, cy + half_h, cz],  # 0: UL
            [cx - half_w, cy - half_h, cz],  # 1: DL
            [cx + half_w, cy + half_h, cz],  # 2: UR
            [cx + half_w, cy - half_h, cz],  # 3: DR
        ], dtype=np.float32)
        self.set_points(corners)

        # 8. Activar updater temporal si existen uniformes dinámicos
        if self.dynamic_uniforms:
            self.add_updater(self._update_uniforms)

    def _write_shader_file(self, user_code: str) -> str:
        adapted_code = user_code.replace("uniforms.", "mob.")
        full_source = SHADER_MOBJECT_TEMPLATE.replace("///// USER_WGSL_CODE /////", adapted_code)
        
        filename = f"_custom_shader_{abs(id(self))}.wgsl"
        filepath = os.path.join(get_shader_dir(), filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_source)
        return filename

    def _update_uniforms(self, mob: ShaderMobject, dt: float) -> None:
        mob.internal_time += dt
        t = mob.internal_time
        for key, func in mob.dynamic_uniforms.items():
            size = infer_uniform_size(func)
            mob.uniforms[key] = sanitize_uniform_value(func(t), size)

    def set_uniform(self, key: str, value: Any) -> ShaderMobject:
        size = infer_uniform_size(value)
        self.uniforms[key] = sanitize_uniform_value(value, size)
        return self
