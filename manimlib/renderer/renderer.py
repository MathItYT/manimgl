from __future__ import annotations

import wgpu
import numpy as np

from manimlib.renderer.gpu import Gpu, RenderPass
from manimlib.renderer.material import Material
from manimlib.renderer.post_process import MobjectPostProcessor
from manimlib.utils.iterables import batch_by_comparison
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable
    from manimlib.mobject.mobject import Mobject
    from manimlib.renderer.drawing import Drawing
    from manimlib.renderer.shared_buffer import SharedBuffer


FRAMES_BEFORE_BUNDLING = 2


class Bundling(object):
    def __init__(self, allowed: bool = True):
        self.allowed = allowed
        self.bundle: Any = None
        self.settled = 0
        self.stale = True

    def invalidate(self) -> None:
        self.stale = True

    def take(self, make: Callable[[], Any]) -> Any:
        if self.stale:
            self.stale = False
            self.settled = 0
            self.bundle = None
        else:
            self.settled += 1
            if self.bundle is None and self.allowed and self.settled >= FRAMES_BEFORE_BUNDLING:
                self.bundle = make()
        return self.bundle


class Renderer(object):
    def __init__(self, gpu: Gpu, bundle: bool = True, together: bool = True):
        self.gpu = gpu
        self.camera: Any = None
        self.post_processor = MobjectPostProcessor(self.gpu)
        self.post_processor.renderer = self
        self.may_merge = together
        self.bundling = Bundling(allowed=bundle)
        self.materials: dict[tuple, Material] = dict()
        self.drawings: dict[Mobject, Drawing] = dict()
        self.drawn: list[Drawing] = []
        self.leaders: list[Drawing] = []
        self.run_lengths: tuple = ()
        self.samples = gpu.samples

    def get_vfx_group(self, mob: Mobject) -> Mobject | None:
        group = getattr(mob, "_vfx_group", None)
        if group is not None and getattr(group, "has_vfx", lambda: False)():
            return group
        local = getattr(mob, "_vfx_local", None)
        if local is not None and getattr(local, "has_vfx", lambda: False)():
            return local
        if getattr(mob, "has_vfx", lambda: False)():
            return mob
        return None

    def optimize_leaders_dag(self, leaders: list[Drawing]) -> list[Drawing]:
        """Reordena líderes mediante un DAG de oclusión para agrupar pases de VFX
        sin alterar el orden visual (respetando solapamientos 2D y z_index).
        """
        n = len(leaders)
        if n <= 1:
            return leaders

        # Solo aplicamos en 2D para no alterar la profundidad Back-to-Front en 3D
        frame = getattr(self.camera, "frame", None)
        if frame is not None:
            phi = getattr(frame, "get_phi", lambda: 0.0)()
            theta = getattr(frame, "get_theta", lambda: 0.0)()
            if abs(phi) > 1e-3 or abs(theta) > 1e-3:
                return leaders

        groups = [self.get_vfx_group(ldr.mobject) for ldr in leaders]
        unique_groups = {g for g in groups if g is not None}
        if not unique_groups:
            return leaders

        bboxes: list[tuple[np.ndarray, np.ndarray] | None] = []
        for ldr in leaders:
            mob = ldr.mobject
            if mob.has_points():
                bb = mob.get_bounding_box()
                r_extra = 0.0
                grp = self.get_vfx_group(mob)
                if grp is not None and hasattr(grp, "get_vfx_list"):
                    for eff in grp.get_vfx_list():
                        r_extra = max(r_extra, float(getattr(eff, "radius", 0.0)) * 0.015)
                bboxes.append((bb[0][:2] - r_extra, bb[2][:2] + r_extra))
            else:
                bboxes.append(None)

        in_degree = [0] * n
        adj: list[list[int]] = [[] for _ in range(n)]

        for i in range(n):
            bi = bboxes[i]
            zi = leaders[i].mobject.z_index
            for j in range(i + 1, n):
                zj = leaders[j].mobject.z_index
                bj = bboxes[j]
                overlaps = (
                    bi is not None and bj is not None
                    and not (bi[1][0] < bj[0][0] or bi[0][0] > bj[1][0]
                             or bi[1][1] < bj[0][1] or bi[0][1] > bj[1][1])
                )
                if (zj > zi) or (zj == zi and overlaps):
                    adj[i].append(j)
                    in_degree[j] += 1

        ready = [i for i in range(n) if in_degree[i] == 0]
        ordered: list[Drawing] = []
        current_group: Mobject | None = None

        while ready:
            candidate_idx: int | None = None
            if current_group is not None:
                for idx in ready:
                    if groups[idx] is current_group:
                        candidate_idx = idx
                        break

            if candidate_idx is None:
                candidate_idx = ready[0]
                current_group = groups[candidate_idx]

            ready.remove(candidate_idx)
            ordered.append(leaders[candidate_idx])

            for succ in adj[candidate_idx]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.append(succ)

        return ordered

    def render_mobject_to_view(
        self,
        encoder: Any,
        mobject: Mobject,
        view: Any,
        load_op: wgpu.LoadOp = wgpu.LoadOp.clear
    ) -> None:
        drawings = []
        for sm in mobject.get_family():
            drawing_class = sm.drawing_class
            if len(sm.data) == 0 or not drawing_class.draws(sm):
                continue
            drawing = self.drawings.get(sm)
            if drawing is None or drawing.replacements is not sm.shader_code_replacements:
                drawing = drawing_class(self.material_for(sm, drawing_class), sm)
                self.drawings[sm] = drawing
            drawing.write_uniforms()
            drawing.realize_textures()
            drawing.write_records()
            drawings.append(drawing)

        self.gpu.end_writes()

        if self.samples > 1:
            att = {
                "color_attachments": [{
                    "view": self.post_processor.msaa_mask_view,
                    "resolve_target": view,
                    "load_op": load_op,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                }],
                "depth_stencil_attachment": {
                    "view": self.post_processor.layer_depth_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": wgpu.LoadOp.clear,
                    "depth_store_op": wgpu.StoreOp.store,
                    "stencil_clear_value": 0,
                    "stencil_load_op": wgpu.LoadOp.clear,
                    "stencil_store_op": wgpu.StoreOp.store,
                },
            }
        else:
            att = {
                "color_attachments": [{
                    "view": view,
                    "load_op": load_op,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                }],
                "depth_stencil_attachment": {
                    "view": self.post_processor.layer_depth_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": wgpu.LoadOp.clear,
                    "depth_store_op": wgpu.StoreOp.store,
                    "stencil_clear_value": 0,
                    "stencil_load_op": wgpu.LoadOp.clear,
                    "stencil_store_op": wgpu.StoreOp.store,
                },
            }
        rp = RenderPass(encoder.begin_render_pass(**att), self.gpu.frame_bind_group)
        for drw in drawings:
            drw.draw(rp)
        rp.encoder.end()

    def draw(self, mobjects: Iterable[Mobject], attachments: dict) -> None:
        gpu = self.gpu
        drawings = self.resolve(mobjects)
        if drawings != self.drawn:
            self.bundling.invalidate()
        self.drawn = drawings

        rebinds = gpu.rebinds
        gpu.begin_writes()
        regroup = False
        for drawing in drawings:
            if drawing.write_uniforms():
                regroup = True
            drawing.realize_textures()
            if drawing.invalidated:
                self.bundling.invalidate()

        if self.bundling.stale or regroup:
            lengths = self.group(drawings)
            if lengths != self.run_lengths:
                self.bundling.invalidate()
            self.run_lengths = lengths

        for leader in self.leaders:
            leader.write_records()
            if leader.invalidated:
                self.bundling.invalidate()
        gpu.end_writes()

        if gpu.rebinds != rebinds or gpu.samples != self.samples:
            self.bundling.invalidate()
        self.samples = gpu.samples

        has_any_vfx = any(self.get_vfx_group(leader.mobject) is not None for leader in self.leaders)
        if not has_any_vfx:
            bundle = self.bundling.take(lambda: gpu.bundle(self.make_draws))
            with gpu.render_pass(attachments) as render_pass:
                if bundle is None:
                    self.make_draws(render_pass)
                else:
                    render_pass.replay(bundle)
            return

        self.bundling.invalidate()
        target_view = attachments["color_attachments"][0]["view"]
        color_tex = getattr(target_view, "texture", getattr(target_view, "_texture", None))
        width = int(color_tex.size[0])
        height = int(color_tex.size[1])
        self.post_processor.resize(width, height)
        self.post_processor.camera = self.camera

        # Avanza el tiempo continuo del pipeline en cada frame
        fps = getattr(self.camera, "fps", 30.0) if self.camera else 30.0
        self.post_processor.time += 1.0 / max(float(fps), 1.0)

        # -------------------------------------------------------------
        # 1. Agrupamiento contiguo por grupo de VFX optimizado por DAG
        # -------------------------------------------------------------
        ordered_leaders = self.optimize_leaders_dag(self.leaders)
        batches: list[tuple[list[Drawing], Mobject | None]] = []
        curr_batch: list[Drawing] = []
        curr_group: Mobject | None = None
        for leader in ordered_leaders:
            grp = self.get_vfx_group(leader.mobject)
            if not batches:
                curr_batch.append(leader)
                curr_group = grp
                batches.append((curr_batch, curr_group))
            elif grp is curr_group:
                curr_batch.append(leader)
            else:
                curr_batch = [leader]
                curr_group = grp
                batches.append((curr_batch, curr_group))

        # -------------------------------------------------------------
        # 2. UN SOLO COMMAND ENCODER PARA TODO EL FRAME
        # -------------------------------------------------------------
        encoder = gpu.device.create_command_encoder()
        first_draw = True
        clear_color = tuple(attachments["color_attachments"][0]["clear_value"])
        accum_view = self.post_processor.scene_view if self.samples > 1 else target_view
        scene_depth_view = attachments["depth_stencil_attachment"]["view"]

        for batch_leaders, group in batches:
            if group is None:
                # Lotes convencionales sin efectos de post-proceso
                if first_draw:
                    if self.samples > 1:
                        pass_att = {
                            "color_attachments": [{
                                "view": target_view,
                                "resolve_target": accum_view,
                                "load_op": wgpu.LoadOp.clear,
                                "store_op": wgpu.StoreOp.store,
                                "clear_value": clear_color,
                            }],
                            "depth_stencil_attachment": {
                                "view": scene_depth_view,
                                "depth_load_op": wgpu.LoadOp.clear,
                                "depth_store_op": wgpu.StoreOp.store,
                                "depth_clear_value": 1.0,
                                "stencil_load_op": wgpu.LoadOp.clear,
                                "stencil_store_op": wgpu.StoreOp.store,
                                "stencil_clear_value": 0,
                            },
                        }
                    else:
                        pass_att = {
                            "color_attachments": [{
                                "view": target_view,
                                "load_op": wgpu.LoadOp.clear,
                                "store_op": wgpu.StoreOp.store,
                                "clear_value": clear_color,
                            }],
                            "depth_stencil_attachment": {
                                "view": scene_depth_view,
                                "depth_load_op": wgpu.LoadOp.clear,
                                "depth_store_op": wgpu.StoreOp.store,
                                "depth_clear_value": 1.0,
                                "stencil_load_op": wgpu.LoadOp.clear,
                                "stencil_store_op": wgpu.StoreOp.store,
                                "stencil_clear_value": 0,
                            },
                        }
                    rp = RenderPass(encoder.begin_render_pass(**pass_att), gpu.frame_bind_group)
                    for ldr in batch_leaders:
                        ldr.draw(rp)
                    rp.encoder.end()
                    first_draw = False
                else:
                    if self.samples > 1:
                        layer_att = self.post_processor.get_layer_attachments(
                            depth_view=scene_depth_view,
                            clear_color=True,
                            depth_load_op=wgpu.LoadOp.load,
                        )
                        rp = RenderPass(encoder.begin_render_pass(**layer_att), gpu.frame_bind_group)
                        for ldr in batch_leaders:
                            ldr.draw(rp)
                        rp.encoder.end()
                        self.post_processor.blit(encoder, self.post_processor.layer_view, accum_view)
                    else:
                        pass_att = {
                            "color_attachments": [{
                                "view": target_view,
                                "load_op": wgpu.LoadOp.load,
                                "store_op": wgpu.StoreOp.store,
                                "clear_value": clear_color,
                            }],
                            "depth_stencil_attachment": {
                                "view": scene_depth_view,
                                "depth_load_op": wgpu.LoadOp.load,
                                "depth_store_op": wgpu.StoreOp.store,
                                "depth_clear_value": 1.0,
                                "stencil_load_op": wgpu.LoadOp.load,
                                "stencil_store_op": wgpu.StoreOp.store,
                                "stencil_clear_value": 0,
                            },
                        }
                        rp = RenderPass(encoder.begin_render_pass(**pass_att), gpu.frame_bind_group)
                        for ldr in batch_leaders:
                            ldr.draw(rp)
                        rp.encoder.end()
            else:
                # Lotes que poseen efectos visuales
                if first_draw:
                    if self.samples > 1:
                        clear_pass_msaa = encoder.begin_render_pass(
                            color_attachments=[{
                                "view": target_view,
                                "load_op": wgpu.LoadOp.clear,
                                "store_op": wgpu.StoreOp.store,
                                "clear_value": clear_color,
                            }],
                            depth_stencil_attachment={
                                "view": scene_depth_view,
                                "depth_clear_value": 1.0,
                                "depth_load_op": wgpu.LoadOp.clear,
                                "depth_store_op": wgpu.StoreOp.store,
                                "stencil_clear_value": 0,
                                "stencil_load_op": wgpu.LoadOp.clear,
                                "stencil_store_op": wgpu.StoreOp.store,
                            }
                        )
                        clear_pass_msaa.end()
                        clear_pass_accum = encoder.begin_render_pass(
                            color_attachments=[{
                                "view": accum_view,
                                "load_op": wgpu.LoadOp.clear,
                                "store_op": wgpu.StoreOp.store,
                                "clear_value": clear_color,
                            }]
                        )
                        clear_pass_accum.end()
                    else:
                        clear_pass = encoder.begin_render_pass(
                            color_attachments=[{
                                "view": target_view,
                                "load_op": wgpu.LoadOp.clear,
                                "store_op": wgpu.StoreOp.store,
                                "clear_value": clear_color,
                            }],
                            depth_stencil_attachment={
                                "view": scene_depth_view,
                                "depth_clear_value": 1.0,
                                "depth_load_op": wgpu.LoadOp.clear,
                                "depth_store_op": wgpu.StoreOp.store,
                                "stencil_clear_value": 0,
                                "stencil_load_op": wgpu.LoadOp.clear,
                                "stencil_store_op": wgpu.StoreOp.store,
                            }
                        )
                        clear_pass.end()
                    first_draw = False

                # Separar estáticos de mobjects locales dinámicos
                static_leaders: list[Drawing] = []
                moving_local_map: dict[Mobject, list[Drawing]] = dict()
                for ldr in batch_leaders:
                    local_mob = getattr(ldr.mobject, "_vfx_local", None)
                    if local_mob is not None and local_mob is not group and getattr(local_mob, "has_vfx", lambda: False)():
                        is_active = any(
                            getattr(e, "is_active", lambda p: True)(self.post_processor)
                            for e in local_mob.get_vfx_list()
                        )
                        if is_active:
                            if local_mob not in moving_local_map:
                                moving_local_map[local_mob] = []
                            moving_local_map[local_mob].append(ldr)
                            continue
                    static_leaders.append(ldr)

                # Renderizar la capa base del grupo
                layer_att = self.post_processor.get_layer_attachments(
                    depth_view=scene_depth_view,
                    clear_color=True,
                    depth_load_op=wgpu.LoadOp.load,
                )
                if static_leaders:
                    layer_rp = RenderPass(encoder.begin_render_pass(**layer_att), gpu.frame_bind_group)
                    for ldr in static_leaders:
                        ldr.draw(layer_rp)
                    layer_rp.encoder.end()
                else:
                    clear_pass = encoder.begin_render_pass(color_attachments=[{
                        "view": self.post_processor.layer_view,
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                        "clear_value": (0.0, 0.0, 0.0, 0.0),
                    }])
                    clear_pass.end()

                # Renderizar locales dinámicos en el buffer scratch
                for local_mob, local_ldrs in moving_local_map.items():
                    scratch_att = self.post_processor.get_scratch_attachments(
                        depth_view=scene_depth_view,
                        clear_color=True,
                        depth_load_op=wgpu.LoadOp.load,
                    )
                    scratch_rp = RenderPass(encoder.begin_render_pass(**scratch_att), gpu.frame_bind_group)
                    for ldr in local_ldrs:
                        ldr.draw(scratch_rp)
                    scratch_rp.encoder.end()

                    local_active_effects = sorted(
                        [eff for eff in local_mob.get_vfx_list() if getattr(eff, "is_active", lambda p: True)(self.post_processor)],
                        key=lambda e: getattr(e, "order", 50)
                    )
                    for eff in local_active_effects:
                        eff.apply(self.post_processor, encoder, self.post_processor.layer_view_b, self.post_processor.layer_view)

                # Snapshot de fondo condicional mediante `needs_bg`
                all_candidate_effects = list(getattr(group, "get_vfx_list", lambda: [])())
                for local_mob in moving_local_map:
                    all_candidate_effects.extend(getattr(local_mob, "get_vfx_list", lambda: [])())

                needs_bg_snapshot = any(
                    getattr(eff, "needs_bg", False)
                    and getattr(eff, "is_active", lambda p: True)(self.post_processor)
                    for eff in all_candidate_effects
                )
                if needs_bg_snapshot:
                    self.post_processor.copy_target_to_bg(encoder, accum_view)

                # Procesar la cadena de efectos del grupo y componer sobre accum_view
                self.post_processor.process_mobject(encoder, accum_view, group)

        # -------------------------------------------------------------
        # 3. Resolución final y envío a GPU
        # -------------------------------------------------------------
        if self.samples > 1:
            final_target = attachments["color_attachments"][0].get("resolve_target") or target_view
            self.post_processor.blit_fullscreen(encoder, accum_view, final_target)

        # Un solo buffer de comandos enviado a la cola del dispositivo
        gpu.queue.submit([encoder.finish()])

    def make_draws(self, render_pass: RenderPass) -> None:
        for leader in self.leaders:
            leader.draw(render_pass)

    def group(self, drawings: list[Drawing]) -> tuple:
        if not self.may_merge:
            self.leaders = list(drawings)
            return (1,) * len(drawings)
        self.compare_uniforms(drawings)
        runs = batch_by_comparison(
            drawings,
            lambda prev, drawing: (
                drawing.can_follow(prev)
                and self.get_vfx_group(prev.mobject) is self.get_vfx_group(drawing.mobject)
            )
        )
        self.leaders = []
        lengths = []
        for run in runs:
            run[0].members = run if len(run) > 1 else None
            self.leaders.append(run[0])
            lengths.append(len(run))
        return tuple(lengths)

    def compare_uniforms(self, drawings: list[Drawing]) -> None:
        matching: dict[SharedBuffer, list[bool]] = dict()
        for drawing in drawings:
            buffer = drawing.material.uniform_buffer
            claims = matching.get(buffer)
            if claims is None:
                claims = matching[buffer] = buffer.matching_claims()
            drawing.repeats_uniforms = claims[drawing.uniform_offset // buffer.window]

    def resolve(self, mobjects: Iterable[Mobject]) -> list[Drawing]:
        held = self.drawings
        self.drawings = dict()
        drawn = []
        for mobject in mobjects:
            for mob in mobject.get_family():
                if getattr(mob, "hidden", False):
                    continue
                drawing_class = mob.drawing_class
                if len(mob.data) == 0 or not drawing_class.draws(mob):
                    continue
                drawing = held.get(mob)
                material = self.material_for(mob, drawing_class)
                if (
                    drawing is None
                    or drawing.material is not material
                    or drawing.replacements is not mob.shader_code_replacements
                ):
                    drawing = drawing_class(material, mob)
                self.drawings[mob] = drawing
                drawn.append(drawing)

        # Si la cámara tiene rotación 3D activa, ordenamos según profundidad de cámara
        # para que la composición de capas no pinte objetos traseros sobre los delanteros
        frame = getattr(self.camera, "frame", None)
        phi = getattr(frame, "get_phi", lambda: 0.0)() if frame else 0.0
        theta = getattr(frame, "get_theta", lambda: 0.0)() if frame else 0.0

        if abs(phi) > 1e-3 or abs(theta) > 1e-3:
            cam_pos = frame.get_implied_camera_location() if frame else np.array([0, 0, 10])
            # Ordenar de más lejano a más cercano a la cámara (Back-to-Front)
            drawn.sort(key=lambda d: (
                d.mobject.z_index,
                -np.linalg.norm(d.mobject.get_center() - cam_pos)
            ))
        else:
            # En 2D se respeta estrictamente el z_index
            drawn.sort(key=lambda d: d.mobject.z_index)

        return drawn

    def material_for(self, mobject: Mobject, drawing_class: type) -> Material:
        key = drawing_class.key(mobject)
        if key not in self.materials:
            self.materials[key] = Material(
                self.gpu, mobject, drawing_class.module_specs(mobject),
            )
        return self.materials[key]