import ast
import math
import random
import queue
import io
import contextlib
import traceback
import json
import typing
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import manimlib
import manimlib.constants
from openai import OpenAI
from pydantic import BaseModel, Field, create_model, ConfigDict, ValidationError

# ==============================================================================
# Modelos Pydantic Estrictos (Base Global)
# ==============================================================================

class Ref(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ref: str

class EvalStr(BaseModel):
    model_config = ConfigDict(extra='forbid')
    eval: str = Field(description="Consider that if it's a function you must put lambda in front of it, for example: lambda x: x**2")

class Vector3(BaseModel):
    model_config = ConfigDict(extra='forbid')
    x: float
    y: float
    z: float

RefOrVect = Union[Ref, Vector3, EvalStr]
DynNum = Union[float, EvalStr]
DynStr = Union[str, EvalStr]

class Kwarg(BaseModel):
    model_config = ConfigDict(extra='forbid')
    key: str
    value: DynStr

# Configuración Base estricta: NO se permiten parámetros inventados
class BaseParams(BaseModel):
    model_config = ConfigDict(extra='forbid')

class MobjectParams(BaseParams):
    color: Optional[DynStr] 
    opacity: Optional[DynNum] 
    z_index: Optional[DynNum] 
    # Shading separado
    reflectiveness: Optional[DynNum] 
    gloss: Optional[DynNum] 
    shadow: Optional[DynNum] 
    
class VMobjectParams(MobjectParams):
    fill_color: Optional[DynStr] 
    fill_opacity: Optional[DynNum] 
    stroke_color: Optional[DynStr] 
    stroke_opacity: Optional[DynNum] 
    stroke_width: Optional[DynNum] 
    
class SurfaceParams(MobjectParams):
    # Superficies separadas (u_range, v_range y resolution)
    u_min: Optional[DynNum] 
    u_max: Optional[DynNum] 
    u_samples: Optional[DynNum] 
    v_min: Optional[DynNum] 
    v_max: Optional[DynNum] 
    v_samples: Optional[DynNum] 
    
class AnimParams(BaseParams):
    run_time: Optional[DynNum] 
    lag_ratio: Optional[DynNum] 
    rate_func: Optional[DynStr] 

# ------------------------------------------------------------------------------
# Definición Estricta de Parámetros por Clase (Diccionarios Globales)
# ------------------------------------------------------------------------------

MOBJECT_DEFS = {
    "Mobject": (MobjectParams, {}),
    "Group": (MobjectParams, {"mobjects": (Optional[List[Union[Ref, EvalStr]]], ...)}),
    "Point": (MobjectParams, {"location": (Optional[RefOrVect], ...)}),
    "VMobject": (VMobjectParams, {}),
    "VGroup": (VMobjectParams, {"vmobjects": (Optional[List[Union[Ref, EvalStr]]], ...)}),
    "Circle": (VMobjectParams, {"radius": (Optional[DynNum], ...), "arc_center": (Optional[RefOrVect], ...)}),
    "Dot": (VMobjectParams, {"point": (Optional[RefOrVect], ...), "radius": (Optional[DynNum], ...)}),
    "Line": (VMobjectParams, {"start": (Optional[RefOrVect], ...), "end": (Optional[RefOrVect], ...)}),
    "DashedLine": (VMobjectParams, {"start": (Optional[RefOrVect], ...), "end": (Optional[RefOrVect], ...), "dash_length": (Optional[DynNum], ...)}),
    "Arrow": (VMobjectParams, {"start": (Optional[RefOrVect], ...), "end": (Optional[RefOrVect], ...)}),
    "Vector": (VMobjectParams, {"direction": (Optional[RefOrVect], ...)}),
    "Rectangle": (VMobjectParams, {"width": (Optional[DynNum], ...), "height": (Optional[DynNum], ...)}),
    "Square": (VMobjectParams, {"side_length": (Optional[DynNum], ...)}),
    "Polygon": (VMobjectParams, {"vertices": (Optional[List[RefOrVect]], ...)}),
    "RegularPolygon": (VMobjectParams, {"n": (Optional[DynNum], ...), "radius": (Optional[DynNum], ...)}),
    "Text": (VMobjectParams, {"text": (DynStr, ...), "font_size": (Optional[DynNum], ...)}),
    "MarkupText": (VMobjectParams, {"text": (DynStr, ...)}),
    "MarkdownMobject": (VMobjectParams, {"markdown": (DynStr, ...), "font_size": (Optional[DynNum], ...)}),
    "Brace": (VMobjectParams, {"mobject": (Ref, ...), "direction": (Optional[RefOrVect], ...)}),
    "Axes": (VMobjectParams, {
        "x_min": (Optional[DynNum], ...), "x_max": (Optional[DynNum], ...), "x_samples": (Optional[DynNum], ...),
        "y_min": (Optional[DynNum], ...), "y_max": (Optional[DynNum], ...), "y_samples": (Optional[DynNum], ...)
    }),
    "ThreeDAxes": (VMobjectParams, {
        "x_min": (Optional[DynNum], ...), "x_max": (Optional[DynNum], ...), "x_samples": (Optional[DynNum], ...),
        "y_min": (Optional[DynNum], ...), "y_max": (Optional[DynNum], ...), "y_samples": (Optional[DynNum], ...),
        "z_min": (Optional[DynNum], ...), "z_max": (Optional[DynNum], ...), "z_samples": (Optional[DynNum], ...)
    }),
    "NumberPlane": (VMobjectParams, {
        "x_min": (Optional[DynNum], ...), "x_max": (Optional[DynNum], ...), "x_samples": (Optional[DynNum], ...),
        "y_min": (Optional[DynNum], ...), "y_max": (Optional[DynNum], ...), "y_samples": (Optional[DynNum], ...)
    }),
    "FunctionGraph": (VMobjectParams, {
        "function": (EvalStr, ...), 
        "x_min": (Optional[DynNum], ...), "x_max": (Optional[DynNum], ...), "x_samples": (Optional[DynNum], ...)
    }),
    "ParametricCurve": (VMobjectParams, {
        "t_func": (EvalStr, ...), 
        "t_min": (Optional[DynNum], ...), "t_max": (Optional[DynNum], ...), "t_samples": (Optional[DynNum], ...)
    }),
    "Sphere": (SurfaceParams, {"radius": (Optional[DynNum], ...)}),
    "Cube": (SurfaceParams, {"side_length": (Optional[DynNum], ...)}),
    "Cylinder": (SurfaceParams, {"height": (Optional[DynNum], ...), "radius": (Optional[DynNum], ...)}),
    "Cone": (SurfaceParams, {"height": (Optional[DynNum], ...), "radius": (Optional[DynNum], ...)}),
    "Torus": (SurfaceParams, {"r1": (Optional[DynNum], ...), "r2": (Optional[DynNum], ...)}),
    "ParametricSurface": (SurfaceParams, {"uv_func": (EvalStr, ...)}),
    "VectorField": (VMobjectParams, {"func": (EvalStr, ...), "coordinate_system": (Optional[Union[Ref, EvalStr]], ...)}),
    "StreamLines": (VMobjectParams, {"func": (EvalStr, ...), "coordinate_system": (Optional[Union[Ref, EvalStr]], ...)}),
}

ANIMATION_DEFS = {
    "Animation": (AnimParams, {"mobject": (Ref, ...)}),
    "ShowCreation": (AnimParams, {"mobject": (Ref, ...)}),
    "Uncreate": (AnimParams, {"mobject": (Ref, ...)}),
    "Write": (AnimParams, {"vmobject": (Ref, ...)}),
    "DrawBorderThenFill": (AnimParams, {"vmobject": (Ref, ...)}),
    "FadeIn": (AnimParams, {"mobject": (Ref, ...), "shift": (Optional[RefOrVect], ...), "scale": (Optional[DynNum], ...)}),
    "FadeOut": (AnimParams, {"mobject": (Ref, ...), "shift": (Optional[RefOrVect], ...), "scale": (Optional[DynNum], ...)}),
    "FadeTransform": (AnimParams, {"mobject": (Ref, ...), "target_mobject": (Ref, ...)}),
    "Transform": (AnimParams, {"mobject": (Ref, ...), "target_mobject": (Ref, ...)}),
    "ReplacementTransform": (AnimParams, {"mobject": (Ref, ...), "target_mobject": (Ref, ...)}),
    "MoveAlongPath": (AnimParams, {"mobject": (Ref, ...), "path": (Ref, ...)}),
    "Indicate": (AnimParams, {"mobject": (Ref, ...), "scale_factor": (Optional[DynNum], ...)}),
    "FocusOn": (AnimParams, {"focus_point": (RefOrVect, ...)}),
    "WiggleOutThenIn": (AnimParams, {"mobject": (Ref, ...)}),
}

METHOD_DEFS = {
    "center": {}, "clear": {}, "clear_updaters": {}, "fix_in_frame": {}, 
    "generate_target": {}, "restore": {}, "save_state": {}, 
    "resume_updating": {}, "unfix_from_frame": {}, "reverse_points": {},
    "get_x": {}, "get_y": {}, "get_z": {}, "get_center": {}, "get_width": {}, 
    "get_height": {}, "get_depth": {}, "get_top": {}, "get_bottom": {}, "get_left": {}, 
    "get_right": {}, "get_bounding_box": {}, "get_continuous_bounding_box": {}, 
    "get_value": {}, "get_family": {}, "family_members_with_points": {}, 
    "get_axes": {}, "get_x_axis": {}, "get_y_axis": {}, "get_z_axis": {},
    "add": {"mobjects": (List[Ref], ...)},
    "add_to_back": {"mobjects": (List[Ref], ...)},
    "remove": {"mobjects": (List[Ref], ...)},
    "move_to": {"point_or_mobject": (RefOrVect, ...), "aligned_edge": (Optional[RefOrVect], ...)},
    "next_to": {"mobject_or_point": (RefOrVect, ...), "direction": (Optional[RefOrVect], ...), "buff": (Optional[DynNum], ...), "aligned_edge": (Optional[RefOrVect], ...)},
    "shift": {"vector": (RefOrVect, ...)},
    "scale": {"scale_factor": (DynNum, ...), "about_point": (Optional[RefOrVect], ...), "about_edge": (Optional[RefOrVect], ...)},
    "scale_about_point": {"scale_factor": (DynNum, ...), "point": (RefOrVect, ...)},
    "rotate": {"angle": (DynNum, ...), "axis": (Optional[RefOrVect], ...), "about_point": (Optional[RefOrVect], ...)},
    "rotate_about_origin": {"angle": (DynNum, ...), "axis": (Optional[RefOrVect], ...)},
    "flip": {"axis": (Optional[RefOrVect], ...)},
    "stretch": {"factor": (DynNum, ...), "dim": (DynNum, ...)},
    "stretch_to_fit_width": {"width": (DynNum, ...)},
    "stretch_to_fit_height": {"height": (DynNum, ...)},
    "stretch_to_fit_depth": {"depth": (DynNum, ...)},
    "set_width": {"width": (DynNum, ...), "stretch": (Optional[bool], ...)},
    "set_height": {"height": (DynNum, ...), "stretch": (Optional[bool], ...)},
    "set_depth": {"depth": (DynNum, ...), "stretch": (Optional[bool], ...)},
    "set_max_width": {"max_width": (DynNum, ...)},
    "set_max_height": {"max_height": (DynNum, ...)},
    "set_max_depth": {"max_depth": (DynNum, ...)},
    "align_to": {"mobject_or_point": (RefOrVect, ...), "direction": (Optional[RefOrVect], ...)},
    "to_corner": {"corner": (RefOrVect, ...), "buff": (Optional[DynNum], ...)},
    "to_edge": {"edge": (RefOrVect, ...), "buff": (Optional[DynNum], ...)},
    "align_on_border": {"direction": (RefOrVect, ...), "buff": (Optional[DynNum], ...)},
    "set_x": {"x": (DynNum, ...), "direction": (Optional[RefOrVect], ...)},
    "set_y": {"y": (DynNum, ...), "direction": (Optional[RefOrVect], ...)},
    "set_z": {"z": (DynNum, ...), "direction": (Optional[RefOrVect], ...)},
    "set_color": {"color": (DynStr, ...)},
    "set_opacity": {"opacity": (DynNum, ...)},
    "set_fill": {"color": (Optional[DynStr], ...), "opacity": (Optional[DynNum], ...)},
    "set_stroke": {"color": (Optional[DynStr], ...), "width": (Optional[DynNum], ...), "opacity": (Optional[DynNum], ...)},
    "set_style": {"fill_color": (Optional[DynStr], ...), "fill_opacity": (Optional[DynNum], ...), "stroke_color": (Optional[DynStr], ...), "stroke_width": (Optional[DynNum], ...), "stroke_opacity": (Optional[DynNum], ...)},
    "fade": {"darkness": (Optional[DynNum], ...)},
    "become": {"mobject": (Ref, ...)},
    "replace": {"mobject": (Ref, ...)},
    "replace_submobject": {"old_submob": (Ref, ...), "new_submob": (Ref, ...)},
    "match_color": {"mobject": (Ref, ...)},
    "match_style": {"mobject": (Ref, ...)},
    "match_width": {"mobject": (Ref, ...)},
    "match_height": {"mobject": (Ref, ...)},
    "match_depth": {"mobject": (Ref, ...)},
    "match_x": {"mobject": (Ref, ...)},
    "match_y": {"mobject": (Ref, ...)},
    "match_z": {"mobject": (Ref, ...)},
    "set_value": {"value": (DynNum, ...)},
    "increment_value": {"d_value": (DynNum, ...)},
    "set_theta": {"theta": (DynNum, ...)},
    "set_phi": {"phi": (DynNum, ...)},
    "set_gamma": {"gamma": (DynNum, ...)},
    "increment_theta": {"d_theta": (DynNum, ...)},
    "increment_phi": {"d_phi": (DynNum, ...)},
    "increment_gamma": {"d_gamma": (DynNum, ...)},
    "set_euler_angles": {"theta": (Optional[DynNum], ...), "phi": (Optional[DynNum], ...), "gamma": (Optional[DynNum], ...)},
    "set_field_of_view": {"fov": (DynNum, ...)},
    "set_points": {"points": (List[RefOrVect], ...)},
    "set_points_as_corners": {"points": (List[RefOrVect], ...)},
    "set_points_smoothly": {"points": (List[RefOrVect], ...)},
    "add_line_to": {"point": (RefOrVect, ...)},
    "add_smooth_curve_to": {"point": (RefOrVect, ...)},
    "add_quadratic_bezier_curve_to": {"handle": (RefOrVect, ...), "point": (RefOrVect, ...)},
    "add_cubic_bezier_curve_to": {"handle1": (RefOrVect, ...), "handle2": (RefOrVect, ...), "point": (RefOrVect, ...)},
    "put_start_and_end_on": {"start": (RefOrVect, ...), "end": (RefOrVect, ...)},
    "get_corner": {"direction": (RefOrVect, ...)},
    "get_edge_center": {"direction": (RefOrVect, ...)},
    "get_boundary_point": {"direction": (RefOrVect, ...)},
    "add_coordinate_labels": {"font_size": (Optional[DynNum], ...)},
    "add_x_labels": {"font_size": (Optional[DynNum], ...)},
    "add_y_labels": {"font_size": (Optional[DynNum], ...)},
    "c2p": {"x": (DynNum, ...), "y": (DynNum, ...), "z": (Optional[DynNum], ...)},
    "coords_to_point": {"x": (DynNum, ...), "y": (DynNum, ...), "z": (Optional[DynNum], ...)},
    "p2c": {"point": (RefOrVect, ...)},
    "point_to_coords": {"point": (RefOrVect, ...)},
    "get_implicit_curve": {"function": (EvalStr, ...), "color": (Optional[DynStr], ...)},
    "get_graph": {"function": (EvalStr, ...), "x_min": (Optional[DynNum], ...), "x_max": (Optional[DynNum], ...), "color": (Optional[DynStr], ...)},
    "get_parametric_curve": {"function": (EvalStr, ...), "t_min": (Optional[DynNum], ...), "t_max": (Optional[DynNum], ...), "color": (Optional[DynStr], ...)},
    "get_area_under_graph": {"graph": (Ref, ...), "x_min": (DynNum, ...), "x_max": (DynNum, ...), "color": (Optional[DynStr], ...)},
    "get_riemann_rectangles": {"graph": (Ref, ...), "x_min": (DynNum, ...), "x_max": (DynNum, ...), "dx": (Optional[DynNum], ...)},
    "get_scatterplot": {"x_values": (List[DynNum], ...), "y_values": (List[DynNum], ...), "color": (Optional[DynStr], ...)},
    "apply_matrix": {"matrix": (List[List[DynNum]], ...)},
    "apply_function": {"function": (EvalStr, ...)},
    "apply_complex_function": {"function": (EvalStr, ...)},
    "apply_function_to_position": {"function": (EvalStr, ...)},
    "apply_function_to_submobject_positions": {"function": (EvalStr, ...)},
    "arrange": {"direction": (Optional[RefOrVect], ...), "buff": (Optional[DynNum], ...), "center": (Optional[bool], ...)},
    "arrange_in_grid": {"n_rows": (Optional[DynNum], ...), "n_cols": (Optional[DynNum], ...), "buff": (Optional[DynNum], ...)},
    "surround": {"mobject": (Ref, ...), "dim_to_match": (Optional[DynNum], ...), "stretch": (Optional[bool], ...), "buff": (Optional[DynNum], ...)},
    "interpolate": {"mobject1": (Ref, ...), "mobject2": (Ref, ...), "alpha": (DynNum, ...)},
    "pointwise_become_partial": {"mobject": (Ref, ...), "a": (DynNum, ...), "b": (DynNum, ...)},
    "point_from_proportion": {"alpha": (DynNum, ...)},
    "proportion_from_point": {"point": (RefOrVect, ...)},
    "insert_n_curves": {"n": (DynNum, ...)},
    "reorient": {"vector": (RefOrVect, ...)},
    "space_out_submobjects": {"factor": (DynNum, ...)},
    "waggle": {"direction": (Optional[RefOrVect], ...), "max_theta": (Optional[DynNum], ...)},
    "add_updater": {"update_function": (EvalStr, ...)},
    "remove_updater": {"update_function": (EvalStr, ...)}
}

# ==============================================================================
# Acciones Restantes Comunes (Globales)
# ==============================================================================

class PlayKwargs(BaseParams):
    run_time: Optional[DynNum] 
    lag_ratio: Optional[DynNum] 
    rate_func: Optional[EvalStr] 

class WaitAction(BaseParams):
    type: Literal["wait"]
    duration: DynNum

class AddAction(BaseParams):
    type: Literal["add"]
    targets: List[str]

class RemoveAction(BaseParams):
    type: Literal["remove"]
    targets: List[str]


# ==============================================================================
# Controlador de la Escena (Main Controller) con Soporte Dinámico
# ==============================================================================

class LLMSceneController:
    """
    Controlador interactivo. Garantiza seguridad y esquemas estrictos gracias 
    a los modelos dinámicos de Pydantic definidos en tiempo de ejecución.
    """

    def __init__(
        self,
        scene: manimlib.InteractiveScene,
        client: OpenAI,
        model: str,
    ):
        self.scene = scene
        self.client = client
        self.model = model
        
        self.registered_objects: Dict[str, Any] = {}
        self._chat_histories: Dict[str, List[Dict[str, str]]] = {}
        self.max_history_messages: int = 40
        
        self.execution_queue = queue.Queue()
        self._is_processing_queue_item: bool = False
        self.execution_result_timeout_seconds: float = 90.0

        self._install_main_loop_queue_hook()

        # Copias locales de las definiciones para permitir extensiones sin mutar globales
        self.mobject_defs = dict(MOBJECT_DEFS)
        self.animation_defs = dict(ANIMATION_DEFS)
        self.method_defs = dict(METHOD_DEFS)

        # Builders Especiales
        self.mobject_builders = {
            "FunctionGraph": self._build_function_graph,
            "ParametricCurve": self._build_parametric_curve,
            "ParametricSurface": self._build_parametric_surface,
            "VectorField": self._build_vector_field,
            "StreamLines": self._build_stream_lines,
            "MarkdownMobject": self._build_markdown,
            "Group": self._build_group,
            "VGroup": self._build_vgroup,
            "Polygon": self._build_polygon,
        }
        self.animation_builders = {
            "AnimationGroup": self._build_animation_group,
            "Succession": self._build_succession,
            "LaggedStart": self._build_lagged_start,
        }

        # Generar los esquemas por primera vez al instanciar
        self._rebuild_schemas()

    # ==========================================================================
    # GENERACIÓN Y REGISTRO DINÁMICO DE ESQUEMAS
    # ==========================================================================

    def _rebuild_schemas(self):
        """Reconstruye todos los modelos Pydantic dinámicos basados en los diccionarios actuales."""
        
        def to_pascal(snake: str) -> str:
            return "".join(x.title() for x in snake.split("_"))

        # --- 1. CREATE ACTION ---
        create_target_models = []
        for name, (base_model, fields) in self.mobject_defs.items():
            ParamsModel = create_model(f"{name}Params", __base__=base_model, __config__=ConfigDict(extra='forbid'), **fields)
            TargetModel = create_model(
                f"Create{name}Target",
                class_name=(Literal[name], ...),
                params=(ParamsModel, ...),
                __config__=ConfigDict(extra='forbid')
            )
            create_target_models.append(TargetModel)

        AnyCreateTarget = typing.Union[tuple(create_target_models)]
        self.CreateAction = create_model(
            "CreateAction", 
            type=(Literal["create"], ...), 
            name=(str, ...), 
            target=(AnyCreateTarget, ...), 
            __base__=BaseParams
        )

        # --- 2. PLAY ACTION ---
        play_anim_models = []
        for name, (base_model, fields) in self.animation_defs.items():
            ParamsModel = create_model(f"{name}Params", __base__=base_model, __config__=ConfigDict(extra='forbid'), **fields)
            PlayAnimModel = create_model(
                f"Play{name}Anim",
                class_name=(Literal[name], ...),
                params=(ParamsModel, ...),
                __config__=ConfigDict(extra='forbid')
            )
            play_anim_models.append(PlayAnimModel)

        AnyPlayAnimBase = typing.Union[tuple(play_anim_models)]
        GroupParamsModel = create_model(
            "GroupParams", 
            animations=(List[AnyPlayAnimBase], ...), 
            __base__=AnimParams, 
            __config__=ConfigDict(extra='forbid')
        )

        for name in ["AnimationGroup", "Succession", "LaggedStart"]:
            PlayGroupModel = create_model(
                f"Play{name}Anim",
                class_name=(Literal[name], ...),
                params=(GroupParamsModel, ...),
                __config__=ConfigDict(extra='forbid')
            )
            play_anim_models.append(PlayGroupModel)

        AnyPlayAnim = typing.Union[tuple(play_anim_models)]
        self.PlayAction = create_model(
            "PlayAction", 
            type=(Literal["play"], ...), 
            animations=(List[AnyPlayAnim], ...), 
            kwargs=(PlayKwargs, ...), 
            __base__=BaseParams
        )

        # --- 3. CALL ACTION ---
        call_execute_models = []
        for name, fields in self.method_defs.items():
            ParamsModel = create_model(f"{to_pascal(name)}Params", __base__=BaseParams, __config__=ConfigDict(extra='forbid'), **fields)
            ExecuteModel = create_model(
                f"Call{to_pascal(name)}Execute",
                method=(Literal[name], ...),
                params=(ParamsModel, ...),
                __config__=ConfigDict(extra='forbid')
            )
            call_execute_models.append(ExecuteModel)

        AnyCallExecute = typing.Union[tuple(call_execute_models)]
        self.CallAction = create_model(
            "CallAction", 
            type=(Literal["call"], ...), 
            target=(str, ...), 
            execute=(AnyCallExecute, ...), 
            save_as=(Optional[str], ...), 
            __base__=BaseParams
        )

        # --- 4. MODELO FINAL (ActionResponse) ---
        AnyAction = typing.Union[
            self.CreateAction, 
            self.CallAction, 
            self.PlayAction, 
            WaitAction, 
            AddAction, 
            RemoveAction
        ]
        self.ActionResponse = create_model(
            "ActionResponse", 
            version=(Literal[2], ...), 
            actions=(List[AnyAction], ...), 
            __base__=BaseParams
        )

    def register_mobject_type(self, name: str, base_model: type[BaseModel], fields: dict, builder: Optional[typing.Callable] = None):
        """Registra un nuevo tipo de Mobject y actualiza el esquema."""
        self.mobject_defs[name] = (base_model, fields)
        if builder:
            self.mobject_builders[name] = builder
        self._rebuild_schemas()

    def register_animation_type(self, name: str, base_model: type[BaseModel], fields: dict, builder: Optional[typing.Callable] = None):
        """Registra una nueva animación y actualiza el esquema."""
        self.animation_defs[name] = (base_model, fields)
        if builder:
            self.animation_builders[name] = builder
        self._rebuild_schemas()

    def register_method_type(self, name: str, fields: dict):
        """Registra un nuevo método de Mobject y actualiza el esquema."""
        self.method_defs[name] = fields
        self._rebuild_schemas()

    def _install_main_loop_queue_hook(self) -> None:
        add_callback = getattr(self.scene, "add_main_loop_callback", None)
        if not callable(add_callback):
            raise RuntimeError("La escena no expone add_main_loop_callback")
        add_callback(self._process_queue)

    def register_object(self, name: str, obj: Any) -> None:
        self.registered_objects[name] = obj

    def clear_chat_history(self, response_mode: Optional[str] ) -> None:
        if response_mode is None:
            self._chat_histories.clear()
        else:
            self._chat_histories.pop(response_mode, None)

    # --------------------------------------------------------------------------
    # Preprocesamiento de Atributos Extraídos (Ranges, Shading, etc)
    # --------------------------------------------------------------------------

    def _format_kwargs(self, kwargs: dict) -> dict:
        """
        Reconstruye argumentos de manimlib a partir de los valores descompuestos
        por seguridad (shading, *_range, resolution).
        """
        if any(k in kwargs for k in ("reflectiveness", "gloss", "shadow")):
            refl = kwargs.pop("reflectiveness", 0.7)
            gloss = kwargs.pop("gloss", 0.1)
            shadow = kwargs.pop("shadow", 0.2)
            kwargs["shading"] = (refl, gloss, shadow)

        for prefix in ("x", "y", "z", "t"):
            min_key, max_key, samples_key = f"{prefix}_min", f"{prefix}_max", f"{prefix}_samples"
            if min_key in kwargs and max_key in kwargs:
                min_val = kwargs.pop(min_key)
                max_val = kwargs.pop(max_key)
                samples = kwargs.pop(samples_key, None)
                if samples is not None and samples > 0:
                    step = (max_val - min_val) / samples
                    kwargs[f"{prefix}_range"] = (min_val, max_val, step)
                else:
                    kwargs[f"{prefix}_range"] = (min_val, max_val)
            kwargs.pop(min_key, None)
            kwargs.pop(max_key, None)
            kwargs.pop(samples_key, None)

        res = []
        for prefix in ("u", "v"):
            min_key, max_key = f"{prefix}_min", f"{prefix}_max"
            if min_key in kwargs and max_key in kwargs:
                kwargs[f"{prefix}_range"] = (kwargs.pop(min_key), kwargs.pop(max_key))
            else:
                kwargs.pop(min_key, None)
                kwargs.pop(max_key, None)
                
            samples_key = f"{prefix}_samples"
            if samples_key in kwargs:
                res.append(kwargs.pop(samples_key))
            else:
                res.append(None)
        
        if res[0] is not None and res[1] is not None:
            kwargs["resolution"] = (res[0], res[1])

        if "texs_to_color_map" in kwargs and isinstance(kwargs["texs_to_color_map"], list):
            color_map = {}
            for item in kwargs["texs_to_color_map"]:
                k = item.get("key") if isinstance(item, dict) else getattr(item, "key", None)
                v = item.get("value") if isinstance(item, dict) else getattr(item, "value", None)
                if k is not None and v is not None:
                    color_map[k] = v
            kwargs["texs_to_color_map"] = color_map

        return kwargs

    # --------------------------------------------------------------------------
    # Builders Personalizados (Parseo seguro)
    # --------------------------------------------------------------------------

    def _build_function_graph(self, scene, kwargs: dict):
        func = kwargs.pop("function", "lambda x: x")
        if isinstance(func, str):
            func = self._safe_eval_callable_expression(func)
        return manimlib.FunctionGraph(func, **kwargs)

    def _build_parametric_curve(self, scene, kwargs: dict):
        func = kwargs.pop("t_func", "lambda t: np.array([t, 0, 0])")
        if isinstance(func, str):
            func = self._safe_eval_callable_expression(func)
        return manimlib.ParametricCurve(func, **kwargs)

    def _build_parametric_surface(self, scene, kwargs: dict):
        func = kwargs.pop("uv_func", "lambda u, v: np.array([u, v, 0])")
        if isinstance(func, str):
            func = self._safe_eval_callable_expression(func)
        return manimlib.ParametricSurface(func, **kwargs)
        
    def _build_vector_field(self, scene, kwargs: dict):
        func = kwargs.pop("func", "lambda p: p")
        if isinstance(func, str):
            func = self._safe_eval_callable_expression(func)
        return manimlib.VectorField(func, **kwargs)
        
    def _build_stream_lines(self, scene, kwargs: dict):
        func = kwargs.pop("func", "lambda p: p")
        if isinstance(func, str):
            func = self._safe_eval_callable_expression(func)
        return manimlib.StreamLines(func, **kwargs)
        
    def _build_markdown(self, scene, kwargs: dict):
        markdown_str = kwargs.pop("markdown", "")
        return manimlib.MarkdownMobject(markdown_str, **kwargs)

    def _build_group(self, scene, kwargs: dict):
        mobjects = kwargs.pop("mobjects", [])
        return manimlib.Group(*mobjects, **kwargs)

    def _build_vgroup(self, scene, kwargs: dict):
        vmobjects = kwargs.pop("vmobjects", [])
        return manimlib.VGroup(*vmobjects, **kwargs)

    def _build_polygon(self, scene, kwargs: dict):
        vertices = kwargs.pop("vertices", [])
        return manimlib.Polygon(*vertices, **kwargs)

    def _build_animation_group(self, scene, kwargs: dict):
        raw_anims = kwargs.pop("animations", [])
        anims = [self._resolve_anim(a) for a in raw_anims]
        return manimlib.AnimationGroup(*anims, **kwargs)

    def _build_succession(self, scene, kwargs: dict):
        raw_anims = kwargs.pop("animations", [])
        anims = [self._resolve_anim(a) for a in raw_anims]
        return manimlib.Succession(*anims, **kwargs)

    def _build_lagged_start(self, scene, kwargs: dict):
        raw_anims = kwargs.pop("animations", [])
        anims = [self._resolve_anim(a) for a in raw_anims]
        return manimlib.LaggedStart(*anims, **kwargs)

    def _resolve_anim(self, anim_dict: dict) -> manimlib.Animation:
        cls_name = anim_dict.get("class_name")
        anim_kwargs = anim_dict.get("params", {})
        anim_kwargs = self._format_kwargs(anim_kwargs)
        
        builder = self.animation_builders.get(cls_name)
        if builder:
            return builder(self.scene, anim_kwargs)
        else:
            return getattr(manimlib, cls_name)(**anim_kwargs)

    # --------------------------------------------------------------------------
    # Prompt y Ejecución (Sistema Principal)
    # --------------------------------------------------------------------------

    def run_prompt(
        self, 
        prompt: str, 
        additional_system_prompt: str | None , 
        response_mode: str = "code", 
        **kwargs
    ) -> bool:
        context_lines = [f"- {name} ({type(obj).__name__})" for name, obj in self.registered_objects.items()]
        context_str = "\n".join(context_lines) if context_lines else "(none)"

        history = self._chat_histories.setdefault(response_mode, [])
        
        if response_mode == "actions":
            schema = self.ActionResponse.model_json_schema()
            constants_str = "\n".join([f"- {name} = {value}" for name, value in manimlib.constants.__dict__.items() if not name.startswith("_")]) or "(none)"
            system_prompt = f"""
You control a ManimGL scene strictly.
Registered objects currently in memory:
{context_str}

Available constants in ManimGL:
{constants_str}

Return a valid JSON strictly following this schema. 
DO NOT INVENT PARAMETERS OR CLASSES. The schema enforces strict validation!

Important Guidelines:
- ALL parameters defined in the schema are absolutely REQUIRED. You MUST include every key in the `params` object, even if you leave the value as `null`.
- To reference an existing object, use `{{"ref": "object_name"}}`.
- To evaluate dynamic mathematical expressions, constants or methods, use `{{"eval": "expression"}}` (e.g., `{{"eval": "np.pi / 2"}}`, `{{"eval": "obj1.get_center() + UP"}}`).
- For creating Mobjects (`CreateAction`), the configuration is nested under `target`: `{{"type": "create", "name": "obj_name", "target": {{"class_name": "...", "params": {{...}}}}}}`.
- For method calls (`CallAction`), the configuration is nested under `execute`: `{{"type": "call", "target": "obj_name", "execute": {{"method": "...", "params": {{...}}}}}}`.
- For 3D points/vectors, use `{{"x": 1.0, "y": 2.0, "z": 3.0}}`.
- For `play` action `kwargs`, use an object like `{{"run_time": 1.0, "lag_ratio": null, "rate_func": null}}`.
- For shading, use `reflectiveness`, `gloss`, and `shadow` separately.
- For ranges (like x, y, t), use `x_min`, `x_max`, and `x_samples` (not `x_range`).
- Adjust positions after creation so they don't overlap!
"""
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "ManimJSON", "strict": True, "schema": schema}}
        else:
            system_prompt = f"""
You are an advanced AI controlling ManimGL via Python code.
Registered objects you can use globally:
{context_str}

Return ONLY a valid Python code block wrapped in ```python. 
Use `scene.play()`. Do NOT use `print()` or `import`.
"""
        if additional_system_prompt:
            system_prompt += f"\n\n**EXTRA RULES:**\n{additional_system_prompt}"

        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": prompt}]

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
                response_text = completion.choices[0].message.content or ""
                
                if response_mode == "actions":
                    data = json.loads(response_text)
                    parsed_response = self.ActionResponse.model_validate(data)
                    self.execution_queue.put((parsed_response, "actions"))
                else:
                    code_to_execute = self._extract_code(response_text)
                    if not code_to_execute:
                        raise ValueError("No valid Python code block found.")
                    self.execution_queue.put((code_to_execute, "code"))

                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": response_text})
                if len(history) > self.max_history_messages * 2:
                    self._chat_histories[response_mode] = history[-self.max_history_messages * 2:]

                return True

            except ValidationError as e:
                print(f"[Intento {attempt + 1}/{max_attempts}] Pydantic Validation Error:\n{e}")
                if attempt < max_attempts - 1:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "user", 
                        "content": f"El JSON generado no cumple el esquema estricto. Error de validación:\n{e}\nPor favor, corrige tu error y genera un JSON válido que cumpla estrictamente las reglas (incluyendo TODAS las llaves listadas, dejándolas en null si no las usas)."
                    })
                else:
                    return False
            except json.JSONDecodeError as e:
                print(f"[Intento {attempt + 1}/{max_attempts}] JSON Decode Error:\n{e}")
                if attempt < max_attempts - 1:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "user", 
                        "content": f"El output no es un JSON válido. Error:\n{e}\nPor favor, devuelve SOLO un JSON correctamente formateado."
                    })
                else:
                    return False
            except ValueError as e:
                print(f"[Intento {attempt + 1}/{max_attempts}] ValueError:\n{e}")
                if attempt < max_attempts - 1:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Error: {e}\nCorrige tu respuesta y asegúrate de proveer la estructura o bloque de código correcto."})
                else:
                    return False
            except Exception as e:
                print(f"[Intento {attempt + 1}/{max_attempts}] Fallo general al procesar prompt:\n{e}")
                if attempt == max_attempts - 1:
                    return False

        return False

    def _process_queue(self) -> None:
        if self._is_processing_queue_item:
            return

        try:
            self._is_processing_queue_item = True
            item = self.execution_queue.get_nowait()
            payload, mode = item
            
            snapshot = self._capture_execution_snapshot()
            
            try:
                if mode == "actions":
                    self._execute_actions_v2(payload.actions)
                else:
                    self._execute_code(payload)
            except Exception:
                self._restore_execution_snapshot(snapshot)
                traceback.print_exc()

        except queue.Empty:
            pass
        finally:
            self._is_processing_queue_item = False

    def _execute_actions_v2(self, actions: List[Any]):
        for action in actions:
            if hasattr(action, "type") and action.type == "create":
                cls_name = action.target.class_name
                coerced_params = self._coerce_refs(action.target.params)
                coerced_params = self._format_kwargs(coerced_params)
                
                builder = self.mobject_builders.get(cls_name)
                if builder:
                    obj = builder(self.scene, coerced_params)
                else:
                    ctor = getattr(manimlib, cls_name)
                    obj = ctor(**coerced_params)
                self.registered_objects[action.name] = obj

            elif hasattr(action, "type") and action.type == "play":
                anims = []
                for a in action.animations:
                    anim_dict = self._coerce_refs(a)
                    anims.append(self._resolve_anim(anim_dict))
                
                play_kwargs = self._coerce_refs(action.kwargs)
                self.scene.play(*anims, **play_kwargs)

            elif hasattr(action, "type") and action.type == "call":
                target = self.registered_objects[action.target]
                method = getattr(target, action.execute.method)
                
                coerced_params = self._coerce_refs(action.execute.params)
                coerced_params = self._format_kwargs(coerced_params)
                
                if action.execute.method in ("add", "add_to_back", "remove"):
                    result = method(*coerced_params.pop("mobjects", []))
                else:
                    result = method(**coerced_params)
                
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], manimlib.Animation):
                    self.scene.play(manimlib.Succession(*result))
                elif isinstance(result, manimlib.Animation):
                    self.scene.play(result)
                    
                if getattr(action, "save_as", None):
                    self.registered_objects[action.save_as] = result

            elif hasattr(action, "type") and action.type == "wait":
                self.scene.wait(action.duration)

            elif hasattr(action, "type") and action.type == "add":
                targets = [self.registered_objects[t] for t in action.targets]
                self.scene.add(*targets)

            elif hasattr(action, "type") and action.type == "remove":
                targets = [self.registered_objects[t] for t in action.targets]
                self.scene.remove(*targets)

    # --------------------------------------------------------------------------
    # Utilidades Seguras y Manejo de Entorno
    # --------------------------------------------------------------------------

    def _extract_code(self, text: str) -> Optional[str]:
        import re
        ticks = chr(96) * 3
        pattern = f"{ticks}(?:python|py)?\n(.*?)\n{ticks}"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip() if "scene." in text else None

    def _coerce_refs(self, value: Any) -> Any:
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return value
            
        if isinstance(value, BaseModel):
            if isinstance(value, Ref):
                if value.ref not in self.registered_objects:
                    raise KeyError(f"Reference not found: {value.ref}")
                return self.registered_objects[value.ref]
            if isinstance(value, EvalStr):
                return self._safe_eval_callable_expression(value.eval)
            if isinstance(value, Vector3):
                return np.array([value.x, value.y, value.z])
            
            return {k: self._coerce_refs(v) for k, v in value.model_dump(exclude_none=True).items()}
            
        if isinstance(value, dict):
            if "ref" in value and len(value) == 1:
                if value["ref"] not in self.registered_objects:
                    raise KeyError(f"Reference not found: {value['ref']}")
                return self.registered_objects[value["ref"]]
            if "eval" in value and len(value) == 1:
                return self._safe_eval_callable_expression(value["eval"])
            if {"x", "y", "z"}.issubset(value.keys()) and len(value) == 3:
                return np.array([value["x"], value["y"], value["z"]])
                
            return {k: self._coerce_refs(v) for k, v in value.items()}
            
        if isinstance(value, list):
            return [self._coerce_refs(v) for v in value]
            
        return value

    def _safe_eval_callable_expression(self, expression: str) -> Any:
        allowed_globals = {
            "__builtins__": {}, 
            "np": np, "math": math, "random": random, "linear": manimlib.linear,
            "smooth": manimlib.smooth, "PI": manimlib.PI, "TAU": manimlib.TAU,
            "ORIGIN": manimlib.ORIGIN, "UP": manimlib.UP, "DOWN": manimlib.DOWN,
            "LEFT": manimlib.LEFT, "RIGHT": manimlib.RIGHT, "IN": manimlib.IN, "OUT": manimlib.OUT,
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
            "float": float, "int": int, "complex": complex, "round": round,
        }
        allowed_globals.update({k: v for k, v in self.registered_objects.items() if not k.startswith("_")})
        
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Attribute, ast.Name)) and getattr(node, "attr", getattr(node, "id", "")).startswith("__"):
                raise ValueError("Dunder methods or attributes are not allowed.")
                
        return eval(compile(tree, "<llm_expr>", "eval"), allowed_globals)

    def _execute_code(self, code: str) -> None:
        exec_globals = {
            "scene": self.scene, "manimlib": manimlib, "np": np, "math": math,
            "print": lambda *args, **kwargs: None
        }
        exec_globals.update(self.registered_objects)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, exec_globals, exec_globals)

    def _capture_execution_snapshot(self) -> Dict[str, Any]:
        return {
            "scene_state": manimlib.SceneState(self.scene),
            "registered_objects": dict(self.registered_objects),
        }

    def _restore_execution_snapshot(self, snapshot: Dict[str, Any]) -> None:
        snapshot["scene_state"].restore_scene(self.scene)
        self.registered_objects = dict(snapshot["registered_objects"])