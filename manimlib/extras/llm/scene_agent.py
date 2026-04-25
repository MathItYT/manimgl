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
import threading

import numpy as np
import manimlib
import manimlib.constants
from openai import OpenAI
from pydantic import BaseModel, Field, create_model, ConfigDict, ValidationError

# ==============================================================================
# Modelos Pydantic Estrictos (Base Global)
# ==============================================================================

class BaseParams(BaseModel):
    model_config = ConfigDict(extra='forbid')

class MobjectParams(BaseParams):
    color: Optional[str] 
    opacity: Optional[str] 
    
class VMobjectParams(MobjectParams):
    fill_color: Optional[str] 
    fill_opacity: Optional[str] 
    stroke_color: Optional[str] 
    stroke_width: Optional[str]

class SurfaceParams(MobjectParams):
    u_min: str
    u_max: str
    u_samples: str
    v_min: str
    v_max: str
    v_samples: str
    
class AnimParams(BaseParams):
    run_time: Optional[str] 
    lag_ratio: Optional[str] 
    rate_func: Optional[str]

# ------------------------------------------------------------------------------
# Definición Estricta de Parámetros por Clase (Diccionarios Globales)
# ------------------------------------------------------------------------------

MOBJECT_DEFS = {
    "Group": (MobjectParams, {"mobjects": (Optional[List[Union[str, str]]], ...)}),
    "VGroup": (VMobjectParams, {"vmobjects": (Optional[List[Union[str, str]]], ...)}),
    "Circle": (VMobjectParams, {"radius": (Optional[str], ...), "arc_center": (Optional[str], ...)}),
    "Line": (VMobjectParams, {"start": (Optional[str], ...), "end": (Optional[str], ...)}),
    "Arrow": (VMobjectParams, {"start": (Optional[str], ...), "end": (Optional[str], ...)}),
    "Rectangle": (VMobjectParams, {"width": (Optional[str], ...), "height": (Optional[str], ...)}),
    "Polygon": (VMobjectParams, {"vertices": (Optional[List[str]], ...)}),
    "MarkdownMobject": (VMobjectParams, {"markdown": (str, ...), "font_size": (Optional[str], ...)}),
    "Brace": (VMobjectParams, {"mobject": (str, ...), "direction": (Optional[str], ...)}),
    "NumberPlane": (VMobjectParams, {
        "x_min": (Optional[str], ...), "x_max": (Optional[str], ...),
        "y_min": (Optional[str], ...), "y_max": (Optional[str], ...)
    }),
    "ThreeDAxes": (VMobjectParams, {
        "x_min": (Optional[str], ...), "x_max": (Optional[str], ...),
        "y_min": (Optional[str], ...), "y_max": (Optional[str], ...),
        "z_min": (Optional[str], ...), "z_max": (Optional[str], ...)
    }),
    "ParametricCurve": (VMobjectParams, {
        "t_func": (str, ...), 
        "t_min": (str, ...), "t_max": (str, ...), "t_samples": (str, ...)
    }),
    "ParametricSurface": (SurfaceParams, {"uv_func": (str, ...)}),
    "Sphere": (SurfaceParams, {"radius": (Optional[str], ...)}),
    "Cube": (SurfaceParams, {"side_length": (Optional[str], ...)}),
    "Cylinder": (SurfaceParams, {"height": (Optional[str], ...), "radius": (Optional[str], ...)}),
    "Cone": (SurfaceParams, {"height": (Optional[str], ...), "radius": (Optional[str], ...)}),
    "Torus": (SurfaceParams, {"r1": (Optional[str], ...), "r2": (Optional[str], ...)}),
}

ANIMATION_DEFS = {
    "ShowCreation": (AnimParams, {"mobject": (str, ...)}),
    "Write": (AnimParams, {"vmobject": (str, ...)}),
    "DrawBorderThenFill": (AnimParams, {"vmobject": (str, ...)}),
    "FadeIn": (AnimParams, {"mobject": (str, ...), "shift": (Optional[str], ...), "scale": (Optional[str], ...)}),
    "FadeOut": (AnimParams, {"mobject": (str, ...), "shift": (Optional[str], ...), "scale": (Optional[str], ...)}),
    "Transform": (AnimParams, {"mobject": (str, ...), "target_mobject": (str, ...)}),
    "ReplacementTransform": (AnimParams, {"mobject": (str, ...), "target_mobject": (str, ...)}),
    "FadeTransform": (AnimParams, {"mobject": (str, ...), "target_mobject": (str, ...)}),
}

METHOD_DEFS = {
    "add": {"mobjects": (List[str], ...)},
    "remove": {"mobjects": (List[str], ...)},
    "move_to": {"point_or_mobject": (str, ...), "aligned_edge": (Optional[str], ...)},
    "next_to": {"mobject_or_point": (str, ...), "direction": (Optional[str], ...), "buff": (Optional[str], ...)},
    "shift": {"vector": (str, ...)},
    "scale": {"scale_factor": (str, ...), "about_point": (Optional[str], ...)},
    "rotate": {"angle": (str, ...), "axis": (Optional[str], ...), "about_point": (Optional[str], ...)},
    "stretch": {"factor": (str, ...), "dim": (str, ...)},
    "set_width": {"width": (str, ...), "stretch": (Optional[bool], ...)},
    "set_height": {"height": (str, ...), "stretch": (Optional[bool], ...)},
    "set_color": {"color": (str, ...)},
    "set_fill": {"color": (Optional[str], ...), "opacity": (Optional[str], ...)},
    "set_stroke": {"color": (Optional[str], ...), "width": (Optional[str], ...), "opacity": (Optional[str], ...)},
    "set_style": {"fill_color": (Optional[str], ...), "fill_opacity": (Optional[str], ...), "stroke_color": (Optional[str], ...), "stroke_width": (Optional[str], ...)},
    "become": {"mobject": (str, ...)},
    "apply_function": {"function": (str, ...)},
    "arrange": {"direction": (Optional[str], ...), "buff": (Optional[str], ...), "center": (Optional[bool], ...)},
    "add_updater": {"update_function": (str, ...)},
    "remove_updater": {"update_function": (str, ...)}
}

# ==============================================================================
# Acciones Restantes Comunes (Globales)
# ==============================================================================

class PlayKwargs(BaseParams):
    run_time: Optional[str] 
    lag_ratio: Optional[str] 
    rate_func: Optional[str] 

class WaitAction(BaseParams):
    type: Literal["wait"]
    duration: str

class AddAction(BaseParams):
    type: Literal["add"]
    targets: List[str]

class RemoveAction(BaseParams):
    type: Literal["remove"]
    targets: List[str]
    unregister: bool


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
        self.result_queue = queue.Queue()
        self._is_processing_queue_item: bool = False
        self.execution_result_timeout_seconds: float = 90.0

        self._install_thread()

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
        """Reconstruye todos los modelos Pydantic dinámicos sin recursividad."""
        
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

        # Eliminados los grupos recursivos. ManimGL puede encadenar nativamente en kwargs.
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
            RemoveAction,
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

    def _install_thread(self) -> None:
        threading.Thread(target=self._process_queue, daemon=True).start()

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

        return {k: v for k, v in kwargs.items() if v is not None}

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
- All parameters (numbers, positions, colors, references) MUST be provided as strings.
- To reference an existing object, just write its name (e.g., `"my_circle"`).
- To evaluate math expressions or positions, write the code inside the string (e.g., `"np.pi / 2"`, `"obj1.get_center() + UP"`, `"np.array([1, 2, 0])"`). DO NOT output nested dictionaries like `{{"x": 1}}`.
- For literal text or color names, just write them normally (e.g., `"Hello World"`, `"RED"`).
- Adjust positions after creation so they don't overlap!
- If you need to remove a list of objects, but you will reuse them later, use the "remove" action with "unregister": false. If you won't reuse them, use "unregister": true to free up memory and unregister them from available objects.
- Don't create or save objects with the name of an existing object as it will cause conflicts. If a name already is taken, choose a different one or remove the old one first with unregistering being true. 
- All strings will be evaluated with a Python eval, so if you need to provide a literal string as a parameter, wrap it in a valid Python string (e.g., `"This is a literal string with special characters like \\"quotes\\" and newlines\\nThis is the second line"`).
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

                result = self.result_queue.get()
                if not result.get("success", False):
                    raise RuntimeError("Execution failed")
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
                        "content": f"El JSON generado no cumple el esquema estricto. Error de validación:\n{e}\nPor favor, corrige tu error y genera un JSON válido que cumpla estrictamente las reglas (incluyendo TODAS las llaves listadas, dejándolas en None si no las usas)."
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
        while True:
            try:
                self._is_processing_queue_item = True
                item = self.execution_queue.get()
                payload, mode = item
                
                snapshot = self._capture_execution_snapshot()
                def get_callback(func, param) -> None:
                    def callback():
                        self.scene.remove_main_loop_callback(callback)
                        traceback_str = None
                        try:
                            func(param)
                        except Exception:
                            self._restore_execution_snapshot(snapshot)
                            traceback.print_exc()
                            traceback_str = traceback.format_exc()
                        self.result_queue.put({"success": traceback_str is None, "traceback": traceback_str})
                    self.scene.add_main_loop_callback(callback)
                if mode == "actions":
                    get_callback(self._execute_actions_v2, payload.actions)
                else:
                    get_callback(self._execute_code, payload)
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
                if action.name in self.registered_objects:
                    raise ValueError(f"Object name '{action.name}' already exists. Don't use '{action.name}' and choose a different name")
                self.registered_objects[action.name] = obj

            elif hasattr(action, "type") and action.type == "play":
                anims = []
                for a in action.animations:
                    anim_dict = {"class_name": a.class_name, "params": self._coerce_refs(a.params)}
                    anims.append(self._resolve_anim(anim_dict))
                
                play_kwargs = self._coerce_refs(action.kwargs)
                self.scene.play(*anims, **play_kwargs)

            elif hasattr(action, "type") and action.type == "call":
                target = self._safe_eval_callable_expression(action.target)
                method = getattr(target, action.execute.method)
                
                coerced_params = self._coerce_refs(action.execute.params)
                coerced_params = self._format_kwargs(coerced_params)
                
                if action.execute.method in ("add", "add_to_back", "remove"):
                    result = method(*(mob for mob in coerced_params.pop("mobjects", []) if mob is not target))
                else:
                    result = method(**coerced_params)
                
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], manimlib.Animation):
                    self.scene.play(manimlib.Succession(*result))
                elif isinstance(result, manimlib.Animation):
                    self.scene.play(result)
                    
                if getattr(action, "save_as", None):
                    if action.save_as in self.registered_objects:
                        raise ValueError(f"Object name '{action.save_as}' already exists. Don't use '{action.save_as}' and choose a different name")
                    self.registered_objects[action.save_as] = result

            elif hasattr(action, "type") and action.type == "wait":
                self.scene.wait(action.duration)

            elif hasattr(action, "type") and action.type == "add":
                targets = [self.registered_objects[t] for t in action.targets]
                self.scene.add(*targets)

            elif hasattr(action, "type") and action.type == "remove":
                targets = [self.registered_objects[t] for t in action.targets]
                self.scene.remove(*targets)
                if action.unregister:
                    for t in action.targets:
                        self.registered_objects.pop(t, None)

    # --------------------------------------------------------------------------
    # Utilidades Seguras y Manejo de Entorno
    # --------------------------------------------------------------------------

    def _extract_code(self, text: str) -> Optional[str]:
        import re
        ticks = chr(96) * 3
        pattern = f"{ticks}(?:python|py)?\n(.*?)\n{ticks}"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip() if "scene." in text else None

    def _coerce_refs(self, value: Any, dynamic: bool = True) -> Any:
        if isinstance(value, str) and dynamic:
            # 1. Si es el nombre de un objeto registrado, devuélvelo
            if value in self.registered_objects:
                return self.registered_objects[value]
            try:
                evaluated = self._safe_eval_callable_expression(value)
                return evaluated
            except Exception:
                return value

        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return value
            
        if isinstance(value, BaseModel):
            return {k: self._coerce_refs(v) for k, v in value.model_dump(exclude_none=True).items()}
            
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                coerced_key = self._coerce_refs(k, dynamic=False)
                if coerced_key is None:
                    continue
                result[coerced_key] = self._coerce_refs(v)
            return result
            
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
        allowed_globals.update({k: v for k, v in manimlib.__dict__.items() if not k.startswith("_")})
        
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
        exec_globals.update({k: v for k, v in manimlib.constants.__dict__.items() if not k.startswith("_")})
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