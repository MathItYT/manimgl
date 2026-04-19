"""Manual mobject builder registry for LLM actions v2.

This module registers mobject constructors one by one, with explicit JSON
schemas and builder wiring for the LLM actions mode.
"""

from __future__ import annotations

import ast
import math
import random
from functools import wraps
from typing import Any

import manimlib
import numpy as np

from manimlib.extras.llm.scene_agent import LLMSceneController


def _num(description: str | None = None) -> dict[str, Any]:
    schema = {"type": "number"}
    if description is not None:
        schema["description"] = description
    return schema


def _int(description: str | None = None) -> dict[str, Any]:
    schema = {"type": "integer"}
    if description is not None:
        schema["description"] = description
    return schema


def _bool(description: str | None = None) -> dict[str, Any]:
    schema = {"type": "boolean"}
    if description is not None:
        schema["description"] = description
    return schema


def _str(description: str | None = None) -> dict[str, Any]:
    schema = {"type": "string"}
    if description is not None:
        schema["description"] = description
    return schema


def _vector3() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": _num(),
            "y": _num(),
            "z": _num(),
        },
        "required": ["x", "y", "z"],
    }


def _shading_components() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reflectiveness": _num(),
            "gloss": _num(),
            "shadow": _num(),
        },
        "required": ["reflectiveness", "gloss", "shadow"],
    }


def _py_function_expr(
    signature_hint: str, example: str
) -> dict[str, Any]:
    return {
        "type": "string",
        "description": (
            "Python expression that evaluates to a callable "
            f"({signature_hint}). Use numpy as np and random. "
            f"Example: {example}"
        ),
    }


def _array_of(
    item_schema: dict[str, Any],
    *,
    min_items: int | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    # Keep min/max parameters in the helper signature for backwards
    # compatibility in call sites, but do not emit minItems/maxItems because
    # many OpenAI-compatible providers support only a smaller JSON-schema subset.
    _ = (min_items, max_items)
    schema: dict[str, Any] = {
        "type": "array",
        "items": item_schema,
    }
    return schema


def _dict_of(value_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": value_schema,
    }


def _any_of(*schemas: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": list(schemas)}


def _merge_props(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return merged


def _is_vector3_object(value: Any) -> bool:
    return isinstance(value, dict) and set(value.keys()) == {
        "x",
        "y",
        "z",
    }


def _is_shading_object(value: Any) -> bool:
    return isinstance(value, dict) and set(value.keys()) == {
        "reflectiveness",
        "gloss",
        "shadow",
    }


def _normalize_tuple_like(value: Any) -> Any:
    if _is_vector3_object(value):
        return np.array(
            [value["x"], value["y"], value["z"]], dtype=float
        )

    if _is_shading_object(value):
        return (
            float(value["reflectiveness"]),
            float(value["gloss"]),
            float(value["shadow"]),
        )

    if isinstance(value, list):
        return [_normalize_tuple_like(v) for v in value]

    if isinstance(value, tuple):
        return tuple(_normalize_tuple_like(v) for v in value)

    if isinstance(value, dict):
        return {k: _normalize_tuple_like(v) for k, v in value.items()}

    return value


def _wrap_builder_for_tuples(builder: Any) -> Any:
    @wraps(builder)
    def _wrapped(*args, **kwargs):
        norm_args = [_normalize_tuple_like(arg) for arg in args]
        norm_kwargs = {
            k: _normalize_tuple_like(v) for k, v in kwargs.items()
        }
        return builder(*norm_args, **norm_kwargs)

    return _wrapped


def _safe_eval_globals(
    controller: LLMSceneController,
) -> dict[str, Any]:
    allowed: dict[str, Any] = {
        "np": np,
        "random": random,
        "math": math,
        "linear": manimlib.linear,
        "smooth": manimlib.smooth,
        "double_smooth": manimlib.double_smooth,
        "there_and_back": manimlib.there_and_back,
        "wiggle": manimlib.wiggle,
        "PI": manimlib.PI,
        "TAU": manimlib.TAU,
        "DEG": manimlib.DEG,
        "ORIGIN": manimlib.ORIGIN,
        "UP": manimlib.UP,
        "DOWN": manimlib.DOWN,
        "LEFT": manimlib.LEFT,
        "RIGHT": manimlib.RIGHT,
        "IN": manimlib.IN,
        "OUT": manimlib.OUT,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "float": float,
        "int": int,
        "complex": complex,
        "round": round,
    }
    for name, obj in controller.registered_objects.items():
        if name.isidentifier() and not name.startswith("_"):
            allowed[name] = obj
    return allowed


def _validate_safe_expression(
    tree: ast.AST,
    *,
    allowed_names: set[str],
) -> None:
    local_names = {
        node.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.arg)
    }
    allowed_nodes = (
        ast.Expression,
        ast.Lambda,
        ast.arguments,
        ast.arg,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Dict,
        ast.keyword,
        ast.Call,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.IfExp,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(
                f"Unsafe expression node: {type(node).__name__}"
            )

        if isinstance(node, ast.Attribute) and node.attr.startswith(
            "__"
        ):
            raise ValueError("Dunder attribute access is not allowed")

        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise ValueError("Dunder names are not allowed")
            if (
                isinstance(node.ctx, ast.Load)
                and node.id not in allowed_names
                and node.id not in local_names
            ):
                raise ValueError(
                    f"Name '{node.id}' is not allowed in function expression"
                )


def _safe_eval_callable_expression(
    expression: str,
    controller: LLMSceneController,
) -> Any:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError(
            "Function expression must be a non-empty string"
        )

    allowed_globals = _safe_eval_globals(controller)
    tree = ast.parse(expression, mode="eval")
    _validate_safe_expression(
        tree, allowed_names=set(allowed_globals.keys())
    )
    compiled = compile(tree, "<llm_function_expr>", "eval")
    result = eval(
        compiled,
        {"__builtins__": {}, **allowed_globals},
        {},
    )

    if not callable(result):
        raise TypeError("Expression must evaluate to a callable")
    return result


def _make_function_expression_builder(
    class_name: str,
    *function_fields: str,
):
    def _builder(scene, controller, data=None, **kwargs):
        ctor = getattr(manimlib, class_name, None)
        if ctor is None or not callable(ctor):
            raise ValueError(f"Unknown mobject class: {class_name}")

        params = dict(kwargs)
        raw_params = data if isinstance(data, dict) else {}
        for field in function_fields:
            raw_value = raw_params.get(field, params.get(field))
            if isinstance(raw_value, str):
                params[field] = _safe_eval_callable_expression(
                    raw_value,
                    controller,
                )
        return ctor(**params)

    return _builder


_FUNCTION_FIELD_NAMES = {
    "function",
    "func",
    "homotopy",
    "complex_homotopy",
    "update_function",
    "number_update_func",
    "on_click",
}


def _is_function_field_name(name: str) -> bool:
    return name in _FUNCTION_FIELD_NAMES or name.endswith("_func")


def _wrap_builder_for_function_fields(
    builder: Any,
    controller: LLMSceneController,
    function_fields: list[str],
) -> Any:
    if not function_fields:
        return builder

    field_set = tuple(function_fields)

    @wraps(builder)
    def _wrapped(*args, **kwargs):
        mapped = dict(kwargs)
        for field in field_set:
            value = mapped.get(field)
            if isinstance(value, str):
                mapped[field] = _safe_eval_callable_expression(
                    value,
                    controller,
                )
        return builder(*args, **mapped)

    return _wrapped


def _register(
    controller: LLMSceneController,
    name: str,
    *,
    properties: dict[str, Any],
    required: list[str] | None = None,
    positional: list[str] | None = None,
    spread: list[str] | None = None,
    builder: Any = None,
) -> None:
    ctor = (
        builder
        if builder is not None
        else getattr(manimlib, name, None)
    )
    if ctor is None or not callable(ctor):
        return
    function_fields = [
        key
        for key in properties.keys()
        if _is_function_field_name(key)
    ]
    ctor = _wrap_builder_for_function_fields(
        ctor,
        controller,
        function_fields,
    )
    ctor = _wrap_builder_for_tuples(ctor)
    controller.register_mobject_builder(
        name,
        ctor,
        properties=properties,
        required=required,
        positional=positional,
        spread=spread,
        allow_extra=False,
    )


def _register_animation(
    controller: LLMSceneController,
    name: str,
    *,
    properties: dict[str, Any],
    required: list[str] | None = None,
    positional: list[str] | None = None,
    spread: list[str] | None = None,
    builder: Any = None,
) -> None:
    ctor = (
        builder
        if builder is not None
        else getattr(manimlib, name, None)
    )
    if ctor is None or not callable(ctor):
        return
    function_fields = [
        key
        for key in properties.keys()
        if _is_function_field_name(key)
    ]
    ctor = _wrap_builder_for_function_fields(
        ctor,
        controller,
        function_fields,
    )
    ctor = _wrap_builder_for_tuples(ctor)
    controller.register_animation_builder(
        name,
        ctor,
        properties=properties,
        required=required,
        positional=positional,
        spread=spread,
        allow_extra=False,
    )


def _anim_spec_schema(value_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "class": _str(),
            "params": {
                "type": "object",
                "additionalProperties": value_schema,
            },
        },
        "required": ["class", "params"],
    }


def _build_animation_from_spec(
    controller: LLMSceneController,
    spec: Any,
):
    if isinstance(spec, manimlib.Animation):
        return spec
    if not isinstance(spec, dict):
        raise TypeError("Animation spec must be an object")

    cls_name = spec.get("class")
    if not isinstance(cls_name, str) or not cls_name:
        raise ValueError(
            "Animation spec requires a non-empty 'class'"
        )

    raw_params = spec.get("params", {})
    if not isinstance(raw_params, dict):
        raise TypeError("Animation spec 'params' must be an object")

    schema = controller._animation_schemas.get(cls_name)
    if schema is None:
        raise KeyError(f"Unregistered animation class: {cls_name}")

    args, kwargs = controller._args_kwargs_from_params(
        raw_params, schema
    )
    coerced_params = {
        k: controller._coerce_value(v) for k, v in raw_params.items()
    }
    builder = controller._animation_builders.get(cls_name)
    if builder is None:
        builder = controller._resolve_manimlib_callable(cls_name)

    anim = controller._call_builder(
        builder,
        args,
        kwargs,
        params=coerced_params,
        raw_params=raw_params,
    )
    if not isinstance(anim, manimlib.Animation):
        raise TypeError(
            f"play '{cls_name}' did not return an Animation"
        )
    return anim


def _build_animation_group(
    scene,
    controller,
    animations,
    run_time: float = -1.0,
    lag_ratio: float = 0.0,
    group=None,
    group_type=None,
    **kwargs,
):
    built_anims = [
        _build_animation_from_spec(controller, spec)
        for spec in animations
    ]
    return manimlib.AnimationGroup(
        *built_anims,
        run_time=run_time,
        lag_ratio=lag_ratio,
        group=group,
        group_type=group_type,
        **kwargs,
    )


def _build_succession(
    scene,
    controller,
    animations,
    lag_ratio: float = 1.0,
    **kwargs,
):
    built_anims = [
        _build_animation_from_spec(controller, spec)
        for spec in animations
    ]
    return manimlib.Succession(
        *built_anims,
        lag_ratio=lag_ratio,
        **kwargs,
    )


def _build_lagged_start(
    scene,
    controller,
    animations,
    lag_ratio: float = 0.05,
    **kwargs,
):
    built_anims = [
        _build_animation_from_spec(controller, spec)
        for spec in animations
    ]
    return manimlib.LaggedStart(
        *built_anims,
        lag_ratio=lag_ratio,
        **kwargs,
    )


def _build_lagged_start_map(
    scene,
    anim_class,
    group,
    anim_params=None,
    run_time: float = 2.0,
    lag_ratio: float = 0.05,
    **kwargs,
):
    anim_ctor = getattr(manimlib, anim_class, None)
    if anim_ctor is None or not callable(anim_ctor):
        raise ValueError(f"Unknown animation class: {anim_class}")

    item_params = dict(anim_params or {})

    def anim_func(submob):
        return anim_ctor(submob, **item_params)

    return manimlib.LaggedStartMap(
        anim_func,
        group,
        run_time=run_time,
        lag_ratio=lag_ratio,
        **kwargs,
    )


def _build_apply_method(
    scene,
    mobject,
    method,
    args=None,
    method_kwargs=None,
    **kwargs,
):
    method_obj = getattr(mobject, method, None)
    if method_obj is None or not callable(method_obj):
        raise ValueError(
            f"Object does not have callable method '{method}'"
        )
    call_args = list(args or [])
    if method_kwargs:
        call_args.append(method_kwargs)
    return manimlib.ApplyMethod(method_obj, *call_args, **kwargs)


def _build_animate_methods(
    scene,
    controller,
    mobject,
    methods,
    data=None,
    **animate_kwargs,
):
    """Build a chained mobject.animate(...).method1(...).method2(...) animation.

    Every chained method must be registered in the controller method catalog
    for the target mobject, and `params` must match the registered schema.
    """
    if not isinstance(mobject, manimlib.Mobject):
        raise TypeError(
            "AnimateMethods requires a mobject in 'mobject'"
        )

    raw_methods = None
    if isinstance(data, dict):
        raw_methods = data.get("methods")
    if raw_methods is None:
        raw_methods = methods

    if not isinstance(raw_methods, list) or not raw_methods:
        raise ValueError(
            "AnimateMethods requires a non-empty 'methods' list"
        )

    clean_anim_kwargs = {
        k: v for k, v in animate_kwargs.items() if v is not None
    }

    animate_builder = mobject.animate
    if clean_anim_kwargs:
        animate_builder = animate_builder(**clean_anim_kwargs)

    for i, method_spec in enumerate(raw_methods):
        if not isinstance(method_spec, dict):
            raise ValueError(
                f"AnimateMethods step #{i} must be an object"
            )

        method_name = method_spec.get("method")
        if (
            not isinstance(method_name, str)
            or not method_name
            or method_name.startswith("_")
        ):
            raise ValueError(
                f"AnimateMethods step #{i} requires a valid non-private 'method'"
            )

        method_obj = getattr(mobject, method_name, None)
        if method_obj is None or not callable(method_obj):
            raise AttributeError(
                f"Target '{type(mobject).__name__}' has no method '{method_name}'"
            )

        method_schema = controller._resolve_registered_method_schema(
            mobject,
            method_name,
        )
        if method_schema is None:
            if controller._has_registered_method_catalog(mobject):
                raise ValueError(
                    f"Method '{type(mobject).__name__}.{method_name}' is not in the registered method catalog"
                )
            raise ValueError(
                f"No registered method catalog found for '{type(mobject).__name__}'. "
                "Call init_all_mobject_methods(...) before using AnimateMethods."
            )

        raw_params = method_spec.get("params")
        if not isinstance(raw_params, dict):
            raise ValueError(
                f"AnimateMethods step #{i} requires 'params' object"
            )

        controller._validate_params_against_registered_schema(
            raw_params,
            method_schema,
            where=f"AnimateMethods.{type(mobject).__name__}.{method_name}",
        )

        typed_params = controller._coerce_typed_call_params(
            raw_params,
            target=mobject,
            method=method_obj,
            method_schema=method_schema,
        )
        args, kwargs = controller._args_kwargs_from_params(
            typed_params,
            method_schema,
        )
        args, kwargs = controller._coerce_call_callable_args(
            method_name,
            mobject,
            method_obj,
            args,
            kwargs,
            method_schema=method_schema,
        )

        animate_builder = getattr(animate_builder, method_name)(
            *args, **kwargs
        )

    return animate_builder.build()


def _build_apply_matrix(
    scene,
    mobject,
    matrix=None,
    data=None,
    **kwargs,
):
    raw_matrix = matrix
    if isinstance(data, dict) and "matrix" in data:
        raw_matrix = data["matrix"]
    if raw_matrix is None:
        raise ValueError("ApplyMatrix requires 'matrix'")
    return manimlib.ApplyMatrix(raw_matrix, mobject, **kwargs)


def _build_video_mobject(
    scene,
    iterator=None,
    video_path_or_camera=None,
    flip_horizontal: bool = False,
    auto_play: bool = False,
    **kwargs,
):
    if iterator is not None:
        mob = manimlib.VideoMobject(iterator=iterator, **kwargs)
    elif video_path_or_camera is not None:
        mob = manimlib.VideoMobject.from_video(
            video_path_or_camera=video_path_or_camera,
            flip_horizontal=flip_horizontal,
            **kwargs,
        )
    else:
        raise ValueError(
            "VideoMobject requires either 'iterator' or "
            "'video_path_or_camera'."
        )

    if auto_play:
        mob.play()
    return mob


def _build_magnifying_glass(
    scene,
    **kwargs,
):
    return manimlib.MagnifyingGlass(scene=scene, **kwargs)


def _build_mask_mobject(
    scene,
    src_mobject,
    mask_mobject,
    **kwargs,
):
    return manimlib.MaskMobject(
        scene=scene,
        src_mobject=src_mobject,
        mask_mobject=mask_mobject,
        **kwargs,
    )


def _resolve_coordinate_system_method(
    coordinate_system,
    method_name: str,
):
    if coordinate_system is None:
        raise ValueError("coordinate_system is required")
    method = getattr(coordinate_system, method_name, None)
    if method is None or not callable(method):
        raise ValueError(
            "Object does not have callable coordinate-system "
            f"method '{method_name}'"
        )
    return method


def _require_three_d_axes(coordinate_system):
    three_d_axes_cls = getattr(manimlib, "ThreeDAxes", None)
    if three_d_axes_cls is None or not isinstance(
        coordinate_system, three_d_axes_cls
    ):
        raise TypeError(
            "This builder requires a ThreeDAxes object in "
            "'coordinate_system'."
        )
    return coordinate_system

def _build_function_graph(
    scene,
    function,
    x_min: float | None = None,
    x_max: float | None = None,
    x_samples: int | None = None,
    **kwargs,
):
    x_range = None
    if x_min is not None and x_max is not None and x_samples is None:
        x_range = (x_min, x_max)
    elif x_min is not None and x_max is not None and x_samples is not None:
        x_range = (x_min, x_max, (x_max - x_min) / x_samples)
    return manimlib.FunctionGraph(function, x_range=x_range, **kwargs)

def _build_implicit_function(
    scene,
    function,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    **kwargs,
):
    x_range = (x_min, x_max) if x_min is not None and x_max is not None else None
    y_range = (y_min, y_max) if y_min is not None and y_max is not None else None
    return manimlib.ImplicitFunction(
        function,
        x_range=x_range,
        y_range=y_range,
        **kwargs,
    )

def _build_coordinate_system_graph(
    scene,
    coordinate_system,
    function,
    x_min: float | None = None,
    x_max: float | None = None,
    x_samples: int | None = None,
    bind: bool = False,
    **kwargs,
):
    three_d_axes_cls = getattr(manimlib, "ThreeDAxes", None)
    if three_d_axes_cls is not None and isinstance(
        coordinate_system, three_d_axes_cls
    ):
        raise TypeError(
            "CoordinateSystemGraph is for 2D systems. "
            "Use ThreeDAxesGraphSurface for 3D surfaces."
        )
    method = _resolve_coordinate_system_method(
        coordinate_system, "get_graph"
    )
    x_range = None
    if x_min is not None and x_max is not None and x_samples is None:
        x_range = (x_min, x_max)
    elif x_min is not None and x_max is not None and x_samples is not None:
        x_range = (x_min, x_max, (x_max - x_min) / x_samples)
    return method(function, x_range=x_range, bind=bind, **kwargs)

def _build_coordinate_system_parametric_curve(
    scene,
    coordinate_system,
    function,
    t_min: float | None = None,
    t_max: float | None = None,
    t_samples: int | None = None,
    **kwargs,
):
    method = _resolve_coordinate_system_method(
        coordinate_system, "get_parametric_curve"
    )
    t_range = None
    if t_min is not None and t_max is not None and t_samples is None:
        t_range = (t_min, t_max)
    elif t_min is not None and t_max is not None and t_samples is not None:
        t_range = (t_min, t_max, (t_max - t_min) / t_samples)
    return method(function, t_range=t_range, **kwargs)


def _build_coordinate_system_scatterplot(
    scene,
    coordinate_system,
    x_values,
    y_values,
    **kwargs,
):
    method = _resolve_coordinate_system_method(
        coordinate_system, "get_scatterplot"
    )
    return method(x_values, y_values, **kwargs)


def _build_coordinate_system_tangent_line(
    scene,
    coordinate_system,
    x,
    graph,
    length: float = 5.0,
    **kwargs,
):
    method = _resolve_coordinate_system_method(
        coordinate_system, "get_tangent_line"
    )
    return method(x, graph, length=length, **kwargs)


def _build_coordinate_system_riemann_rectangles(
    scene,
    coordinate_system,
    graph,
    x_min: float | None = None,
    x_max: float | None = None,
    x_samples: int | None = None,
    input_sample_type: str = "left",
    **kwargs,
):
    method = _resolve_coordinate_system_method(
        coordinate_system, "get_riemann_rectangles"
    )
    x_range = None
    dx = None
    if x_min is not None and x_max is not None:
        x_range = (x_min, x_max)
        if x_samples is not None:
            dx = (x_max - x_min) / x_samples
    return method(
        graph,
        x_range=x_range,
        dx=dx,
        input_sample_type=input_sample_type,
        **kwargs,
    )


def _build_number_line(
    scene,
    x_min: float | None = None,
    x_max: float | None = None,
    x_tick_step: float | None = None,
    **kwargs,
):
    x_range = None
    if x_min is not None and x_max is not None and x_tick_step is None:
        x_range = (x_min, x_max)
    elif x_min is not None and x_max is not None and x_tick_step is not None:
        x_range = (x_min, x_max, x_tick_step)
    return manimlib.NumberLine(x_range=x_range, **kwargs)


def _build_axes(
    scene,
    x_min: float | None = None,
    x_max: float | None = None,
    x_tick_step: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    y_tick_step: float | None = None,
    **kwargs,
):
    x_range = None
    if x_min is not None and x_max is not None and x_tick_step is None:
        x_range = (x_min, x_max)
    elif x_min is not None and x_max is not None and x_tick_step is not None:
        x_range = (x_min, x_max, x_tick_step)
    y_range = None
    if y_min is not None and y_max is not None and y_tick_step is None:
        y_range = (y_min, y_max)
    elif y_min is not None and y_max is not None and y_tick_step is not None:
        y_range = (y_min, y_max, y_tick_step)
    return manimlib.Axes(x_range=x_range, y_range=y_range, **kwargs)


def _build_three_d_axes(
    scene,
    x_min: float | None = None,
    x_max: float | None = None,
    x_tick_step: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    y_tick_step: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    z_tick_step: float | None = None,
    **kwargs,
):
    x_range = None
    if x_min is not None and x_max is not None and x_tick_step is None:
        x_range = (x_min, x_max)
    elif x_min is not None and x_max is not None and x_tick_step is not None:
        x_range = (x_min, x_max, x_tick_step)
    y_range = None
    if y_min is not None and y_max is not None and y_tick_step is None:
        y_range = (y_min, y_max)
    elif y_min is not None and y_max is not None and y_tick_step is not None:
        y_range = (y_min, y_max, y_tick_step)
    z_range = None
    if z_min is not None and z_max is not None and z_tick_step is None:
        z_range = (z_min, z_max)
    elif z_min is not None and z_max is not None and z_tick_step is not None:
        z_range = (z_min, z_max, z_tick_step)
    return manimlib.ThreeDAxes(x_range=x_range, y_range=y_range, z_range=z_range, **kwargs)


def _build_sphere(
    scene,
    u_min: float | None = None,
    u_max: float | None = None,
    u_samples: int | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    v_samples: int | None = None,
    **kwargs,
):
    u_range = None
    v_range = None
    resolution = None
    if u_min is not None and u_max is not None:
        u_range = (u_min, u_max)
    if v_min is not None and v_max is not None:
        v_range = (v_min, v_max)
    if u_samples is not None and v_samples is not None:
        resolution = (u_samples, v_samples)
    return manimlib.Sphere(
        u_range=u_range,
        v_range=v_range,
        resolution=resolution,
        **kwargs,
    )


def _build_square_3d(
    scene,
    u_min: float | None = None,
    u_max: float | None = None,
    u_samples: int | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    v_samples: int | None = None,
    **kwargs,
):
    u_range = None
    v_range = None
    resolution = None
    if u_min is not None and u_max is not None:
        u_range = (u_min, u_max)
    if v_min is not None and v_max is not None:
        v_range = (v_min, v_max)
    if u_samples is not None and v_samples is not None:
        resolution = (u_samples, v_samples)
    return manimlib.Square3D(
        u_range=u_range,
        v_range=v_range,
        resolution=resolution,
        **kwargs,
    )


def _build_cube(
    scene,
    square_u_samples: int | None = None,
    square_v_samples: int | None = None,
    **kwargs,
):
    return manimlib.Cube(
        square_resolution=(square_u_samples, square_v_samples) if square_u_samples is not None and square_v_samples is not None else None,
        **kwargs,
    )


def _build_prism(
    scene,
    square_u_samples: int | None = None,
    square_v_samples: int | None = None,
    **kwargs,
):
    return manimlib.Prism(
        square_resolution=(square_u_samples, square_v_samples) if square_u_samples is not None and square_v_samples is not None else None,
        **kwargs,
    )


def _build_line_3d(
    scene,
    u_min: float | None = None,
    u_max: float | None = None,
    u_samples: int | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    v_samples: int | None = None,
    **kwargs,
):
    u_range = None
    v_range = None
    resolution = None
    if u_min is not None and u_max is not None:
        u_range = (u_min, u_max)
    if v_min is not None and v_max is not None:
        v_range = (v_min, v_max)
    if u_samples is not None and v_samples is not None:
        resolution = (u_samples, v_samples)
    return manimlib.Line3D(
        u_range=u_range,
        v_range=v_range,
        resolution=resolution,
        **kwargs,
    )


def _build_disk_3d(
    scene,
    u_min: float | None = None,
    u_max: float | None = None,
    u_samples: int | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    v_samples: int | None = None,
    **kwargs,
):
    u_range = None
    v_range = None
    resolution = None
    if u_min is not None and u_max is not None:
        u_range = (u_min, u_max)
    if v_min is not None and v_max is not None:
        v_range = (v_min, v_max)
    if u_samples is not None and v_samples is not None:
        resolution = (u_samples, v_samples)
    return manimlib.Disk3D(
        u_range=u_range,
        v_range=v_range,
        resolution=resolution,
        **kwargs,
    )


def _build_cone(
    scene,
    u_min: float | None = None,
    u_max: float | None = None,
    u_samples: int | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    v_samples: int | None = None,
    **kwargs,
):
    u_range = None
    v_range = None
    resolution = None
    if u_min is not None and u_max is not None:
        u_range = (u_min, u_max)
    if v_min is not None and v_max is not None:
        v_range = (v_min, v_max)
    if u_samples is not None and v_samples is not None:
        resolution = (u_samples, v_samples)
    return manimlib.Cone(
        u_range=u_range,
        v_range=v_range,
        resolution=resolution,
        **kwargs,
    )


def _build_cylinder(
    scene,
    u_min: float | None = None,
    u_max: float | None = None,
    u_samples: int | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    v_samples: int | None = None,
    **kwargs,
):
    u_range = None
    v_range = None
    resolution = None
    if u_min is not None and u_max is not None:
        u_range = (u_min, u_max)
    if v_min is not None and v_max is not None:
        v_range = (v_min, v_max)
    if u_samples is not None and v_samples is not None:
        resolution = (u_samples, v_samples)
    return manimlib.Cylinder(
        u_range=u_range,
        v_range=v_range,
        resolution=resolution,
        **kwargs,
    )



def _build_torus(
    scene,
    u_min: float | None = None,
    u_max: float | None = None,
    u_samples: int | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    v_samples: int | None = None,
    **kwargs,
):
    u_range = None
    v_range = None
    resolution = None
    if u_min is not None and u_max is not None:
        u_range = (u_min, u_max)
    if v_min is not None and v_max is not None:
        v_range = (v_min, v_max)
    if u_samples is not None and v_samples is not None:
        resolution = (u_samples, v_samples)
    return manimlib.Torus(
        u_range=u_range,
        v_range=v_range,
        resolution=resolution,
        **kwargs,
    )


def _build_parametric_surface(
    scene,
    uv_func,
    u_min: float | None = None,
    u_max: float | None = None,
    u_samples: int | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    v_samples: int | None = None,
    **kwargs,
):
    u_range = None
    v_range = None
    resolution = None
    if u_min is not None and u_max is not None:
        u_range = (u_min, u_max)
    if v_min is not None and v_max is not None:
        v_range = (v_min, v_max)
    if u_samples is not None and v_samples is not None:
        resolution = (u_samples, v_samples)
    return manimlib.ParametricSurface(
        uv_func,
        u_range=u_range,
        v_range=v_range,
        resolution=resolution,
        **kwargs,
    )


def _build_parametric_curve(
    scene,
    t_func,
    t_min: float | None = None,
    t_max: float | None = None,
    t_samples: int | None = None,
    **kwargs,
):
    t_range = None
    if t_min is not None and t_max is not None and t_samples is None:
        t_range = (t_min, t_max)
    elif t_min is not None and t_max is not None and t_samples is not None:
        t_range = (t_min, t_max, (t_max - t_min) / t_samples)
    return manimlib.ParametricCurve(
        t_func,
        t_range=t_range,
        **kwargs,
    )



def _build_number_plane(
    scene,
    x_min: float | None = None,
    x_max: float | None = None,
    x_tick_step: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    y_tick_step: float | None = None,
    **kwargs,
):
    x_range = None
    if x_min is not None and x_max is not None and x_tick_step is None:
        x_range = (x_min, x_max)
    elif x_min is not None and x_max is not None and x_tick_step is not None:
        x_range = (x_min, x_max, x_tick_step)
    y_range = None
    if y_min is not None and y_max is not None and y_tick_step is None:
        y_range = (y_min, y_max)
    elif y_min is not None and y_max is not None and y_tick_step is not None:
        y_range = (y_min, y_max, y_tick_step)
    return manimlib.NumberPlane(x_range=x_range, y_range=y_range, **kwargs)


def _build_vector_field(
    scene,
    func,
    coordinate_system,
    norm_to_opacity_func=None,
    magnitude_min: float | None = None,
    magnitude_max: float | None = None,
    **kwargs,
):
    magnitude_range = None
    if magnitude_min is not None and magnitude_max is not None:
        magnitude_range = (magnitude_min, magnitude_max)
    return manimlib.VectorField(
        func=func,
        norm_to_opacity_func=norm_to_opacity_func,
        coordinate_system=coordinate_system,
        magnitude_range=magnitude_range,
        **kwargs,
    )


def _build_stream_lines(
    scene,
    func,
    coordinate_system,
    magnitude_min: float | None = None,
    magnitude_max: float | None = None,
    **kwargs,
):
    magnitude_range = None
    if magnitude_min is not None and magnitude_max is not None:
        magnitude_range = (magnitude_min, magnitude_max)
    return manimlib.StreamLines(
        func=func,
        coordinate_system=coordinate_system,
        magnitude_range=magnitude_range,
        **kwargs,
    )


def _build_complex_plane(
    scene,
    x_min: float | None = None,
    x_max: float | None = None,
    x_tick_step: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    y_tick_step: float | None = None,
    **kwargs,
):
    x_range = None
    if x_min is not None and x_max is not None and x_tick_step is None:
        x_range = (x_min, x_max)
    elif x_min is not None and x_max is not None and x_tick_step is not None:
        x_range = (x_min, x_max, x_tick_step)
    y_range = None
    if y_min is not None and y_max is not None and y_tick_step is None:
        y_range = (y_min, y_max)
    elif y_min is not None and y_max is not None and y_tick_step is not None:
        y_range = (y_min, y_max, y_tick_step)
    return manimlib.ComplexPlane(x_range=x_range, y_range=y_range, **kwargs)



def _build_coordinate_system_area_under_graph(
    scene,
    coordinate_system,
    graph,
    x_min: float | None = None,
    x_max: float | None = None,
    fill_color=manimlib.BLUE,
    fill_opacity: float = 0.5,
    **kwargs,
):
    method = _resolve_coordinate_system_method(
        coordinate_system, "get_area_under_graph"
    )
    x_range = None
    if x_min is not None and x_max is not None:
        x_range = (x_min, x_max)
    return method(
        graph,
        x_range=x_range,
        fill_color=fill_color,
        fill_opacity=fill_opacity,
        **kwargs,
    )


def _build_coordinate_system_graph_label(
    scene,
    coordinate_system,
    graph,
    label="f(x)",
    x=None,
    direction=manimlib.RIGHT,
    buff=manimlib.MED_SMALL_BUFF,
    color=None,
    **kwargs,
):
    method = _resolve_coordinate_system_method(
        coordinate_system, "get_graph_label"
    )
    return method(
        graph,
        label=label,
        x=x,
        direction=direction,
        buff=buff,
        color=color,
        **kwargs,
    )


def _build_coordinate_system_v_line_to_graph(
    scene,
    coordinate_system,
    x,
    graph,
    **kwargs,
):
    method = _resolve_coordinate_system_method(
        coordinate_system, "get_v_line_to_graph"
    )
    return method(x, graph, **kwargs)


def _build_slider(
    scene,
    value_tracker,
    x_min: float | None = None,
    x_max: float | None = None,
    **kwargs,
):
    if value_tracker is None or not isinstance(
        value_tracker, manimlib.ValueTracker
    ):
        raise TypeError("Slider requires a ValueTracker in 'value_tracker'")
    x_range = None
    if x_min is not None and x_max is not None:
        x_range = (x_min, x_max)
    return manimlib.Slider(value_tracker=value_tracker, x_range=x_range, **kwargs)


def _build_coordinate_system_h_line_to_graph(
    scene,
    coordinate_system,
    x,
    graph,
    **kwargs,
):
    method = _resolve_coordinate_system_method(
        coordinate_system, "get_h_line_to_graph"
    )
    return method(x, graph, **kwargs)


def _build_time_varying_vector_field(
    scene,
    func,
    norm_to_opacity_func=None,
    coordinate_system=None,
    magnitude_min: float | None = None,
    magnitude_max: float | None = None,
    **kwargs,
):
    magnitude_range = None
    if magnitude_min is not None and magnitude_max is not None:
        magnitude_range = (magnitude_min, magnitude_max)
    return manimlib.TimeVaryingVectorField(
        time_func=func,
        norm_to_opacity_func=norm_to_opacity_func,
        coordinate_system=coordinate_system,
        magnitude_range=magnitude_range,
        **kwargs,
    )




def _build_three_d_axes_graph_surface(
    scene,
    coordinate_system,
    function,
    color=manimlib.BLUE_E,
    opacity: float = 0.9,
    u_min: float | None = None,
    u_max: float | None = None,
    u_samples: int | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    v_samples: int | None = None,
    **kwargs,
):
    axes = _require_three_d_axes(coordinate_system)
    method = _resolve_coordinate_system_method(axes, "get_graph")
    u_range = None
    v_range = None
    resolution = None
    if u_min is not None and u_max is not None:
        u_range = (u_min, u_max)
    if v_min is not None and v_max is not None:
        v_range = (v_min, v_max)
    if u_samples is not None and v_samples is not None:
        resolution = (u_samples, v_samples)
    return method(
        function,
        color=color,
        opacity=opacity,
        u_range=u_range,
        v_range=v_range,
        resolution=resolution,
        **kwargs,
    )


def _build_three_d_axes_parametric_surface(
    scene,
    coordinate_system,
    function,
    color=manimlib.BLUE_E,
    opacity: float = 0.9,
    **kwargs,
):
    axes = _require_three_d_axes(coordinate_system)
    method = _resolve_coordinate_system_method(
        axes, "get_parametric_surface"
    )
    return method(function, color=color, opacity=opacity, **kwargs)


def init_all_mobjects(controller: LLMSceneController) -> None:
    """Register all mobject builders and schemas for LLM actions v2."""
    value = controller.schema_value()
    ref = controller.schema_ref()

    vect = _vector3()
    vect_or_ref = _any_of(vect, ref)
    vect_array = _array_of(vect)
    ref_array = _array_of(ref)
    str_array = _array_of(_str())
    dict_value = _dict_of(value)

    # Base kwargs schemas
    mobject_kwargs = {
        "color": value,
        "opacity": _num(),
        "shading": _shading_components(),
        "is_fixed_in_frame": _bool(),
        "depth_test": _bool(),
        "z_index": _int(),
    }

    vmobject_kwargs = _merge_props(
        {key: value for key, value in mobject_kwargs.items() if key != "depth_test"},
        {
            "fill_color": value,
            "fill_opacity": value,
            "stroke_color": value,
            "stroke_opacity": value,
            "stroke_width": value,
            "stroke_behind": _bool(),
            "background_image_file": _str(),
            "long_lines": _bool(),
            "joint_type": {"type": "string", "enum": ["bevel", "miter", "auto", "no_joint"]},
            "flat_stroke": _bool(),
            "scale_stroke_with_zoom": _bool(),
            "use_simple_quadratic_approx": _bool(),
            "anti_alias_width": _num(),
            "fill_border_width": _num(),
        },
    )

    svg_kwargs = _merge_props(
        vmobject_kwargs,
        {
            "file_name": _str(),
            "svg_string": _str(),
            "should_center": _bool(),
            "height": _num(),
            "width": _num(),
            "unbatch": value,
            "svg_default": dict_value,
            "path_string_config": dict_value,
            "needs_flip": _bool(),
        },
    )

    string_kwargs = _merge_props(
        svg_kwargs,
        {
            "fill_border_width": _num(),
            "base_color": value,
            "isolate": value,
            "protect": value,
            "use_labelled_svg": _bool(),
        },
    )

    tex_kwargs = _merge_props(
        string_kwargs,
        {
            "font_size": _int(),
            "alignment": _str(),
            "template": _str(),
            "additional_preamble": _str(),
            "tex_to_color_map": dict_value,
            "t2c": dict_value,
            "isolate": value,
            "use_labelled_svg": _bool(),
        },
    )

    typst_kwargs = _merge_props(
        string_kwargs,
        {
            "font_size": _int(),
            "text_font": value,
            "math_font": value,
            "code_font": value,
            "typst_to_color_map": dict_value,
            "t2c": dict_value,
        },
    )

    surface_kwargs = _merge_props(
        mobject_kwargs,
        {
            "u_min": _num(),
            "u_max": _num(),
            "v_min": _num(),
            "v_max": _num(),
            "u_samples": _int(),
            "v_samples": _int(),
            "preferred_creation_axis": _int(),
            "epsilon": _num(),
            "normal_nudge": _num(),
        },
    )

    numberline_base = _merge_props(
        vmobject_kwargs,
        {
            "x_min": _num(),
            "x_max": _num(),
            "x_tick_step": _num(),
            "color": value,
            "stroke_width": _num(),
            "unit_size": _num(),
            "width": _num(),
            "include_ticks": _bool(),
            "tick_size": _num(),
            "longer_tick_multiple": _num(),
            "tick_offset": _num(),
            "big_tick_spacing": _num(),
            "big_tick_numbers": _array_of(_num()),
            "include_numbers": _bool(),
            "line_to_number_direction": vect_or_ref,
            "line_to_number_buff": _num(),
            "include_tip": _bool(),
            "tip_config": dict_value,
            "decimal_number_config": dict_value,
            "numbers_to_exclude": value,
        },
    )

    axes_base = _merge_props(
        vmobject_kwargs,
        {
            "x_min": _num(),
            "x_max": _num(),
            "x_tick_step": _num(),
            "y_min": _num(),
            "y_max": _num(),
            "y_tick_step": _num(),
            "num_sampled_graph_points_per_tick": _int(),
            "axis_config": dict_value,
            "x_axis_config": dict_value,
            "y_axis_config": dict_value,
            "height": _num(),
            "width": _num(),
            "unit_size": _num(),
        },
    )

    # Core mobjects
    _register(
        controller,
        "Mobject",
        properties=mobject_kwargs,
    )
    _register(
        controller,
        "Group",
        properties=_merge_props(
            mobject_kwargs, {"mobjects": ref_array}
        ),
        positional=["mobjects"],
        spread=["mobjects"],
        required=[],
    )
    _register(
        controller,
        "Point",
        properties=_merge_props(
            mobject_kwargs,
            {
                "location": vect_or_ref,
                "artificial_width": _num(),
                "artificial_height": _num(),
            },
        ),
    )

    # Vectorized / point-cloud base types
    _register(
        controller,
        "VMobject",
        properties=vmobject_kwargs,
    )
    _register(
        controller,
        "VGroup",
        properties=_merge_props(
            vmobject_kwargs, {"vmobjects": ref_array}
        ),
        positional=["vmobjects"],
        spread=["vmobjects"],
        required=[],
    )
    _register(
        controller,
        "VectorizedPoint",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "location": vect_or_ref,
            },
        ),
    )
    _register(
        controller,
        "CurvesAsSubmobjects",
        properties=_merge_props(vmobject_kwargs, {"vmobject": ref}),
        required=["vmobject"],
    )
    _register(
        controller,
        "DashedVMobject",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "vmobject": ref,
                "num_dashes": _int(),
                "positive_space_ratio": _num(),
            },
        ),
        required=["vmobject"],
    )
    _register(
        controller,
        "VHighlight",
        properties={
            "vmobject": ref,
            "n_layers": _int(),
            "color_bounds": _array_of(
                value, min_items=2, max_items=2
            ),
            "max_stroke_addition": _num(),
        },
        required=["vmobject"],
    )

    _register(controller, "PMobject", properties=mobject_kwargs)
    _register(
        controller,
        "PGroup",
        properties=_merge_props(mobject_kwargs, {"pmobs": ref_array}),
        positional=["pmobs"],
        spread=["pmobs"],
        required=[],
    )

    # Surface family
    _register(
        controller,
        "ParametricSurface",
        properties=_merge_props(
            surface_kwargs,
            {
                "uv_func": _py_function_expr(
                    "(u, v) -> np.ndarray",
                    "lambda u, v: np.array([u, v, 0.0])",
                ),
            },
        ),
        required=["uv_func"],
        builder=_build_parametric_surface,   
    )
    _register(
        controller,
        "SGroup",
        properties={"parametric_surfaces": ref_array},
        positional=["parametric_surfaces"],
        spread=["parametric_surfaces"],
        required=[],
    )
    _register(
        controller,
        "TexturedSurface",
        properties=_merge_props(
            surface_kwargs,
            {
                "uv_surface": ref,
                "image_file": _str(),
                "dark_image_file": _str(),
            },
        ),
        required=["uv_surface", "image_file"],
    )
    _register(
        controller,
        "ThreeDModel",
        properties={
            "obj_file": _str(),
            "height": _num(),
        },
        required=["obj_file"],
    )

    # Dot cloud / image / special types
    _register(
        controller,
        "DotCloud",
        properties=_merge_props(
            mobject_kwargs,
            {
                "points": vect_array,
                "radius": _num(),
                "glow_factor": _num(),
                "anti_alias_width": _num(),
            },
        ),
    )
    _register(
        controller,
        "TrueDot",
        properties=_merge_props(
            mobject_kwargs,
            {
                "center": vect_or_ref,
                "radius": _num(),
                "glow_factor": _num(),
                "anti_alias_width": _num(),
            },
        ),
    )
    _register(
        controller,
        "GlowDots",
        properties=_merge_props(
            mobject_kwargs,
            {
                "points": vect_array,
                "radius": _num(),
                "glow_factor": _num(),
                "anti_alias_width": _num(),
            },
        ),
    )
    _register(
        controller,
        "GlowDot",
        properties=_merge_props(
            mobject_kwargs,
            {
                "center": vect_or_ref,
                "radius": _num(),
                "glow_factor": _num(),
                "anti_alias_width": _num(),
            },
        ),
    )

    _register(
        controller,
        "ImageMobject",
        properties=_merge_props(
            mobject_kwargs,
            {
                "filename": value,
                "height": _num(),
            },
        ),
        required=["filename"],
    )
    _register(
        controller,
        "VideoMobject",
        properties=_merge_props(
            mobject_kwargs,
            {
                "iterator": ref,
                "video_path_or_camera": _any_of(_str(), _int()),
                "flip_horizontal": _bool(),
                "auto_play": _bool(),
                "height": _num(),
            },
        ),
        builder=_build_video_mobject,
    )

    _register(
        controller,
        "MagnifyingGlass",
        properties=_merge_props(
            mobject_kwargs,
            {
                "radius": _num(),
                "magnification": _num(),
                "fixed_in_frame": _bool(),
                "rasterize": _bool(),
            },
        ),
        builder=_build_magnifying_glass,
    )
    _register(
        controller,
        "MagnifyingGlassGroup",
        properties=_merge_props(
            mobject_kwargs, {"children": ref_array}
        ),
        positional=["children"],
        spread=["children"],
        required=[],
    )

    _register(
        controller,
        "MaskMobject",
        properties=_merge_props(
            mobject_kwargs,
            {
                "src_mobject": ref,
                "mask_mobject": ref,
                "height": _num(),
            },
        ),
        required=["src_mobject", "mask_mobject"],
        builder=_build_mask_mobject,
    )
    _register(
        controller,
        "MaskMobjectGroup",
        properties=_merge_props(
            mobject_kwargs, {"mobjects": ref_array}
        ),
        positional=["mobjects"],
        spread=["mobjects"],
        required=[],
    )

    # Geometry
    _register(
        controller, "TipableVMobject", properties=vmobject_kwargs
    )
    _register(
        controller,
        "Arc",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "start_angle": _num(),
                "angle": _num(),
                "radius": _num(),
                "n_components": _int(),
                "arc_center": vect_or_ref,
            },
        ),
    )
    _register(
        controller,
        "ArcBetweenPoints",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "start": vect_or_ref,
                "end": vect_or_ref,
                "angle": _num(),
                "radius": _num(),
                "n_components": _int(),
                "arc_center": vect_or_ref,
                "start_angle": _num(),
            },
        ),
        required=["start", "end"],
    )
    _register(
        controller,
        "CurvedArrow",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "start_point": vect_or_ref,
                "end_point": vect_or_ref,
                "angle": _num(),
                "radius": _num(),
            },
        ),
        required=["start_point", "end_point"],
    )
    _register(
        controller,
        "CurvedDoubleArrow",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "start_point": vect_or_ref,
                "end_point": vect_or_ref,
                "angle": _num(),
                "radius": _num(),
            },
        ),
        required=["start_point", "end_point"],
    )
    _register(
        controller,
        "Circle",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "start_angle": _num(),
                "stroke_color": value,
                "radius": _num(),
                "angle": _num(),
                "n_components": _int(),
                "arc_center": vect_or_ref,
            },
        ),
    )
    _register(
        controller,
        "Dot",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "point": vect_or_ref,
                "radius": _num(),
                "stroke_color": value,
                "stroke_width": _num(),
                "fill_opacity": _num(),
                "fill_color": value,
                "start_angle": _num(),
                "angle": _num(),
            },
        ),
    )
    _register(
        controller,
        "SmallDot",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "point": vect_or_ref,
                "radius": _num(),
                "stroke_color": value,
                "stroke_width": _num(),
                "fill_opacity": _num(),
                "fill_color": value,
            },
        ),
    )
    _register(
        controller,
        "Ellipse",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "width": _num(),
                "height": _num(),
                "radius": _num(),
                "start_angle": _num(),
                "angle": _num(),
            },
        ),
    )
    _register(
        controller,
        "AnnularSector",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "angle": _num(),
                "start_angle": _num(),
                "inner_radius": _num(),
                "outer_radius": _num(),
                "arc_center": vect_or_ref,
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_width": _num(),
            },
        ),
    )
    _register(
        controller,
        "Sector",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "angle": _num(),
                "radius": _num(),
                "start_angle": _num(),
                "inner_radius": _num(),
                "outer_radius": _num(),
                "arc_center": vect_or_ref,
            },
        ),
    )
    _register(
        controller,
        "Annulus",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "inner_radius": _num(),
                "outer_radius": _num(),
                "fill_opacity": _num(),
                "stroke_width": _num(),
                "fill_color": value,
                "center": vect_or_ref,
            },
        ),
    )

    line_base = _merge_props(
        vmobject_kwargs,
        {
            "start": _any_of(vect, ref),
            "end": _any_of(vect, ref),
            "buff": _num(),
            "path_arc": _num(),
        },
    )
    _register(controller, "Line", properties=line_base)
    _register(
        controller,
        "DashedLine",
        properties=_merge_props(
            line_base,
            {
                "dash_length": _num(),
                "positive_space_ratio": _num(),
            },
        ),
    )
    _register(
        controller,
        "TangentLine",
        properties=_merge_props(
            line_base,
            {
                "vmob": ref,
                "alpha": _num(),
                "length": _num(),
                "d_alpha": _num(),
            },
        ),
        required=["vmob", "alpha"],
    )
    _register(
        controller,
        "Elbow",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "width": _num(),
                "angle": _num(),
            },
        ),
    )
    _register(
        controller,
        "StrokeArrow",
        properties=_merge_props(
            line_base,
            {
                "stroke_color": value,
                "stroke_width": _num(),
                "tip_width_ratio": _num(),
                "tip_len_to_width": _num(),
                "max_tip_length_to_length_ratio": _num(),
                "max_width_to_length_ratio": _num(),
            },
        ),
        required=["start", "end"],
    )
    _register(
        controller,
        "Arrow",
        properties=_merge_props(
            line_base,
            {
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_width": _num(),
                "thickness": _num(),
                "tip_width_ratio": _num(),
                "tip_angle": _num(),
                "max_tip_length_to_length_ratio": _num(),
                "max_width_to_length_ratio": _num(),
            },
        ),
    )
    _register(
        controller,
        "Vector",
        properties=_merge_props(
            line_base,
            {
                "direction": vect_or_ref,
            },
        ),
    )
    _register(
        controller,
        "CubicBezier",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "a0": vect_or_ref,
                "h0": vect_or_ref,
                "h1": vect_or_ref,
                "a1": vect_or_ref,
            },
        ),
        required=["a0", "h0", "h1", "a1"],
    )
    _register(
        controller,
        "Polygon",
        properties=_merge_props(
            vmobject_kwargs, {"vertices": vect_array}
        ),
        positional=["vertices"],
        spread=["vertices"],
        required=[],
    )
    _register(
        controller,
        "Polyline",
        properties=_merge_props(
            vmobject_kwargs, {"vertices": vect_array}
        ),
        positional=["vertices"],
        spread=["vertices"],
        required=[],
    )
    _register(
        controller,
        "RegularPolygon",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "n": _int(),
                "radius": _num(),
                "start_angle": _num(),
                "vertices": vect_array,
            },
        ),
    )
    _register(
        controller,
        "Triangle",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "n": _int(),
                "radius": _num(),
                "start_angle": _num(),
            },
        ),
    )
    _register(
        controller,
        "ArrowTip",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "angle": _num(),
                "width": _num(),
                "length": _num(),
                "fill_opacity": _num(),
                "fill_color": value,
                "stroke_width": _num(),
                "tip_style": _int(),
            },
        ),
    )
    _register(
        controller,
        "Rectangle",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "width": _num(),
                "height": _num(),
                "vertices": vect_array,
            },
        ),
    )
    _register(
        controller,
        "Square",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "side_length": _num(),
                "width": _num(),
                "height": _num(),
            },
        ),
    )
    _register(
        controller,
        "RoundedRectangle",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "width": _num(),
                "height": _num(),
                "corner_radius": _num(),
            },
        ),
    )

    # Frame
    _register(
        controller,
        "ScreenRectangle",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "aspect_ratio": _num(),
                "height": _num(),
                "width": _num(),
            },
        ),
    )
    _register(
        controller,
        "FullScreenRectangle",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "height": _num(),
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_width": _num(),
                "aspect_ratio": _num(),
                "width": _num(),
            },
        ),
    )
    _register(
        controller,
        "FullScreenFadeRectangle",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "stroke_width": _num(),
                "fill_color": value,
                "fill_opacity": _num(),
                "height": _num(),
                "aspect_ratio": _num(),
                "width": _num(),
            },
        ),
    )

    # Functions and curves
    _register(
        controller,
        "ParametricCurve",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "t_func": _py_function_expr(
                    "(t) -> np.ndarray",
                    "lambda t: np.array([t, np.sin(t), 0.0])",
                ),
                "t_min": _num(),
                "t_max": _num(),
                "t_samples": _int(),
                "epsilon": _num(),
                "discontinuities": _array_of(_num()),
                "use_smoothing": _bool(),
            },
        ),
        required=["t_func"],
        builder=_build_parametric_curve,
    )
    _register(
        controller,
        "FunctionGraph",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "function": _py_function_expr(
                    "(x) -> float | complex",
                    "lambda x: np.sin(x)",
                ),
                "x_min": _num(),
                "x_max": _num(),
                "x_samples": _int(),
                "color": value,
            },
        ),
        required=["function"],
        builder=_build_function_graph,
    )
    _register(
        controller,
        "ImplicitFunction",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "function": _py_function_expr(
                    "(x, y) -> float",
                    "lambda x, y: x**2 + y**2 - 1.0",
                ),
                "x_min": _num(),
                "x_max": _num(),
                "y_min": _num(),
                "y_max": _num(),
                "min_depth": _int(),
                "max_quads": _int(),
                "use_smoothing": _bool(),
            },
        ),
        required=["function"],
        builder=_build_implicit_function,
    )

    # Coordinate systems
    _register(controller, "Axes", properties=axes_base, builder=_build_axes)
    _register(
        controller,
        "ThreeDAxes",
        properties=_merge_props(
            axes_base,
            {
                "z_min": _num(),
                "z_max": _num(),
                "z_tick_step": _num(),
                "z_axis_config": dict_value,
                "z_normal": vect_or_ref,
                "depth": _num(),
            },
        ),
        builder=_build_three_d_axes,
    )
    _register(
        controller,
        "NumberPlane",
        properties=_merge_props(
            axes_base,
            {
                "background_line_style": dict_value,
                "faded_line_style": dict_value,
                "faded_line_ratio": _int(),
                "make_smooth_after_applying_functions": _bool(),
            },
        ),
        builder=_build_number_plane,
    )
    _register(
        controller,
        "ComplexPlane",
        properties=_merge_props(
            axes_base,
            {
                "background_line_style": dict_value,
                "faded_line_style": dict_value,
                "faded_line_ratio": _int(),
                "make_smooth_after_applying_functions": _bool(),
            },
        ),
        builder=_build_complex_plane,
    )
    _register(
        controller,
        "CoordinateSystemGraph",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "coordinate_system": ref,
                "function": _py_function_expr(
                    "(x) -> float | complex",
                    "lambda x: np.sin(x)",
                ),
                "x_min": _num(),
                "x_max": _num(),
                "x_samples": _int(),
                "bind": _bool(),
            },
        ),
        required=["coordinate_system", "function"],
        builder=_build_coordinate_system_graph,
    )
    _register(
        controller,
        "CoordinateSystemParametricCurve",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "coordinate_system": ref,
                "function": _py_function_expr(
                    "(t) -> np.ndarray",
                    "lambda t: np.array([t, np.sin(t), 0.0])",
                ),
                "t_min": _num(),
                "t_max": _num(),
                "t_samples": _int(),
                "epsilon": _num(),
                "discontinuities": _array_of(_num()),
                "use_smoothing": _bool(),
            },
        ),
        required=["coordinate_system", "function"],
        builder=_build_coordinate_system_parametric_curve,
    )
    _register(
        controller,
        "CoordinateSystemScatterplot",
        properties=_merge_props(
            mobject_kwargs,
            {
                "coordinate_system": ref,
                "x_values": _array_of(_num()),
                "y_values": _array_of(_num()),
                "radius": _num(),
                "glow_factor": _num(),
                "anti_alias_width": _num(),
            },
        ),
        required=["coordinate_system", "x_values", "y_values"],
        builder=_build_coordinate_system_scatterplot,
    )
    _register(
        controller,
        "CoordinateSystemTangentLine",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "coordinate_system": ref,
                "x": _num(),
                "graph": ref,
                "length": _num(),
            },
        ),
        required=["coordinate_system", "x", "graph"],
        builder=_build_coordinate_system_tangent_line,
    )
    _register(
        controller,
        "CoordinateSystemRiemannRectangles",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "coordinate_system": ref,
                "graph": ref,
                "x_min": _num(),
                "x_max": _num(),
                "x_samples": _int(),
                "input_sample_type": _str(),
                "stroke_width": _num(),
                "stroke_color": value,
                "fill_opacity": _num(),
                "colors": _array_of(value),
                "negative_color": value,
                "stroke_background": _bool(),
                "show_signed_area": _bool(),
            },
        ),
        required=["coordinate_system", "graph"],
        builder=_build_coordinate_system_riemann_rectangles,
    )
    _register(
        controller,
        "CoordinateSystemAreaUnderGraph",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "coordinate_system": ref,
                "graph": ref,
                "x_min": _num(),
                "x_max": _num(),
                "fill_color": value,
                "fill_opacity": _num(),
            },
        ),
        required=["coordinate_system", "graph"],
        builder=_build_coordinate_system_area_under_graph,
    )
    _register(
        controller,
        "CoordinateSystemGraphLabel",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "coordinate_system": ref,
                "graph": ref,
                "label": value,
                "x": _num(),
                "direction": vect_or_ref,
                "buff": _num(),
                "color": value,
            },
        ),
        required=["coordinate_system", "graph"],
        builder=_build_coordinate_system_graph_label,
    )
    _register(
        controller,
        "CoordinateSystemVLineToGraph",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "coordinate_system": ref,
                "x": _num(),
                "graph": ref,
                "color": value,
                "stroke_width": _num(),
            },
        ),
        required=["coordinate_system", "x", "graph"],
        builder=_build_coordinate_system_v_line_to_graph,
    )
    _register(
        controller,
        "CoordinateSystemHLineToGraph",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "coordinate_system": ref,
                "x": _num(),
                "graph": ref,
                "color": value,
                "stroke_width": _num(),
            },
        ),
        required=["coordinate_system", "x", "graph"],
        builder=_build_coordinate_system_h_line_to_graph,
    )
    _register(
        controller,
        "ThreeDAxesGraphSurface",
        properties=_merge_props(
            surface_kwargs,
            {
                "coordinate_system": ref,
                "function": _py_function_expr(
                    "(u, v) -> float",
                    "lambda u, v: np.sin(u) * np.cos(v)",
                ),
                "color": value,
                "opacity": _num(),
            },
        ),
        required=["coordinate_system", "function"],
        builder=_build_three_d_axes_graph_surface,
    )
    _register(
        controller,
        "ThreeDAxesParametricSurface",
        properties=_merge_props(
            surface_kwargs,
            {
                "coordinate_system": ref,
                "function": _py_function_expr(
                    "(u, v) -> np.ndarray",
                    "lambda u, v: np.array([u, v, np.sin(u) * np.cos(v)])",
                ),
                "color": value,
                "opacity": _num(),
            },
        ),
        required=["coordinate_system", "function"],
        builder=_build_three_d_axes_parametric_surface,
    )

    # Changing / tracing
    _register(
        controller,
        "AnimatedBoundary",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "vmobject": ref,
                "colors": _array_of(value),
                "max_stroke_width": _num(),
                "cycle_rate": _num(),
                "back_and_forth": _bool(),
                "draw_rate_func": _py_function_expr(
                    "(t) -> float",
                    "lambda t: t",
                ),
                "fade_rate_func": _py_function_expr(
                    "(t) -> float",
                    "lambda t: 1.0 - t",
                ),
            },
        ),
        required=["vmobject"],
        builder=_make_function_expression_builder(
            "AnimatedBoundary",
            "draw_rate_func",
            "fade_rate_func",
        ),
    )
    _register(
        controller,
        "TracedPath",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "traced_point_func": _py_function_expr(
                    "() -> np.ndarray",
                    "lambda: np.array([0.0, 0.0, 0.0])",
                ),
                "time_traced": _num(),
                "time_per_anchor": _num(),
                "stroke_color": value,
                "stroke_width": value,
                "stroke_opacity": _num(),
            },
        ),
        required=["traced_point_func"],
        builder=_make_function_expression_builder(
            "TracedPath",
            "traced_point_func",
        ),
    )
    _register(
        controller,
        "TracingTail",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "mobject_or_func": _any_of(
                    ref,
                    _py_function_expr(
                        "() -> np.ndarray",
                        "lambda: np.array([0.0, 0.0, 0.0])",
                    ),
                ),
                "time_traced": _num(),
                "stroke_color": value,
                "stroke_width": value,
                "stroke_opacity": value,
            },
        ),
        required=["mobject_or_func"],
        builder=_make_function_expression_builder(
            "TracingTail",
            "mobject_or_func",
        ),
    )

    # Boolean ops
    _register(
        controller,
        "Union",
        properties=_merge_props(
            vmobject_kwargs, {"vmobjects": ref_array}
        ),
        positional=["vmobjects"],
        spread=["vmobjects"],
        required=["vmobjects"],
    )
    _register(
        controller,
        "Difference",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "subject": ref,
                "clip": ref,
            },
        ),
        required=["subject", "clip"],
    )
    _register(
        controller,
        "Intersection",
        properties=_merge_props(
            vmobject_kwargs, {"vmobjects": ref_array}
        ),
        positional=["vmobjects"],
        spread=["vmobjects"],
        required=["vmobjects"],
    )
    _register(
        controller,
        "Exclusion",
        properties=_merge_props(
            vmobject_kwargs, {"vmobjects": ref_array}
        ),
        positional=["vmobjects"],
        spread=["vmobjects"],
        required=["vmobjects"],
    )

    # Matrix family
    matrix_base = {
        "matrix": value,
        "v_buff": _num(),
        "h_buff": _num(),
        "bracket_h_buff": _num(),
        "bracket_v_buff": _num(),
        "height": _num(),
        "element_config": dict_value,
        "element_alignment_corner": vect_or_ref,
        "ellipses_row": _int(),
        "ellipses_col": _int(),
    }
    _register(
        controller,
        "Matrix",
        properties=matrix_base,
        required=["matrix"],
    )
    _register(
        controller,
        "DecimalMatrix",
        properties=_merge_props(
            matrix_base,
            {
                "num_decimal_places": _int(),
                "decimal_config": dict_value,
            },
        ),
        required=["matrix"],
    )
    _register(
        controller,
        "IntegerMatrix",
        properties=_merge_props(
            matrix_base,
            {
                "num_decimal_places": _int(),
                "decimal_config": dict_value,
            },
        ),
        required=["matrix"],
    )
    _register(
        controller,
        "TexMatrix",
        properties=_merge_props(
            matrix_base,
            {
                "tex_config": dict_value,
            },
        ),
        required=["matrix"],
    )
    _register(
        controller,
        "MobjectMatrix",
        properties={
            "group": ref,
            "n_rows": _int(),
            "n_cols": _int(),
            "height": _num(),
            "element_alignment_corner": vect_or_ref,
            "v_buff": _num(),
            "h_buff": _num(),
            "bracket_h_buff": _num(),
            "bracket_v_buff": _num(),
            "ellipses_row": _int(),
            "ellipses_col": _int(),
        },
        required=["group"],
    )

    # Numeric displays
    _register(
        controller,
        "DecimalNumber",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "number": value,
                "color": value,
                "stroke_width": _num(),
                "fill_opacity": _num(),
                "fill_border_width": _num(),
                "num_decimal_places": _int(),
                "min_total_width": _num(),
                "include_sign": _bool(),
                "group_with_commas": _bool(),
                "digit_buff_per_font_unit": _num(),
                "show_ellipsis": _bool(),
                "unit": _str(),
                "include_background_rectangle": _bool(),
                "hide_zero_components_on_complex": _bool(),
                "edge_to_fix": vect_or_ref,
                "font_size": _num(),
                "text_config": dict_value,
            },
        ),
    )
    _register(
        controller,
        "Integer",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "number": _int(),
                "num_decimal_places": _int(),
                "color": value,
                "stroke_width": _num(),
                "fill_opacity": _num(),
                "fill_border_width": _num(),
                "font_size": _num(),
                "text_config": dict_value,
            },
        ),
    )

    # Number line family
    _register(controller, "NumberLine", properties=numberline_base, builder=_build_number_line)
    _register(
        controller,
        "Slider",
        properties={
            "value_tracker": ref,
            "x_min": _num(),
            "x_max": _num(),
            "var_name": _str(),
            "width": _num(),
            "unit_size": _num(),
            "arrow_width": _num(),
            "arrow_length": _num(),
            "arrow_color": value,
            "font_size": _int(),
            "label_buff": _num(),
            "num_decimal_places": _int(),
            "tick_size": _num(),
            "number_line_config": dict_value,
            "arrow_tip_config": dict_value,
            "decimal_config": dict_value,
            "angle": _num(),
            "label_direction": vect_or_ref,
            "add_tick_labels": _bool(),
            "tick_label_font_size": _int(),
        },
        required=["value_tracker"],
        builder=_build_slider,
    )

    # Probability and shape matchers
    _register(
        controller,
        "SampleSpace",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "width": _num(),
                "height": _num(),
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_width": _num(),
                "stroke_color": value,
                "default_label_scale_val": _num(),
            },
        ),
    )
    _register(
        controller,
        "BarChart",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "values": _array_of(_num()),
                "height": _num(),
                "width": _num(),
                "n_ticks": _int(),
                "include_x_ticks": _bool(),
                "tick_width": _num(),
                "tick_height": _num(),
                "label_y_axis": _bool(),
                "y_axis_label_height": _num(),
                "max_value": _num(),
                "bar_colors": _array_of(value),
                "bar_fill_opacity": _num(),
                "bar_stroke_width": _num(),
                "bar_names": str_array,
                "bar_label_scale_val": _num(),
            },
        ),
        required=["values"],
    )
    _register(
        controller,
        "SurroundingRectangle",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "mobject": ref,
                "buff": _num(),
                "color": value,
            },
        ),
        required=["mobject"],
    )
    _register(
        controller,
        "BackgroundRectangle",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "mobject": ref,
                "color": value,
                "stroke_width": _num(),
                "stroke_opacity": _num(),
                "fill_opacity": _num(),
                "buff": _num(),
            },
        ),
        required=["mobject"],
    )
    _register(
        controller,
        "Cross",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "mobject": ref,
                "stroke_color": value,
                "stroke_width": value,
            },
        ),
        required=["mobject"],
    )
    _register(
        controller,
        "Underline",
        properties=_merge_props(
            line_base,
            {
                "mobject": ref,
                "buff": _num(),
                "stroke_color": value,
                "stroke_width": value,
                "stretch_factor": _num(),
            },
        ),
        required=["mobject"],
    )

    # Value trackers
    _register(
        controller,
        "ValueTracker",
        properties=_merge_props(
            mobject_kwargs,
            {
                "value": value,
            },
        ),
    )
    _register(
        controller,
        "ExponentialValueTracker",
        properties=_merge_props(
            mobject_kwargs,
            {
                "value": value,
            },
        ),
    )
    _register(
        controller,
        "ComplexValueTracker",
        properties=_merge_props(
            mobject_kwargs,
            {
                "value": value,
            },
        ),
    )

    # Interactive mobjects
    _register(
        controller,
        "MotionMobject",
        properties=_merge_props(mobject_kwargs, {"mobject": ref}),
        required=["mobject"],
    )
    _register(
        controller,
        "Button",
        properties=_merge_props(
            mobject_kwargs,
            {
                "mobject": ref,
                "on_click": _py_function_expr(
                    "(mobject) -> None",
                    "lambda mob: mob.set_opacity(0.5)",
                ),
            },
        ),
        required=["mobject", "on_click"],
        builder=_make_function_expression_builder(
            "Button",
            "on_click",
        ),
    )
    _register(
        controller,
        "ControlMobject",
        properties=_merge_props(
            mobject_kwargs,
            {
                "value": value,
                "mobjects": ref_array,
            },
        ),
        positional=["value", "mobjects"],
        spread=["mobjects"],
        required=["value"],
    )
    _register(
        controller,
        "EnableDisableButton",
        properties=_merge_props(
            mobject_kwargs,
            {
                "value": _bool(),
                "value_type": value,
                "rect_kwargs": dict_value,
                "enable_color": value,
                "disable_color": value,
                "mobjects": ref_array,
            },
        ),
    )
    _register(
        controller,
        "Checkbox",
        properties=_merge_props(
            mobject_kwargs,
            {
                "value": _bool(),
                "value_type": value,
                "rect_kwargs": dict_value,
                "checkmark_kwargs": dict_value,
                "cross_kwargs": dict_value,
                "box_content_buff": _num(),
                "mobjects": ref_array,
            },
        ),
    )
    _register(
        controller,
        "LinearNumberSlider",
        properties=_merge_props(
            mobject_kwargs,
            {
                "value": _num(),
                "value_type": value,
                "min_value": _num(),
                "max_value": _num(),
                "step": _num(),
                "rounded_rect_kwargs": dict_value,
                "circle_kwargs": dict_value,
                "mobjects": ref_array,
            },
        ),
    )
    _register(
        controller,
        "ColorSliders",
        properties=_merge_props(
            mobject_kwargs,
            {
                "sliders_kwargs": dict_value,
                "rect_kwargs": dict_value,
                "background_grid_kwargs": dict_value,
                "sliders_buff": _num(),
                "default_rgb_value": _int(),
                "default_a_value": _num(),
            },
        ),
    )
    _register(
        controller,
        "Textbox",
        properties=_merge_props(
            mobject_kwargs,
            {
                "value": _str(),
                "value_type": value,
                "box_kwargs": dict_value,
                "text_kwargs": dict_value,
                "text_buff": _num(),
                "isInitiallyActive": _bool(),
                "active_color": value,
                "deactive_color": value,
                "mobjects": ref_array,
            },
        ),
    )
    _register(
        controller,
        "ControlPanel",
        properties=_merge_props(
            mobject_kwargs,
            {
                "controls": ref_array,
                "panel_kwargs": dict_value,
                "opener_kwargs": dict_value,
                "opener_text_kwargs": dict_value,
            },
        ),
        positional=["controls"],
        spread=["controls"],
        required=[],
    )

    # Vector fields
    _register(
        controller,
        "VectorField",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "func": _py_function_expr(
                    "(p) -> np.ndarray",
                    "lambda p: np.array([p[1], -p[0], 0.0])",
                ),
                "coordinate_system": ref,
                "sample_coords": vect_array,
                "density": _num(),
                "magnitude_min": _num(),
                "magnitude_max": _num(),
                "color": value,
                "color_map_name": _str(),
                "color_map": ref,
                "stroke_opacity": _num(),
                "stroke_width": _num(),
                "tip_width_ratio": _num(),
                "tip_len_to_width": _num(),
                "max_vect_len": _num(),
                "max_vect_len_to_step_size": _num(),
                "flat_stroke": _bool(),
                "norm_to_opacity_func": _py_function_expr(
                    "(norm) -> float",
                    "lambda norm: min(1.0, norm)",
                ),
            },
        ),
        required=["func", "coordinate_system"],
        builder=_build_vector_field,
    )
    _register(
        controller,
        "TimeVaryingVectorField",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "time_func": _py_function_expr(
                    "(p, t) -> np.ndarray",
                    "lambda p, t: np.array([p[1], -p[0], 0.0])",
                ),
                "coordinate_system": ref,
                "sample_coords": vect_array,
                "density": _num(),
                "magnitude_min": _num(),
                "magnitude_max": _num(),
                "color": value,
                "color_map_name": _str(),
                "color_map": ref,
                "stroke_opacity": _num(),
                "stroke_width": _num(),
                "tip_width_ratio": _num(),
                "tip_len_to_width": _num(),
                "max_vect_len": _num(),
                "max_vect_len_to_step_size": _num(),
                "flat_stroke": _bool(),
                "norm_to_opacity_func": _py_function_expr(
                    "(norm) -> float",
                    "lambda norm: min(1.0, norm)",
                ),
            },
        ),
        required=["time_func", "coordinate_system"],
        builder=_build_time_varying_vector_field,
    )
    _register(
        controller,
        "StreamLines",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "func": _py_function_expr(
                    "(p) -> np.ndarray",
                    "lambda p: np.array([p[1], -p[0], 0.0])",
                ),
                "coordinate_system": ref,
                "density": _num(),
                "n_repeats": _int(),
                "noise_factor": _num(),
                "solution_time": _num(),
                "dt": _num(),
                "arc_len": _num(),
                "max_time_steps": _int(),
                "n_samples_per_line": _int(),
                "cutoff_norm": _num(),
                "stroke_width": _num(),
                "stroke_color": value,
                "stroke_opacity": _num(),
                "color_by_magnitude": _bool(),
                "magnitude_min": _num(),
                "magnitude_max": _num(),
                "taper_stroke_width": _bool(),
                "color_map": _str(),
            },
        ),
        required=["func", "coordinate_system"],
        builder=_build_stream_lines,
    )
    _register(
        controller,
        "AnimatedStreamLines",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "stream_lines": ref,
                "lag_range": _num(),
                "rate_multiple": _num(),
                "line_anim_config": dict_value,
            },
        ),
        required=["stream_lines"],
    )

    # 3D mobjects
    _register(
        controller,
        "SurfaceMesh",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "uv_surface": ref,
                "resolution": _array_of(
                    _int(), min_items=2, max_items=2
                ),
                "stroke_width": _num(),
                "stroke_color": value,
                "normal_nudge": _num(),
                "depth_test": _bool(),
            },
        ),
        required=["uv_surface"],
    )
    _register(
        controller,
        "Sphere",
        properties=_merge_props(
            surface_kwargs,
            {
                "radius": _num(),
                "true_normals": _bool(),
                "clockwise": _bool(),
            },
        ),
        builder=_build_sphere,
    )
    _register(
        controller,
        "Torus",
        properties=_merge_props(
            surface_kwargs,
            {
                "r1": _num(),
                "r2": _num(),
            },
        ),
        builder=_build_torus,
    )
    _register(
        controller,
        "Cylinder",
        properties=_merge_props(
            surface_kwargs,
            {
                "height": _num(),
                "radius": _num(),
                "axis": vect_or_ref,
            },
        ),
        builder=_build_cylinder,
    )

    _register(
        controller,
        "Cone",
        properties=_merge_props(
            surface_kwargs,
            {
                "height": _num(),
                "radius": _num(),
                "axis": vect_or_ref,
            },
        ),
        builder=_build_cone,
    )
    _register(
        controller,
        "Line3D",
        properties=_merge_props(
            surface_kwargs,
            {
                "start": vect_or_ref,
                "end": vect_or_ref,
                "width": _num(),
                "height": _num(),
                "radius": _num(),
                "axis": vect_or_ref,
            },
        ),
        required=["start", "end"],
        builder=_build_line_3d,
    )
    _register(
        controller,
        "Disk3D",
        properties=_merge_props(
            surface_kwargs,
            {
                "radius": _num(),
            },
        ),
        builder=_build_disk_3d,
    )
    _register(
        controller,
        "Square3D",
        properties=_merge_props(
            surface_kwargs,
            {
                "side_length": _num(),
            },
        ),
        builder=_build_square_3d,
    )
    _register(
        controller,
        "Cube",
        properties=_merge_props(
            surface_kwargs,
            {
                "color": value,
                "opacity": _num(),
                "square_resolution": _array_of(
                    _int(), min_items=2, max_items=2
                ),
                "side_length": _num(),
                "parametric_surfaces": ref_array,
            },
        ),
        builder=_build_cube,
    )
    _register(
        controller,
        "Prism",
        properties=_merge_props(
            surface_kwargs,
            {
                "width": _num(),
                "height": _num(),
                "depth": _num(),
                "color": value,
                "opacity": _num(),
                "square_resolution": _array_of(
                    _int(), min_items=2, max_items=2
                ),
                "side_length": _num(),
            },
        ),
        builder=_build_prism,
    )

    # SVG family and text
    _register(controller, "SVGMobject", properties=svg_kwargs)
    _register(
        controller,
        "VMobjectFromSVGPath",
        properties=_merge_props(svg_kwargs, {"path_obj": ref}),
        required=["path_obj"],
    )

    _register(
        controller,
        "Tex",
        properties=_merge_props(
            tex_kwargs, {"tex_strings": str_array}
        ),
        positional=["tex_strings"],
        spread=["tex_strings"],
        required=[],
    )
    _register(
        controller,
        "TexText",
        properties=_merge_props(
            tex_kwargs, {"tex_strings": str_array}
        ),
        positional=["tex_strings"],
        spread=["tex_strings"],
        required=[],
    )

    _register(
        controller,
        "MarkupText",
        properties=_merge_props(
            string_kwargs,
            {
                "text": _str(),
                "font_size": _int(),
                "height": _num(),
                "justify": _bool(),
                "indent": _num(),
                "alignment": _str(),
                "line_width": _num(),
                "font": _str(),
                "slant": _str(),
                "weight": _str(),
                "gradient": _array_of(value),
                "line_spacing_height": _num(),
                "text2color": dict_value,
                "text2font": dict_value,
                "text2gradient": dict_value,
                "text2slant": dict_value,
                "text2weight": dict_value,
                "lsh": _num(),
                "t2c": dict_value,
                "t2f": dict_value,
                "t2g": dict_value,
                "t2s": dict_value,
                "t2w": dict_value,
                "global_config": dict_value,
                "local_configs": dict_value,
                "disable_ligatures": _bool(),
                "isolate": value,
            },
        ),
        required=["text"],
    )
    _register(
        controller,
        "Text",
        properties=_merge_props(
            string_kwargs,
            {
                "text": _str(),
                "isolate": value,
                "use_labelled_svg": _bool(),
                "path_string_config": dict_value,
                "font_size": _int(),
                "font": _str(),
                "slant": _str(),
                "weight": _str(),
            },
        ),
        required=["text"],
    )
    _register(
        controller,
        "Code",
        properties=_merge_props(
            string_kwargs,
            {
                "code": _str(),
                "font": _str(),
                "font_size": _int(),
                "lsh": _num(),
                "fill_color": value,
                "stroke_color": value,
                "language": _str(),
                "code_style": _str(),
            },
        ),
        required=["code"],
    )

    _register(
        controller,
        "TypstMobject",
        properties=_merge_props(
            typst_kwargs, {"typst_strings": str_array}
        ),
        positional=["typst_strings"],
        spread=["typst_strings"],
        required=[],
    )
    _register(
        controller,
        "TypstTextMobject",
        properties=_merge_props(
            typst_kwargs,
            {
                "text": _str(),
            },
        ),
        required=["text"],
    )
    _register(
        controller,
        "MarkdownMobject",
        properties=_merge_props(
            typst_kwargs,
            {
                "markdown": _str(),
            },
        ),
        required=["markdown"],
    )

    _register(
        controller,
        "SingleStringTex",
        properties=_merge_props(
            svg_kwargs,
            {
                "tex_string": _str(),
                "height": _num(),
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_width": _num(),
                "svg_default": dict_value,
                "path_string_config": dict_value,
                "font_size": _int(),
                "alignment": _str(),
                "math_mode": _bool(),
                "organize_left_to_right": _bool(),
                "template": _str(),
                "additional_preamble": _str(),
            },
        ),
        required=["tex_string"],
    )
    _register(
        controller,
        "OldTex",
        properties=_merge_props(
            svg_kwargs,
            {
                "tex_strings": str_array,
                "arg_separator": _str(),
                "isolate": str_array,
                "tex_to_color_map": dict_value,
                "tex_string": _str(),
                "height": _num(),
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_width": _num(),
                "svg_default": dict_value,
                "path_string_config": dict_value,
                "font_size": _int(),
                "alignment": _str(),
                "math_mode": _bool(),
                "organize_left_to_right": _bool(),
                "template": _str(),
                "additional_preamble": _str(),
            },
        ),
        positional=["tex_strings"],
        spread=["tex_strings"],
        required=[],
    )
    _register(
        controller,
        "OldTexText",
        properties=_merge_props(
            svg_kwargs,
            {
                "tex_strings": str_array,
                "math_mode": _bool(),
                "arg_separator": _str(),
                "isolate": str_array,
                "tex_to_color_map": dict_value,
                "tex_string": _str(),
                "height": _num(),
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_width": _num(),
                "svg_default": dict_value,
                "path_string_config": dict_value,
                "font_size": _int(),
                "alignment": _str(),
                "organize_left_to_right": _bool(),
                "template": _str(),
                "additional_preamble": _str(),
            },
        ),
        positional=["tex_strings"],
        spread=["tex_strings"],
        required=[],
    )

    _register(
        controller,
        "BulletedList",
        properties=_merge_props(
            vmobject_kwargs,
            tex_kwargs,
            {
                "items": str_array,
                "buff": _num(),
                "aligned_edge": vect_or_ref,
                "numbered": _bool(),
            },
        ),
        positional=["items"],
        spread=["items"],
        required=[],
    )
    _register(
        controller,
        "TexTextFromPresetString",
        properties=tex_kwargs,
    )
    _register(
        controller,
        "Title",
        properties=_merge_props(
            tex_kwargs,
            {
                "text_parts": str_array,
                "font_size": _int(),
                "include_underline": _bool(),
                "underline_width": _num(),
                "match_underline_width_to_text": _bool(),
                "underline_buff": _num(),
                "underline_style": dict_value,
            },
        ),
        positional=["text_parts"],
        spread=["text_parts"],
        required=[],
    )

    _register(
        controller,
        "Brace",
        properties=_merge_props(
            tex_kwargs,
            {
                "mobject": ref,
                "direction": vect_or_ref,
                "buff": _num(),
                "tex_string": _str(),
            },
        ),
        required=["mobject"],
    )
    _register(
        controller,
        "BraceLabel",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "obj": value,
                "text": value,
                "brace_direction": vect_or_ref,
                "label_scale": _num(),
                "label_buff": _num(),
            },
        ),
        required=["obj", "text"],
    )
    _register(
        controller,
        "BraceText",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "obj": value,
                "text": value,
                "brace_direction": vect_or_ref,
                "label_scale": _num(),
                "label_buff": _num(),
            },
        ),
        required=["obj", "text"],
    )
    _register(
        controller,
        "LineBrace",
        properties=_merge_props(
            tex_kwargs,
            {
                "line": ref,
                "direction": vect_or_ref,
                "buff": _num(),
                "tex_string": _str(),
            },
        ),
        required=["line"],
    )

    # Drawing mobjects
    _register(controller, "Checkmark", properties=tex_kwargs)
    _register(controller, "Exmark", properties=tex_kwargs)
    _register(
        controller,
        "Lightbulb",
        properties=_merge_props(
            svg_kwargs,
            {
                "height": _num(),
                "color": value,
                "stroke_width": _num(),
                "fill_opacity": _num(),
            },
        ),
    )
    _register(
        controller,
        "Speedometer",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "arc_angle": _num(),
                "num_ticks": _int(),
                "tick_length": _num(),
                "needle_width": _num(),
                "needle_height": _num(),
                "needle_color": value,
            },
        ),
    )
    _register(
        controller,
        "Laptop",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "width": _num(),
                "body_dimensions": _array_of(
                    _num(), min_items=3, max_items=3
                ),
                "screen_thickness": _num(),
                "keyboard_width_to_body_width": _num(),
                "keyboard_height_to_body_height": _num(),
                "screen_width_to_screen_plate_width": _num(),
                "key_color_kwargs": dict_value,
                "fill_opacity": _num(),
                "stroke_width": _num(),
                "body_color": value,
                "shaded_body_color": value,
                "open_angle": _num(),
            },
        ),
    )
    _register(
        controller,
        "VideoIcon",
        properties=_merge_props(
            svg_kwargs,
            {
                "width": _num(),
                "color": value,
            },
        ),
    )
    _register(
        controller,
        "VideoSeries",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "num_videos": _int(),
                "gradient_colors": _array_of(value),
                "width": _num(),
            },
        ),
    )
    _register(
        controller,
        "Clock",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "stroke_color": value,
                "stroke_width": _num(),
                "hour_hand_height": _num(),
                "minute_hand_height": _num(),
                "tick_length": _num(),
            },
        ),
    )

    _register(
        controller,
        "Bubble",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "content": value,
                "buff": _num(),
                "filler_shape": _array_of(
                    _num(), min_items=2, max_items=2
                ),
                "pin_point": vect_or_ref,
                "direction": vect_or_ref,
                "add_content": _bool(),
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_color": value,
                "stroke_width": _num(),
            },
        ),
    )
    _register(
        controller,
        "SpeechBubble",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "content": value,
                "buff": _num(),
                "filler_shape": _array_of(
                    _num(), min_items=2, max_items=2
                ),
                "stem_height_to_bubble_height": _num(),
                "stem_top_x_props": _array_of(
                    _num(), min_items=2, max_items=2
                ),
                "pin_point": vect_or_ref,
                "direction": vect_or_ref,
                "add_content": _bool(),
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_color": value,
                "stroke_width": _num(),
            },
        ),
    )
    _register(
        controller,
        "ThoughtBubble",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "content": value,
                "buff": _num(),
                "filler_shape": _array_of(
                    _num(), min_items=2, max_items=2
                ),
                "bulge_radius": _num(),
                "bulge_overlap": _num(),
                "noise_factor": _num(),
                "circle_radii": _array_of(_num()),
                "pin_point": vect_or_ref,
                "direction": vect_or_ref,
                "add_content": _bool(),
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_color": value,
                "stroke_width": _num(),
            },
        ),
    )
    _register(
        controller,
        "OldSpeechBubble",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "content": value,
                "buff": _num(),
                "filler_shape": _array_of(
                    _num(), min_items=2, max_items=2
                ),
                "pin_point": vect_or_ref,
                "direction": vect_or_ref,
                "add_content": _bool(),
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_color": value,
                "stroke_width": _num(),
            },
        ),
    )
    _register(
        controller,
        "DoubleSpeechBubble",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "content": value,
                "buff": _num(),
                "filler_shape": _array_of(
                    _num(), min_items=2, max_items=2
                ),
                "pin_point": vect_or_ref,
                "direction": vect_or_ref,
                "add_content": _bool(),
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_color": value,
                "stroke_width": _num(),
            },
        ),
    )
    _register(
        controller,
        "OldThoughtBubble",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "content": value,
                "buff": _num(),
                "filler_shape": _array_of(
                    _num(), min_items=2, max_items=2
                ),
                "pin_point": vect_or_ref,
                "direction": vect_or_ref,
                "add_content": _bool(),
                "fill_color": value,
                "fill_opacity": _num(),
                "stroke_color": value,
                "stroke_width": _num(),
            },
        ),
    )

    _register(
        controller,
        "VectorizedEarth",
        properties=_merge_props(svg_kwargs, {"height": _num()}),
    )
    _register(
        controller,
        "Piano",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "n_white_keys": _int(),
                "black_pattern": _array_of(_int()),
                "white_keys_per_octave": _int(),
                "white_key_dims": _array_of(
                    _num(), min_items=2, max_items=2
                ),
                "black_key_dims": _array_of(
                    _num(), min_items=2, max_items=2
                ),
                "key_buff": _num(),
                "white_key_color": value,
                "black_key_color": value,
                "total_width": _num(),
            },
        ),
    )
    _register(
        controller,
        "Piano3D",
        properties=_merge_props(
            vmobject_kwargs,
            {
                "stroke_width": _num(),
                "stroke_color": value,
                "key_depth": _num(),
                "black_key_shift": _num(),
                "piano_2d_config": dict_value,
            },
        ),
    )
    _register(
        controller,
        "DieFace",
        properties={
            "value": _int(),
            "side_length": _num(),
            "corner_radius": _num(),
            "stroke_color": value,
            "stroke_width": _num(),
            "fill_color": value,
            "dot_radius": _num(),
            "dot_color": value,
            "dot_coalesce_factor": _num(),
        },
        required=["value"],
    )
    _register(
        controller,
        "Dartboard",
        properties=vmobject_kwargs,
    )


def init_all_animations(controller: LLMSceneController) -> None:
    """Register all animation builders and schemas for LLM actions v2."""
    value = controller.schema_value()
    ref = controller.schema_ref()

    vect = _vector3()
    vect_or_ref = _any_of(vect, ref)
    vect_or_ref_or_null = _any_of(vect, ref, {"type": "null"})
    nonempty_ref_array = _array_of(ref, min_items=1)
    str_array = _array_of(_str())
    dict_value = _dict_of(value)
    pair_refs = _array_of(_array_of(ref, min_items=2, max_items=2))

    anim_spec = _anim_spec_schema(value)
    anim_spec_array = _array_of(anim_spec, min_items=1)
    animate_method_step_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "method": _str(),
            "params": {
                "type": "object",
                "additionalProperties": value,
            },
        },
        "required": ["method", "params"],
    }
    animate_method_step_array = _array_of(
        animate_method_step_schema,
        min_items=1,
    )

    animation_kwargs = {
        "run_time": _num(),
        "time_span": _array_of(_num(), min_items=2, max_items=2),
        "lag_ratio": _num(),
        "rate_func": _py_function_expr(
            "(t) -> float",
            "lambda t: smooth(t)",
        ),
        "name": _str(),
        "remover": _bool(),
        "final_alpha_value": _num(),
        "suspend_mobject_updating": _bool(),
    }

    transform_kwargs = _merge_props(
        animation_kwargs,
        {
            "path_arc": _any_of(
                _num(),
                _array_of(_num(), min_items=2, max_items=2),
            ),
            "path_arc_axis": vect_or_ref,
            "path_func": _py_function_expr(
                "(start, end, alpha) -> np.ndarray",
                "lambda start, end, alpha: (1 - alpha) * start + alpha * end",
            ),
        },
    )

    # Base animation types
    _register_animation(
        controller,
        "Animation",
        properties=_merge_props(
            animation_kwargs,
            {
                "mobject": ref,
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "AnimateMethods",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
                "methods": animate_method_step_array,
            },
        ),
        required=["mobject", "methods"],
        builder=_build_animate_methods,
    )

    # Composition
    _register_animation(
        controller,
        "AnimationGroup",
        properties=_merge_props(
            animation_kwargs,
            {
                "animations": anim_spec_array,
                "group": ref,
                "group_type": ref,
            },
        ),
        required=["animations"],
        builder=_build_animation_group,
    )
    _register_animation(
        controller,
        "Succession",
        properties=_merge_props(
            animation_kwargs,
            {
                "animations": anim_spec_array,
                "lag_ratio": _num(),
            },
        ),
        required=["animations"],
        builder=_build_succession,
    )
    _register_animation(
        controller,
        "LaggedStart",
        properties=_merge_props(
            animation_kwargs,
            {
                "animations": anim_spec_array,
                "lag_ratio": _num(),
            },
        ),
        required=["animations"],
        builder=_build_lagged_start,
    )
    _register_animation(
        controller,
        "LaggedStartMap",
        properties=_merge_props(
            animation_kwargs,
            {
                "anim_class": _str(),
                "group": ref,
                "anim_params": dict_value,
                "run_time": _num(),
                "lag_ratio": _num(),
            },
        ),
        required=["anim_class", "group"],
        builder=_build_lagged_start_map,
    )

    # Creation
    _register_animation(
        controller,
        "ShowCreation",
        properties=_merge_props(
            animation_kwargs,
            {
                "mobject": ref,
                "lag_ratio": _num(),
                "should_match_start": _bool(),
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "Uncreate",
        properties=_merge_props(
            animation_kwargs,
            {
                "mobject": ref,
                "should_match_start": _bool(),
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "DrawBorderThenFill",
        properties=_merge_props(
            animation_kwargs,
            {
                "vmobject": ref,
                "stroke_width": _num(),
                "stroke_color": value,
                "draw_border_animation_config": dict_value,
                "fill_animation_config": dict_value,
            },
        ),
        required=["vmobject"],
    )
    _register_animation(
        controller,
        "Write",
        properties=_merge_props(
            animation_kwargs,
            {
                "vmobject": ref,
                "stroke_color": value,
            },
        ),
        required=["vmobject"],
    )
    _register_animation(
        controller,
        "ShowIncreasingSubsets",
        properties=_merge_props(
            animation_kwargs,
            {
                "group": ref,
                "int_func": _py_function_expr(
                    "(x) -> float",
                    "lambda x: np.round(x)",
                ),
                "suspend_mobject_updating": _bool(),
            },
        ),
        required=["group"],
    )
    _register_animation(
        controller,
        "ShowSubmobjectsOneByOne",
        properties=_merge_props(
            animation_kwargs,
            {
                "group": ref,
                "int_func": _py_function_expr(
                    "(x) -> float",
                    "lambda x: np.ceil(x)",
                ),
            },
        ),
        required=["group"],
    )
    _register_animation(
        controller,
        "AddTextWordByWord",
        properties=_merge_props(
            animation_kwargs,
            {
                "string_mobject": ref,
                "time_per_word": _num(),
            },
        ),
        required=["string_mobject"],
    )

    # Fading
    fade_props = _merge_props(
        transform_kwargs,
        {
            "mobject": ref,
            "shift": vect_or_ref,
            "scale": _num(),
        },
    )
    _register_animation(
        controller,
        "Fade",
        properties=fade_props,
        required=["mobject"],
    )
    _register_animation(
        controller,
        "FadeIn",
        properties=fade_props,
        required=["mobject"],
    )
    _register_animation(
        controller,
        "FadeOut",
        properties=fade_props,
        required=["mobject"],
    )
    _register_animation(
        controller,
        "FadeInFromPoint",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
                "point": vect_or_ref,
            },
        ),
        required=["mobject", "point"],
    )
    _register_animation(
        controller,
        "FadeOutToPoint",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
                "point": vect_or_ref,
            },
        ),
        required=["mobject", "point"],
    )
    fade_transform_props = _merge_props(
        transform_kwargs,
        {
            "mobject": ref,
            "target_mobject": ref,
            "stretch": _bool(),
            "dim_to_match": _int(),
        },
    )
    _register_animation(
        controller,
        "FadeTransform",
        properties=fade_transform_props,
        required=["mobject", "target_mobject"],
    )
    _register_animation(
        controller,
        "FadeTransformPieces",
        properties=fade_transform_props,
        required=["mobject", "target_mobject"],
    )
    _register_animation(
        controller,
        "VFadeIn",
        properties=_merge_props(
            animation_kwargs,
            {
                "vmobject": ref,
                "suspend_mobject_updating": _bool(),
            },
        ),
        required=["vmobject"],
    )
    _register_animation(
        controller,
        "VFadeOut",
        properties=_merge_props(
            animation_kwargs,
            {
                "vmobject": ref,
            },
        ),
        required=["vmobject"],
    )
    _register_animation(
        controller,
        "VFadeInThenOut",
        properties=_merge_props(
            animation_kwargs,
            {
                "vmobject": ref,
                "rate_func": _py_function_expr(
                    "(t) -> float",
                    "lambda t: there_and_back(t)",
                ),
            },
        ),
        required=["vmobject"],
    )

    # Growing
    _register_animation(
        controller,
        "GrowFromPoint",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
                "point": vect_or_ref,
                "point_color": value,
            },
        ),
        required=["mobject", "point"],
    )
    _register_animation(
        controller,
        "GrowFromCenter",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "GrowFromEdge",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
                "edge": vect_or_ref,
            },
        ),
        required=["mobject", "edge"],
    )
    _register_animation(
        controller,
        "GrowArrow",
        properties=_merge_props(
            transform_kwargs,
            {
                "arrow": ref,
            },
        ),
        required=["arrow"],
    )

    # Indication
    _register_animation(
        controller,
        "FocusOn",
        properties=_merge_props(
            transform_kwargs,
            {
                "focus_point": _any_of(vect, ref),
                "opacity": _num(),
                "color": value,
            },
        ),
        required=["focus_point"],
    )
    _register_animation(
        controller,
        "Indicate",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
                "scale_factor": _num(),
                "color": value,
                "rate_func": _py_function_expr(
                    "(t) -> float",
                    "lambda t: there_and_back(t)",
                ),
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "Flash",
        properties=_merge_props(
            animation_kwargs,
            {
                "point": _any_of(vect, ref),
                "color": value,
                "line_length": _num(),
                "num_lines": _int(),
                "flash_radius": _num(),
                "line_stroke_width": _num(),
            },
        ),
        required=["point"],
    )
    _register_animation(
        controller,
        "CircleIndicate",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
                "scale_factor": _num(),
                "rate_func": _py_function_expr(
                    "(t) -> float",
                    "lambda t: there_and_back(t)",
                ),
                "stroke_color": value,
                "stroke_width": _num(),
                "remover": _bool(),
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "ShowPassingFlash",
        properties=_merge_props(
            animation_kwargs,
            {
                "mobject": ref,
                "time_width": _num(),
                "remover": _bool(),
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "VShowPassingFlash",
        properties=_merge_props(
            animation_kwargs,
            {
                "vmobject": ref,
                "time_width": _num(),
                "taper_width": _num(),
                "remover": _bool(),
            },
        ),
        required=["vmobject"],
    )
    flash_around_props = _merge_props(
        animation_kwargs,
        {
            "mobject": ref,
            "time_width": _num(),
            "taper_width": _num(),
            "stroke_width": _num(),
            "color": value,
            "buff": _num(),
            "n_inserted_curves": _int(),
        },
    )
    _register_animation(
        controller,
        "FlashAround",
        properties=flash_around_props,
        required=["mobject"],
    )
    _register_animation(
        controller,
        "FlashUnder",
        properties=flash_around_props,
        required=["mobject"],
    )
    _register_animation(
        controller,
        "ShowCreationThenDestruction",
        properties=_merge_props(
            animation_kwargs,
            {
                "vmobject": ref,
                "time_width": _num(),
            },
        ),
        required=["vmobject"],
    )
    _register_animation(
        controller,
        "ShowCreationThenFadeOut",
        properties=_merge_props(
            animation_kwargs,
            {
                "mobject": ref,
                "remover": _bool(),
            },
        ),
        required=["mobject"],
    )
    surrounding_props = _merge_props(
        animation_kwargs,
        {
            "mobject": ref,
            "stroke_width": _num(),
            "stroke_color": value,
            "buff": _num(),
        },
    )
    _register_animation(
        controller,
        "AnimationOnSurroundingRectangle",
        properties=surrounding_props,
        required=["mobject"],
    )
    _register_animation(
        controller,
        "ShowPassingFlashAround",
        properties=surrounding_props,
        required=["mobject"],
    )
    _register_animation(
        controller,
        "ShowCreationThenDestructionAround",
        properties=surrounding_props,
        required=["mobject"],
    )
    _register_animation(
        controller,
        "ShowCreationThenFadeAround",
        properties=surrounding_props,
        required=["mobject"],
    )
    _register_animation(
        controller,
        "ApplyWave",
        properties=_merge_props(
            animation_kwargs,
            {
                "mobject": ref,
                "direction": vect_or_ref,
                "amplitude": _num(),
                "run_time": _num(),
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "WiggleOutThenIn",
        properties=_merge_props(
            animation_kwargs,
            {
                "mobject": ref,
                "scale_value": _num(),
                "rotation_angle": _num(),
                "n_wiggles": _int(),
                "scale_about_point": vect_or_ref_or_null,
                "rotate_about_point": vect_or_ref_or_null,
                "run_time": _num(),
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "TurnInsideOut",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
                "path_arc": _num(),
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "FlashyFadeIn",
        properties=_merge_props(
            animation_kwargs,
            {
                "vmobject": ref,
                "stroke_width": _num(),
                "fade_lag": _num(),
                "time_width": _num(),
            },
        ),
        required=["vmobject"],
    )

    # Movement
    _register_animation(
        controller,
        "Homotopy",
        properties=_merge_props(
            animation_kwargs,
            {
                "homotopy": _py_function_expr(
                    "(x, y, z, t) -> sequence[float]",
                    "lambda x, y, z, t: np.array([x, y, z])",
                ),
                "mobject": ref,
                "run_time": _num(),
            },
        ),
        required=["homotopy", "mobject"],
    )
    _register_animation(
        controller,
        "SmoothedVectorizedHomotopy",
        properties=_merge_props(
            animation_kwargs,
            {
                "homotopy": _py_function_expr(
                    "(x, y, z, t) -> sequence[float]",
                    "lambda x, y, z, t: np.array([x, y, z])",
                ),
                "mobject": ref,
                "run_time": _num(),
            },
        ),
        required=["homotopy", "mobject"],
    )
    _register_animation(
        controller,
        "ComplexHomotopy",
        properties=_merge_props(
            animation_kwargs,
            {
                "complex_homotopy": _py_function_expr(
                    "(z, t) -> complex",
                    "lambda z, t: z",
                ),
                "mobject": ref,
            },
        ),
        required=["complex_homotopy", "mobject"],
    )
    _register_animation(
        controller,
        "PhaseFlow",
        properties=_merge_props(
            animation_kwargs,
            {
                "function": _py_function_expr(
                    "(p) -> np.ndarray",
                    "lambda p: np.array([p[0], p[1], p[2]])",
                ),
                "mobject": ref,
                "virtual_time": _num(),
                "suspend_mobject_updating": _bool(),
                "rate_func": _py_function_expr(
                    "(t) -> float",
                    "lambda t: linear(t)",
                ),
                "run_time": _num(),
            },
        ),
        required=["function", "mobject"],
    )
    _register_animation(
        controller,
        "MoveAlongPath",
        properties=_merge_props(
            animation_kwargs,
            {
                "mobject": ref,
                "path": ref,
                "suspend_mobject_updating": _bool(),
            },
        ),
        required=["mobject", "path"],
    )

    # Numbers
    _register_animation(
        controller,
        "ChangingDecimal",
        properties=_merge_props(
            animation_kwargs,
            {
                "decimal_mob": ref,
                "number_update_func": _py_function_expr(
                    "(alpha) -> float | complex",
                    "lambda alpha: alpha",
                ),
                "suspend_mobject_updating": _bool(),
            },
        ),
        required=["decimal_mob", "number_update_func"],
    )
    _register_animation(
        controller,
        "ChangeDecimalToValue",
        properties=_merge_props(
            animation_kwargs,
            {
                "decimal_mob": ref,
                "target_number": value,
            },
        ),
        required=["decimal_mob", "target_number"],
    )
    _register_animation(
        controller,
        "CountInFrom",
        properties=_merge_props(
            animation_kwargs,
            {
                "decimal_mob": ref,
                "source_number": value,
            },
        ),
        required=["decimal_mob"],
    )

    # Rotation
    rotating_props = _merge_props(
        animation_kwargs,
        {
            "mobject": ref,
            "angle": _num(),
            "axis": vect_or_ref,
            "about_point": vect_or_ref_or_null,
            "about_edge": vect_or_ref_or_null,
            "run_time": _num(),
            "rate_func": _py_function_expr(
                "(t) -> float",
                "lambda t: linear(t)",
            ),
            "suspend_mobject_updating": _bool(),
        },
    )
    _register_animation(
        controller,
        "Rotating",
        properties=rotating_props,
        required=["mobject"],
    )
    _register_animation(
        controller,
        "Rotate",
        properties=rotating_props,
        required=["mobject"],
    )

    # Specialized
    _register_animation(
        controller,
        "Broadcast",
        properties=_merge_props(
            animation_kwargs,
            {
                "focal_point": vect_or_ref,
                "small_radius": _num(),
                "big_radius": _num(),
                "n_circles": _int(),
                "start_stroke_width": _num(),
                "color": value,
                "run_time": _num(),
                "lag_ratio": _num(),
                "remover": _bool(),
            },
        ),
        required=["focal_point"],
    )

    # Transform
    transform_props = _merge_props(
        transform_kwargs,
        {
            "mobject": ref,
            "target_mobject": ref,
        },
    )
    _register_animation(
        controller,
        "Transform",
        properties=transform_props,
        required=["mobject"],
    )
    _register_animation(
        controller,
        "ReplacementTransform",
        properties=transform_props,
        required=["mobject", "target_mobject"],
    )
    _register_animation(
        controller,
        "TransformFromCopy",
        properties=transform_props,
        required=["mobject", "target_mobject"],
    )
    _register_animation(
        controller,
        "MoveToTarget",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "ApplyMethod",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
                "method": _str(),
                "args": _array_of(value),
                "method_kwargs": dict_value,
            },
        ),
        required=["mobject", "method"],
        builder=_build_apply_method,
    )
    _register_animation(
        controller,
        "ApplyPointwiseFunction",
        properties=_merge_props(
            transform_kwargs,
            {
                "function": _py_function_expr(
                    "(p) -> np.ndarray",
                    "lambda p: p",
                ),
                "mobject": ref,
                "run_time": _num(),
            },
        ),
        required=["function", "mobject"],
    )
    _register_animation(
        controller,
        "ApplyPointwiseFunctionToCenter",
        properties=_merge_props(
            transform_kwargs,
            {
                "function": _py_function_expr(
                    "(p) -> np.ndarray",
                    "lambda p: p",
                ),
                "mobject": ref,
            },
        ),
        required=["function", "mobject"],
    )
    _register_animation(
        controller,
        "FadeToColor",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
                "color": value,
            },
        ),
        required=["mobject", "color"],
    )
    _register_animation(
        controller,
        "ScaleInPlace",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
                "scale_factor": value,
            },
        ),
        required=["mobject", "scale_factor"],
    )
    _register_animation(
        controller,
        "ShrinkToCenter",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "Restore",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobject": ref,
            },
        ),
        required=["mobject"],
    )
    _register_animation(
        controller,
        "ApplyFunction",
        properties=_merge_props(
            transform_kwargs,
            {
                "function": _py_function_expr(
                    "(mobject) -> Mobject",
                    "lambda mob: mob",
                ),
                "mobject": ref,
            },
        ),
        required=["function", "mobject"],
    )
    _register_animation(
        controller,
        "ApplyMatrix",
        properties=_merge_props(
            transform_kwargs,
            {
                "matrix": value,
                "mobject": ref,
            },
        ),
        required=["matrix", "mobject"],
        builder=_build_apply_matrix,
    )
    _register_animation(
        controller,
        "ApplyComplexFunction",
        properties=_merge_props(
            transform_kwargs,
            {
                "function": _py_function_expr(
                    "(z) -> complex",
                    "lambda z: z",
                ),
                "mobject": ref,
            },
        ),
        required=["function", "mobject"],
    )
    _register_animation(
        controller,
        "CyclicReplace",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobjects": nonempty_ref_array,
                "path_arc": _num(),
            },
        ),
        positional=["mobjects"],
        spread=["mobjects"],
        required=["mobjects"],
    )
    _register_animation(
        controller,
        "Swap",
        properties=_merge_props(
            transform_kwargs,
            {
                "mobjects": nonempty_ref_array,
                "path_arc": _num(),
            },
        ),
        positional=["mobjects"],
        spread=["mobjects"],
        required=["mobjects"],
    )

    # Transform matching
    matching_parts_props = _merge_props(
        animation_kwargs,
        {
            "source": ref,
            "target": ref,
            "matched_pairs": pair_refs,
            "match_animation": ref,
            "mismatch_animation": ref,
            "run_time": _num(),
            "lag_ratio": _num(),
        },
    )
    _register_animation(
        controller,
        "TransformMatchingParts",
        properties=matching_parts_props,
        required=["source", "target"],
    )
    _register_animation(
        controller,
        "TransformMatchingShapes",
        properties=matching_parts_props,
        required=["source", "target"],
    )
    matching_strings_props = _merge_props(
        matching_parts_props,
        {
            "matched_keys": str_array,
            "key_map": _dict_of(_str()),
        },
    )
    _register_animation(
        controller,
        "TransformMatchingStrings",
        properties=matching_strings_props,
        required=["source", "target"],
    )
    _register_animation(
        controller,
        "TransformMatchingTex",
        properties=matching_strings_props,
        required=["source", "target"],
    )

    # Update
    _register_animation(
        controller,
        "UpdateFromFunc",
        properties=_merge_props(
            animation_kwargs,
            {
                "mobject": ref,
                "update_function": _py_function_expr(
                    "(mobject) -> Mobject | None",
                    "lambda mob: mob",
                ),
                "suspend_mobject_updating": _bool(),
            },
        ),
        required=["mobject", "update_function"],
    )
    _register_animation(
        controller,
        "UpdateFromAlphaFunc",
        properties=_merge_props(
            animation_kwargs,
            {
                "mobject": ref,
                "update_function": _py_function_expr(
                    "(mobject, alpha) -> Mobject | None",
                    "lambda mob, alpha: mob",
                ),
                "suspend_mobject_updating": _bool(),
            },
        ),
        required=["mobject", "update_function"],
    )
    _register_animation(
        controller,
        "MaintainPositionRelativeTo",
        properties=_merge_props(
            animation_kwargs,
            {
                "mobject": ref,
                "tracked_mobject": ref,
            },
        ),
        required=["mobject", "tracked_mobject"],
    )


def init_all_mobject_methods(
    controller: LLMSceneController,
    include_private: bool = False,
) -> None:
    """Register typed method schemas for all registered mobject classes.

    This includes inherited methods from superclasses.
    """
    controller.init_all_mobject_methods(
        include_private=include_private
    )


def init_all_animation_methods(
    controller: LLMSceneController,
    include_private: bool = False,
) -> None:
    """Register typed method schemas for all registered animation classes.

    This includes inherited methods from superclasses.
    """
    controller.init_all_animation_methods(
        include_private=include_private
    )
