import os
import re
import ast
import json
import copy
import traceback
import queue
import io
import contextlib
from dataclasses import dataclass
from typing import (
    Dict,
    Any,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
)
import types
import collections.abc as cabc
import openai
from openai import OpenAI
import manimlib
import inspect
import random
import math
import numpy as np
import sympy
import time


@dataclass(frozen=True)
class _LLMResponseMode:
    CODE: str = "code"
    ACTIONS: str = "actions"


@dataclass(frozen=True)
class _RegisteredCallableSchema:
    properties: Dict[str, dict]
    required: tuple[str, ...]
    positional: tuple[str, ...]
    spread: frozenset[str]
    allow_extra: bool
    callable_params: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _RegisteredActionSchema:
    properties: Dict[str, dict]
    required: tuple[str, ...]
    allow_extra: bool
    builder: Any


@dataclass(frozen=True)
class _BuilderContext:
    scene: Any
    registered_objects: Dict[str, Any]
    manimlib: Any
    controller: Any


class LLMSceneController:
    """
    Controller that connects a Language Model (LLM) with an interactive ManimGL scene.
    Allows registering objects and running prompts that generate and execute code in real-time.
    Implements a thread-safe queue to prevent OpenGL errors in background threads.
    """

    def __init__(
        self,
        scene: manimlib.InteractiveScene,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.scene = scene
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model

        # The OpenAI SDK handles base URLs compatible with Groq and other OSS.
        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url
        )

        # Dictionary to store Mobjects or variables the LLM can manipulate
        self.registered_objects: Dict[str, Any] = {}

        # Optional strict schema registries (used by actions mode v2)
        self._mobject_schemas: Dict[
            str, _RegisteredCallableSchema
        ] = {}
        self._animation_schemas: Dict[
            str, _RegisteredCallableSchema
        ] = {}

        # Optional builders for v2 (allows custom objects not in manimlib)
        # Keys match the string used in actions payloads ("class").
        self._mobject_builders: Dict[str, Any] = {}
        self._animation_builders: Dict[str, Any] = {}

        # Optional strict schema registries for method calls in actions mode v2.
        self._mobject_method_schemas: Dict[
            str, Dict[str, _RegisteredCallableSchema]
        ] = {}
        self._animation_method_schemas: Dict[
            str, Dict[str, _RegisteredCallableSchema]
        ] = {}

        # Optional custom action schemas/builders for actions mode v2.
        self._custom_action_schemas: Dict[
            str, _RegisteredActionSchema
        ] = {}

        # Queue to send code from the background thread to the main thread
        self.execution_queue = queue.Queue()
        self._is_processing_queue_item: bool = False

        # Prevent worker threads from waiting forever if queue processing fails.
        self.execution_result_timeout_seconds: float = 90.0

        # Persistent chat history by response mode/version.
        self._chat_histories: Dict[str, list[dict[str, str]]] = {}
        self.max_history_messages: Optional[int] = 40

        # Register queue processing on the scene main loop.
        self._install_main_loop_queue_hook()

    def _install_main_loop_queue_hook(self) -> None:
        add_callback = getattr(
            self.scene, "add_main_loop_callback", None
        )
        if not callable(add_callback):
            raise RuntimeError(
                "Scene does not expose add_main_loop_callback"
            )
        add_callback(self._process_queue)

    def register_object(self, name: str, obj: Any) -> None:
        """
        Registers an object in the LLM's execution environment so it can manipulate it.
        """
        self.registered_objects[name] = obj

    def _history_key(
        self,
        response_mode: str,
        actions_version: Optional[int] = None,
    ) -> str:
        if response_mode == _LLMResponseMode.ACTIONS:
            version = (
                actions_version if actions_version is not None else 1
            )
            return f"{response_mode}_v{version}"
        return response_mode

    def _trim_history(
        self, history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        if self.max_history_messages is None:
            return history
        limit = int(self.max_history_messages)
        if limit <= 0:
            return []
        if len(history) <= limit:
            return history
        return history[-limit:]

    def clear_chat_history(
        self,
        response_mode: Optional[str] = None,
        actions_version: Optional[int] = None,
    ) -> None:
        """Clear persistent history for one mode/version or all histories."""
        if response_mode is None:
            self._chat_histories.clear()
            return
        key = self._history_key(response_mode, actions_version)
        self._chat_histories.pop(key, None)

    def _resolve_registered_class(
        self,
        class_name: str,
        *,
        builders: Dict[str, Any],
        base_type: type,
    ) -> Optional[type]:
        builder = builders.get(class_name)
        if inspect.isclass(builder) and issubclass(
            builder, base_type
        ):
            return builder
        fallback = getattr(manimlib, class_name, None)
        if inspect.isclass(fallback) and issubclass(
            fallback, base_type
        ):
            return fallback
        return None

    @staticmethod
    def _is_probably_callable_param_name(name: str) -> bool:
        if not isinstance(name, str):
            return False
        lowered = name.lower()
        if lowered in {
            "func",
            "function",
            "callback",
            "updater",
            "update_function",
            "update_func",
            "on_click",
            "callable",
        }:
            return True
        return (
            lowered.endswith("_func")
            or lowered.endswith("_function")
            or lowered.endswith("_callback")
        )

    def _is_direct_callable_annotation(self, annotation: Any) -> bool:
        if annotation is inspect.Signature.empty or annotation is Any:
            return False
        if isinstance(annotation, str):
            lowered = annotation.lower()
            return (
                "callable" in lowered
                or "updater" in lowered
                or "callback" in lowered
                or lowered.endswith("func")
            )

        origin = get_origin(annotation)
        if origin is cabc.Callable:
            return True
        if annotation is cabc.Callable:
            return True
        if annotation in {
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            types.BuiltinMethodType,
        }:
            return True
        return False

    def _is_callable_annotation(self, annotation: Any) -> bool:
        if self._is_direct_callable_annotation(annotation):
            return True

        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin in (types.UnionType,) or str(origin).endswith(
            "typing.Union"
        ):
            return any(
                self._is_callable_annotation(arg) for arg in args
            )
        return False

    def _build_strict_map_schema(self, value_schema: dict) -> dict:
        strict_value_schema = self._strictify_schema_for_response(
            value_schema
        )
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "key": {"type": "string"},
                            "value": strict_value_schema,
                        },
                        "required": ["key", "value"],
                    },
                }
            },
            "required": ["entries"],
        }

    def _strictify_schema_for_response(self, schema: dict) -> dict:
        if not isinstance(schema, dict) or not schema:
            return schema

        result = copy.deepcopy(schema)

        # Remove non-functional metadata to keep response_format compact.
        for meta_key in (
            "description",
            "title",
            "examples",
            "default",
        ):
            result.pop(meta_key, None)

        if "anyOf" in result and isinstance(result["anyOf"], list):
            result["anyOf"] = [
                self._strictify_schema_for_response(option)
                for option in result["anyOf"]
            ]
            return result

        if "oneOf" in result and isinstance(result["oneOf"], list):
            result["oneOf"] = [
                self._strictify_schema_for_response(option)
                for option in result["oneOf"]
            ]
            return result

        if (
            result.get("type") == "array"
            and isinstance(result.get("items"), dict)
        ):
            result["items"] = self._strictify_schema_for_response(
                result["items"]
            )
            return result

        is_object = result.get("type") == "object" or (
            "properties" in result
            or "additionalProperties" in result
        )
        if not is_object:
            return result

        properties = result.get("properties")
        if isinstance(properties, dict) and properties:
            strict_properties: dict[str, Any] = {}
            required_set = set(result.get("required", []))
            for key, prop_schema in properties.items():
                if isinstance(prop_schema, dict):
                    strict_prop = self._strictify_schema_for_response(
                        prop_schema
                    )
                    if key not in required_set:
                        strict_prop = self._make_nullable_schema(
                            strict_prop
                        )
                    strict_properties[key] = strict_prop
                else:
                    strict_properties[key] = prop_schema
            result["properties"] = strict_properties
            result["required"] = list(strict_properties.keys())
            result["additionalProperties"] = False
            return result

        additional = result.get("additionalProperties")
        if isinstance(additional, dict):
            return self._build_strict_map_schema(additional)

        result["properties"] = (
            result["properties"]
            if isinstance(result.get("properties"), dict)
            else {}
        )
        result["required"] = list(result["properties"].keys())
        result["additionalProperties"] = False
        return result

    @staticmethod
    def _sanitize_def_token(token: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", token or "")
        cleaned = cleaned.strip("_")
        return cleaned or "schema"

    @staticmethod
    def _schema_cache_key(schema: dict) -> str:
        return json.dumps(
            schema,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def _intern_schema_ref(
        self,
        defs: dict,
        schema_pool: dict[str, str],
        schema: dict,
        *,
        prefix: str,
    ) -> dict:
        if isinstance(schema, dict) and set(schema.keys()) == {"$ref"}:
            return schema

        key = self._schema_cache_key(schema)
        if key in schema_pool:
            return {"$ref": f"#/$defs/{schema_pool[key]}"}

        base_name = self._sanitize_def_token(prefix)
        name = base_name
        idx = 1
        while name in defs:
            idx += 1
            name = f"{base_name}_{idx}"

        defs[name] = schema
        schema_pool[key] = name
        return {"$ref": f"#/$defs/{name}"}

    def _annotation_to_schema(self, annotation: Any) -> dict:
        if annotation is inspect.Signature.empty or annotation is Any:
            return self.schema_value()

        if annotation is str:
            return {"type": "string"}
        if annotation is bool:
            return {"type": "boolean"}
        if annotation is int:
            return {"type": "integer"}
        if annotation in (float, complex):
            return {"type": "number"}
        if annotation is type(None):
            return {"type": "null"}

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin in (list, set, frozenset, tuple):
            item_schema = self.schema_value()
            if args:
                item_type = args[0]
                if origin is tuple and args[-1] is not Ellipsis:
                    item_type = args[0]
                item_schema = self._annotation_to_schema(item_type)
            return {
                "type": "array",
                "items": item_schema,
            }

        if origin is dict:
            value_schema = self.schema_value()
            if len(args) >= 2:
                value_schema = self._annotation_to_schema(args[1])
            return self._build_strict_map_schema(value_schema)

        if origin in (types.UnionType,):
            variants = [
                self._annotation_to_schema(arg) for arg in args
            ]
            return {"anyOf": variants}

        # typing.Optional / typing.Union in older-style annotations.
        if str(origin).endswith("typing.Union"):
            variants = [
                self._annotation_to_schema(arg) for arg in args
            ]
            return {"anyOf": variants}

        if self._is_direct_callable_annotation(annotation):
            return {
                "anyOf": [
                    {"type": "string"},
                    self.schema_ref(),
                ]
            }

        return self.schema_value()

    def _schema_from_callable_signature(
        self, callable_obj: Any
    ) -> Optional[_RegisteredCallableSchema]:
        try:
            sig = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return None

        try:
            resolved_hints = get_type_hints(callable_obj)
        except Exception:
            resolved_hints = {}

        properties: Dict[str, dict] = {}
        required: list[str] = []
        positional: list[str] = []
        spread: list[str] = []
        allow_extra = False

        for idx, param in enumerate(sig.parameters.values()):
            if (
                idx == 0
                and param.name in {"self", "cls"}
                and param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ):
                continue

            if param.kind == inspect.Parameter.VAR_KEYWORD:
                allow_extra = True
                continue

            annotation = resolved_hints.get(
                param.name, param.annotation
            )
            schema = self._annotation_to_schema(annotation)

            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                properties[param.name] = {
                    "type": "array",
                    "items": schema,
                }
                positional.append(param.name)
                spread.append(param.name)
                continue

            properties[param.name] = schema

            if param.default is inspect.Signature.empty:
                required.append(param.name)

            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional.append(param.name)

        callable_params = self._infer_callable_params_from_callable(
            callable_obj,
            sig=sig,
            resolved_hints=resolved_hints,
        )

        return self._build_registered_callable_schema(
            properties=properties,
            required=required,
            positional=positional,
            spread=spread,
            allow_extra=allow_extra,
            callable_params=callable_params,
            allow_empty_properties=True,
        )

    def _infer_callable_params_from_callable(
        self,
        callable_obj: Any,
        *,
        sig: Optional[inspect.Signature] = None,
        resolved_hints: Optional[Dict[str, Any]] = None,
    ) -> frozenset[str]:
        if sig is None:
            try:
                sig = inspect.signature(callable_obj)
            except (TypeError, ValueError):
                return frozenset()

        if resolved_hints is None:
            try:
                resolved_hints = get_type_hints(callable_obj)
            except Exception:
                resolved_hints = {}

        callable_params: set[str] = set()
        for idx, param in enumerate(sig.parameters.values()):
            if (
                idx == 0
                and param.name in {"self", "cls"}
                and param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ):
                continue

            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            annotation = resolved_hints.get(
                param.name, param.annotation
            )
            if self._is_callable_annotation(annotation) or (
                annotation in (inspect.Signature.empty, Any)
                and self._is_probably_callable_param_name(param.name)
            ):
                callable_params.add(param.name)

        return frozenset(callable_params)

    def _register_methods_for_class(
        self,
        *,
        registry: Dict[str, Dict[str, _RegisteredCallableSchema]],
        class_name: str,
        cls: type,
        include_private: bool,
    ) -> None:
        class_registry = registry.setdefault(class_name, {})

        for method_name, method_obj in inspect.getmembers(
            cls, predicate=callable
        ):
            if method_name in class_registry:
                continue
            if not include_private and method_name.startswith("_"):
                continue

            schema = self._schema_from_callable_signature(method_obj)
            if schema is None:
                continue
            class_registry[method_name] = schema

    def init_all_mobject_methods(
        self, include_private: bool = False
    ) -> None:
        """Register method schemas for all currently registered mobject classes."""
        for class_name in sorted(self._mobject_schemas.keys()):
            cls = self._resolve_registered_class(
                class_name,
                builders=self._mobject_builders,
                base_type=manimlib.Mobject,
            )
            if cls is None:
                continue
            self._register_methods_for_class(
                registry=self._mobject_method_schemas,
                class_name=class_name,
                cls=cls,
                include_private=include_private,
            )

    def init_all_animation_methods(
        self, include_private: bool = False
    ) -> None:
        """Register method schemas for all currently registered animation classes."""
        for class_name in sorted(self._animation_schemas.keys()):
            cls = self._resolve_registered_class(
                class_name,
                builders=self._animation_builders,
                base_type=manimlib.Animation,
            )
            if cls is None:
                continue
            self._register_methods_for_class(
                registry=self._animation_method_schemas,
                class_name=class_name,
                cls=cls,
                include_private=include_private,
            )

    def register_mobject_method_schema(
        self,
        class_name: str,
        method_name: str,
        *,
        properties: Dict[str, dict],
        required: Optional[list[str]] = None,
        positional: Optional[list[str]] = None,
        spread: Optional[list[str]] = None,
        allow_extra: bool = False,
    ) -> None:
        cls = self._resolve_registered_class(
            class_name,
            builders=self._mobject_builders,
            base_type=manimlib.Mobject,
        )
        if cls is None:
            raise ValueError(
                f"'{class_name}' is not a registered Mobject class"
            )
        method = getattr(cls, method_name, None)
        if method is None or not callable(method):
            raise ValueError(
                f"'{method_name}' is not a callable method of '{class_name}'"
            )

        self._register_method_schema(
            registry=self._mobject_method_schemas,
            class_name=class_name,
            method_name=method_name,
            method=method,
            properties=properties,
            required=required,
            positional=positional,
            spread=spread,
            allow_extra=allow_extra,
        )

    def register_animation_method_schema(
        self,
        class_name: str,
        method_name: str,
        *,
        properties: Dict[str, dict],
        required: Optional[list[str]] = None,
        positional: Optional[list[str]] = None,
        spread: Optional[list[str]] = None,
        allow_extra: bool = False,
    ) -> None:
        cls = self._resolve_registered_class(
            class_name,
            builders=self._animation_builders,
            base_type=manimlib.Animation,
        )
        if cls is None:
            raise ValueError(
                f"'{class_name}' is not a registered Animation class"
            )
        method = getattr(cls, method_name, None)
        if method is None or not callable(method):
            raise ValueError(
                f"'{method_name}' is not a callable method of '{class_name}'"
            )

        self._register_method_schema(
            registry=self._animation_method_schemas,
            class_name=class_name,
            method_name=method_name,
            method=method,
            properties=properties,
            required=required,
            positional=positional,
            spread=spread,
            allow_extra=allow_extra,
        )

    def _register_method_schema(
        self,
        *,
        registry: Dict[str, Dict[str, _RegisteredCallableSchema]],
        class_name: str,
        method_name: str,
        method: Optional[Any],
        properties: Dict[str, dict],
        required: Optional[list[str]],
        positional: Optional[list[str]],
        spread: Optional[list[str]],
        allow_extra: bool,
    ) -> None:
        callable_params: frozenset[str] = frozenset()
        if method is not None and callable(method):
            callable_params = (
                self._infer_callable_params_from_callable(method)
            )

        schema = self._build_registered_callable_schema(
            properties=properties,
            required=required,
            positional=positional,
            spread=spread,
            allow_extra=allow_extra,
            callable_params=callable_params,
            allow_empty_properties=True,
        )
        class_registry = registry.setdefault(class_name, {})
        class_registry[method_name] = schema

    def register_action_builder(
        self,
        action_type: str,
        builder: Any,
        *,
        properties: Optional[Dict[str, dict]] = None,
        required: Optional[list[str]] = None,
        allow_extra: bool = False,
    ) -> None:
        """Register a custom action type for actions v2."""
        if not isinstance(action_type, str) or not action_type:
            raise ValueError("action_type must be a non-empty string")
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", action_type):
            raise ValueError(
                f"Invalid custom action type: {action_type!r}"
            )
        if action_type in {
            "create",
            "call",
            "play",
            "wait",
            "add",
            "remove",
        }:
            raise ValueError(
                f"'{action_type}' is a reserved action type"
            )
        if not callable(builder):
            raise ValueError("builder must be callable")

        action_properties = dict(properties or {})
        for k, v in action_properties.items():
            if not isinstance(k, str) or not k:
                raise ValueError(
                    "custom action properties keys must be non-empty strings"
                )
            if not isinstance(v, dict) or not v:
                raise ValueError(
                    "each custom action property schema must be a non-empty dict"
                )

        required_list = tuple(required or ())
        for name in required_list:
            if name not in action_properties:
                raise ValueError(
                    f"required field '{name}' must be defined in properties"
                )

        self._custom_action_schemas[action_type] = (
            _RegisteredActionSchema(
                properties=action_properties,
                required=required_list,
                allow_extra=bool(allow_extra),
                builder=builder,
            )
        )

    def _resolve_registered_method_schema(
        self,
        target: Any,
        method_name: str,
    ) -> Optional[_RegisteredCallableSchema]:
        registries: list[
            Dict[str, Dict[str, _RegisteredCallableSchema]]
        ] = []
        if isinstance(target, manimlib.Mobject):
            registries.append(self._mobject_method_schemas)
        if isinstance(target, manimlib.Animation):
            registries.append(self._animation_method_schemas)

        for registry in registries:
            for cls in type(target).mro():
                class_methods = registry.get(cls.__name__)
                if not class_methods:
                    continue
                if method_name in class_methods:
                    return class_methods[method_name]
        return None

    def _has_registered_method_catalog(self, target: Any) -> bool:
        registries: list[
            Dict[str, Dict[str, _RegisteredCallableSchema]]
        ] = []
        if isinstance(target, manimlib.Mobject):
            registries.append(self._mobject_method_schemas)
        if isinstance(target, manimlib.Animation):
            registries.append(self._animation_method_schemas)

        for registry in registries:
            for cls in type(target).mro():
                if cls.__name__ in registry:
                    return True
        return False

    def _validate_custom_action_payload(
        self,
        action: dict,
        schema: _RegisteredActionSchema,
        *,
        where: str,
    ) -> None:
        action_data = {k: v for k, v in action.items() if k != "type"}

        for req in schema.required:
            if req not in action_data:
                raise ValueError(
                    f"{where}: missing required field '{req}'"
                )

        if not schema.allow_extra:
            extra = set(action_data.keys()) - set(
                schema.properties.keys()
            )
            if extra:
                raise ValueError(
                    f"{where}: unexpected fields {sorted(extra)}"
                )

        defs = self._get_validation_defs()
        for key, value in action_data.items():
            if key in schema.properties:
                if value is None and key not in schema.required:
                    continue
                self._validate_value_against_schema(
                    value,
                    schema.properties[key],
                    defs=defs,
                    where=f"{where}.{key}",
                )
            elif schema.allow_extra:
                self._validate_value_against_schema(
                    value,
                    defs["value"],
                    defs=defs,
                    where=f"{where}.{key}",
                )

    def _call_action_builder(
        self,
        builder: Any,
        *,
        action_type: str,
        payload: dict,
        raw_action: dict,
    ) -> Any:
        """Call a registered custom action builder with context injection."""
        if builder is None or not callable(builder):
            raise TypeError("action builder must be callable")

        ctx = self._get_builder_context()
        call_args: list[Any] = []
        call_kwargs: dict[str, Any] = {}
        injected_positional_names: set[str] = set()

        try:
            sig = inspect.signature(builder)
        except (TypeError, ValueError):
            return builder(payload)

        params_by_name = sig.parameters

        first_pos_param = None
        for p in params_by_name.values():
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                first_pos_param = p
                break

        if first_pos_param is not None:
            n = first_pos_param.name
            if n in {"context", "ctx", "builder_context"}:
                call_args.insert(0, ctx)
                injected_positional_names.add(n)
            elif n in {"registry", "registered_objects", "objects"}:
                call_args.insert(0, self.registered_objects)
                injected_positional_names.add(n)
            elif n == "scene":
                call_args.insert(0, self.scene)
                injected_positional_names.add(n)
            elif n in {"action", "payload", "params"}:
                call_args.insert(0, payload)
                injected_positional_names.add(n)

        def can_pass_kw(name: str) -> bool:
            p = params_by_name.get(name)
            return p is not None and p.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
            )

        def inject_kw(name: str, value: Any) -> None:
            if name in injected_positional_names:
                return
            if not can_pass_kw(name):
                return
            if name in call_kwargs:
                return
            call_kwargs[name] = value

        inject_kw("context", ctx)
        inject_kw("ctx", ctx)
        inject_kw("builder_context", ctx)
        inject_kw("registry", self.registered_objects)
        inject_kw("registered_objects", self.registered_objects)
        inject_kw("objects", self.registered_objects)
        inject_kw("scene", self.scene)
        inject_kw("manimlib", manimlib)
        inject_kw("controller", self)
        inject_kw("action", payload)
        inject_kw("payload", payload)
        inject_kw("params", payload)
        inject_kw("data", raw_action)
        inject_kw("raw_action", raw_action)
        inject_kw("action_type", action_type)

        return builder(*call_args, **call_kwargs)

    def _format_method_registry_for_prompt(
        self,
        registry: Dict[str, Dict[str, _RegisteredCallableSchema]],
        *,
        max_classes: int = 30,
        max_methods_per_class: int = 20,
    ) -> str:
        if not registry:
            return "(none)"

        lines: list[str] = []
        class_items = sorted(registry.items())
        for class_name, methods in class_items[:max_classes]:
            method_items = sorted(methods.items())
            method_chunks: list[str] = []
            for method_name, schema in method_items[
                :max_methods_per_class
            ]:
                params = ", ".join(schema.properties.keys())
                if params:
                    method_chunks.append(f"{method_name}({params})")
                else:
                    method_chunks.append(f"{method_name}()")
            suffix = ""
            if len(method_items) > max_methods_per_class:
                suffix = f", ... (+{len(method_items) - max_methods_per_class} more)"
            lines.append(
                f"- {class_name}: {', '.join(method_chunks)}{suffix}"
            )

        if len(class_items) > max_classes:
            lines.append(
                f"- ... (+{len(class_items) - max_classes} more classes)"
            )
        return "\n".join(lines)

    def _format_custom_actions_for_prompt(self) -> str:
        if not self._custom_action_schemas:
            return "(none)"

        lines: list[str] = []
        for action_type, schema in sorted(
            self._custom_action_schemas.items()
        ):
            fields = ", ".join(schema.properties.keys())
            if fields:
                lines.append(f"- {action_type}: {fields}")
            else:
                lines.append(f"- {action_type}: (no fields)")
        return "\n".join(lines)

    @staticmethod
    def schema_ref() -> dict:
        return {"$ref": "#/$defs/ref"}

    @staticmethod
    def schema_value() -> dict:
        return {"$ref": "#/$defs/value"}

    def register_mobject_schema(
        self,
        class_name: str,
        *,
        properties: Dict[str, dict],
        required: Optional[list[str]] = None,
        positional: Optional[list[str]] = None,
        spread: Optional[list[str]] = None,
        allow_extra: bool = False,
    ) -> None:
        ctor = getattr(manimlib, class_name, None)
        if not inspect.isclass(ctor) or not issubclass(
            ctor, manimlib.Mobject
        ):
            raise ValueError(
                f"'{class_name}' is not a manimlib.Mobject class"
            )
        self._register_callable_schema(
            registry=self._mobject_schemas,
            class_name=class_name,
            properties=properties,
            required=required,
            positional=positional,
            spread=spread,
            allow_extra=allow_extra,
        )
        self._mobject_builders[class_name] = ctor

    def register_mobject_builder(
        self,
        name: str,
        builder: Any,
        *,
        properties: Dict[str, dict],
        required: Optional[list[str]] = None,
        positional: Optional[list[str]] = None,
        spread: Optional[list[str]] = None,
        allow_extra: bool = False,
    ) -> None:
        """Register a custom mobject constructor for actions v2.

        The builder will be called as: builder(*args, **kwargs) where args/kwargs
        are derived from the validated `params` object using `positional`/`spread`.

                If the builder signature explicitly declares any of these parameters,
                they will be injected automatically:
                    - registry / registered_objects / objects
                    - scene
                    - context / ctx / builder_context
                    - manimlib
                    - controller
                    - params (coerced)
                    - data (raw)

        This is useful for registering user-defined mobjects not exposed in `manimlib`.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not re.match(
            r"^[A-Za-z][A-Za-z0-9_]*$", name
        ) or name.startswith("_"):
            raise ValueError(
                f"Invalid registered mobject name: {name!r}"
            )
        if not callable(builder):
            raise ValueError("builder must be callable")

        self._register_callable_schema(
            registry=self._mobject_schemas,
            class_name=name,
            properties=properties,
            required=required,
            positional=positional,
            spread=spread,
            allow_extra=allow_extra,
        )
        self._mobject_builders[name] = builder

    def register_animation_schema(
        self,
        class_name: str,
        *,
        properties: Dict[str, dict],
        required: Optional[list[str]] = None,
        positional: Optional[list[str]] = None,
        spread: Optional[list[str]] = None,
        allow_extra: bool = False,
    ) -> None:
        ctor = getattr(manimlib, class_name, None)
        if not inspect.isclass(ctor) or not issubclass(
            ctor, manimlib.Animation
        ):
            raise ValueError(
                f"'{class_name}' is not a manimlib.Animation class"
            )
        self._register_callable_schema(
            registry=self._animation_schemas,
            class_name=class_name,
            properties=properties,
            required=required,
            positional=positional,
            spread=spread,
            allow_extra=allow_extra,
        )
        self._animation_builders[class_name] = ctor

    def register_animation_builder(
        self,
        name: str,
        builder: Any,
        *,
        properties: Dict[str, dict],
        required: Optional[list[str]] = None,
        positional: Optional[list[str]] = None,
        spread: Optional[list[str]] = None,
        allow_extra: bool = False,
    ) -> None:
        """Register a custom animation constructor for actions v2.

        The builder will be called as: builder(*args, **kwargs) where args/kwargs
        are derived from the validated `params` object using `positional`/`spread`.

                If the builder signature explicitly declares any of these parameters,
                they will be injected automatically:
                    - registry / registered_objects / objects
                    - scene
                    - context / ctx / builder_context
                    - manimlib
                    - controller
                    - params (coerced)
                    - data (raw)
        """
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not re.match(
            r"^[A-Za-z][A-Za-z0-9_]*$", name
        ) or name.startswith("_"):
            raise ValueError(
                f"Invalid registered animation name: {name!r}"
            )
        if not callable(builder):
            raise ValueError("builder must be callable")

        self._register_callable_schema(
            registry=self._animation_schemas,
            class_name=name,
            properties=properties,
            required=required,
            positional=positional,
            spread=spread,
            allow_extra=allow_extra,
        )
        self._animation_builders[name] = builder

    def _get_builder_context(self) -> _BuilderContext:
        return _BuilderContext(
            scene=self.scene,
            registered_objects=self.registered_objects,
            manimlib=manimlib,
            controller=self,
        )

    def _call_builder(
        self,
        builder: Any,
        args: list[Any],
        kwargs: dict[str, Any],
        *,
        params: Optional[dict] = None,
        raw_params: Optional[dict] = None,
    ) -> Any:
        """Call a registered builder with optional context injection.

        Supported optional parameters in the builder signature:
          - context / ctx / builder_context: receives a _BuilderContext
          - registry / registered_objects / objects: receives self.registered_objects
          - scene: receives self.scene
          - manimlib: receives the manimlib module
          - controller: receives this LLMSceneController
          - params: receives a coerced params dict (refs resolved, vectors coerced)
          - data: receives the raw params dict from the action

        Injection only occurs when the parameter is explicitly present in the
        callable signature (not via **kwargs), to avoid polluting forwarded kwargs.
        """
        if builder is None or not callable(builder):
            raise TypeError("builder must be callable")

        ctx = self._get_builder_context()
        call_args = list(args)
        call_kwargs = dict(kwargs)
        injected_positional_names: set[str] = set()

        try:
            sig = inspect.signature(builder)
        except (TypeError, ValueError):
            return builder(*call_args, **call_kwargs)

        params_by_name = sig.parameters

        # If the first positional parameter looks like an injected dependency,
        # pass it positionally.
        first_pos_param = None
        for p in params_by_name.values():
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                first_pos_param = p
                break

        if first_pos_param is not None:
            n = first_pos_param.name
            if n in {"context", "ctx", "builder_context"}:
                call_args.insert(0, ctx)
                injected_positional_names.add(n)
            elif n in {"registry", "registered_objects", "objects"}:
                call_args.insert(0, self.registered_objects)
                injected_positional_names.add(n)
            elif n == "scene":
                call_args.insert(0, self.scene)
                injected_positional_names.add(n)

        def can_pass_kw(name: str) -> bool:
            p = params_by_name.get(name)
            return p is not None and p.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
            )

        def inject_kw(name: str, value: Any) -> None:
            if name in injected_positional_names:
                return
            if not can_pass_kw(name):
                return
            if name in call_kwargs:
                return
            call_kwargs[name] = value

        # Inject explicit dependencies if requested by name.
        inject_kw("context", ctx)
        inject_kw("ctx", ctx)
        inject_kw("builder_context", ctx)
        inject_kw("registry", self.registered_objects)
        inject_kw("registered_objects", self.registered_objects)
        inject_kw("objects", self.registered_objects)
        inject_kw("scene", self.scene)
        inject_kw("manimlib", manimlib)
        inject_kw("controller", self)
        if params is not None:
            inject_kw("params", params)
        if raw_params is not None:
            inject_kw("data", raw_params)

        return builder(*call_args, **call_kwargs)

    def _build_registered_callable_schema(
        self,
        *,
        properties: Dict[str, dict],
        required: Optional[list[str]],
        positional: Optional[list[str]],
        spread: Optional[list[str]],
        allow_extra: bool,
        callable_params: Optional[
            list[str] | tuple[str, ...] | frozenset[str]
        ] = None,
        allow_empty_properties: bool,
    ) -> _RegisteredCallableSchema:
        if not isinstance(properties, dict) or (
            not properties and not allow_empty_properties
        ):
            raise ValueError("properties must be a non-empty dict")
        for k, v in properties.items():
            if not isinstance(k, str) or not k:
                raise ValueError(
                    "properties keys must be non-empty strings"
                )
            if not isinstance(v, dict) or not v:
                raise ValueError(
                    "each property schema must be a non-empty dict"
                )

        positional_list = tuple(positional or ())
        required_list = tuple(required or positional_list)
        spread_set = frozenset(spread or ())

        for name in positional_list:
            if name not in properties:
                raise ValueError(
                    f"positional param '{name}' must be defined in properties"
                )
        for name in required_list:
            if name not in properties:
                raise ValueError(
                    f"required param '{name}' must be defined in properties"
                )
        if not spread_set.issubset(set(positional_list)):
            raise ValueError(
                "spread params must be a subset of positional params"
            )

        callable_params_set = frozenset(callable_params or ())
        for name in callable_params_set:
            if name not in properties:
                raise ValueError(
                    f"callable param '{name}' must be defined in properties"
                )

        return _RegisteredCallableSchema(
            properties=dict(properties),
            required=required_list,
            positional=positional_list,
            spread=spread_set,
            allow_extra=bool(allow_extra),
            callable_params=callable_params_set,
        )

    def _register_callable_schema(
        self,
        *,
        registry: Dict[str, _RegisteredCallableSchema],
        class_name: str,
        properties: Dict[str, dict],
        required: Optional[list[str]],
        positional: Optional[list[str]],
        spread: Optional[list[str]],
        allow_extra: bool,
    ) -> None:
        registry[class_name] = self._build_registered_callable_schema(
            properties=properties,
            required=required,
            positional=positional,
            spread=spread,
            allow_extra=allow_extra,
            callable_params=None,
            allow_empty_properties=False,
        )

    def run_prompt(
        self,
        prompt: str,
        additional_system_prompt: str | None = None,
        **kwargs,
    ) -> None:
        """
        Sends a prompt to the LLM along with the scene context and registered objects.
        This function is designed to be called from a background thread (threading.Thread).
        """
        # 1. Build the dynamic context with the registered objects
        context_lines = []
        for name, obj in self.registered_objects.items():
            obj_type = type(obj).__name__
            context_lines.append(
                f"- Variable: `{name}` | Type: {obj_type}"
            )

        context_str = "\n".join(context_lines)

        response_mode = kwargs.pop(
            "response_mode", _LLMResponseMode.CODE
        )
        if response_mode not in (
            _LLMResponseMode.CODE,
            _LLMResponseMode.ACTIONS,
        ):
            raise ValueError(
                "Invalid response_mode. Use 'code' or 'actions'."
            )

        actions_version: Optional[int] = None
        if response_mode == _LLMResponseMode.ACTIONS:
            actions_version = kwargs.pop("actions_version", None)
            if actions_version is None:
                actions_version = (
                    2
                    if (
                        self._mobject_schemas
                        and self._animation_schemas
                    )
                    else 1
                )
            if actions_version not in (1, 2):
                raise ValueError("actions_version must be 1 or 2")

        # 2. Define the System Prompt
        if response_mode == _LLMResponseMode.ACTIONS:
            system_prompt = self._build_actions_system_prompt(
                context_str, actions_version
            )
        else:
            system_prompt = self._build_code_system_prompt(
                context_str
            )

        if additional_system_prompt:
            system_prompt += (
                "\n\n**ADDITIONAL CONSIDERATIONS**\n"
                + additional_system_prompt
            )

        history_key = self._history_key(
            response_mode, actions_version
        )
        history_messages = list(
            self._chat_histories.get(history_key, [])
        )

        # 3. API Call and Execution with Retries
        messages = [
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": prompt},
        ]

        api_kwargs = {"model": self.model}
        api_kwargs.update(kwargs)

        # Try to use response_format when supported (OpenAI structured outputs).
        # Some OpenAI-compatible providers (e.g. Groq) may reject this parameter.
        supports_response_format = True

        max_attempts = 10
        attempt = 0
        while True:
            if attempt >= max_attempts:
                print("Max attempts reached. Stopping.")
                break
            try:
                request_kwargs = dict(api_kwargs)
                if (
                    response_mode == _LLMResponseMode.ACTIONS
                    and supports_response_format
                ):
                    request_kwargs["response_format"] = (
                        self._get_actions_response_format(
                            actions_version
                        )
                    )

                response = self.client.chat.completions.create(
                    messages=messages, **request_kwargs
                )
                response_text = (
                    response.choices[0].message.content or ""
                )
                print(
                    f"LLM response received on attempt {attempt}:\n{response_text}"
                )

                # 4. Parse response based on mode
                local_result_queue = queue.Queue()
                if response_mode == _LLMResponseMode.ACTIONS:
                    actions_payload = self._extract_actions(
                        response_text
                    )
                    # 5. Send the actions payload to be executed in the main thread
                    self.execution_queue.put(
                        (
                            actions_payload,
                            local_result_queue,
                            _LLMResponseMode.ACTIONS,
                        )
                    )
                else:
                    code_to_execute = self._extract_code(
                        response_text
                    )
                    if not code_to_execute:
                        empty_code_msg = "No valid Python code block found in the response. Please ensure your response includes a properly formatted Python code block wrapped in triple backticks (```python ... ```)."
                        raise ValueError(empty_code_msg)
                    # 5. Send the code to the queue to be executed in the main thread
                    self.execution_queue.put(
                        (
                            code_to_execute,
                            local_result_queue,
                            _LLMResponseMode.CODE,
                        )
                    )

                # Wait for execution result from the main thread.
                # Use timeout as a safety net so worker threads never block forever.
                try:
                    result = local_result_queue.get(
                        timeout=self.execution_result_timeout_seconds
                    )
                except queue.Empty as exc:
                    raise TimeoutError(
                        "Timed out waiting for scene execution result. "
                        "The main-loop callback may have failed while processing the queue item."
                    ) from exc
                captured_output = result.get("output", "")

                if result["status"] == "error":
                    error_msg = f"Code execution failed with traceback:\n{result['error']}"
                    if captured_output.strip():
                        error_msg += f"\n\nStandard output before crash:\n{captured_output.strip()}"
                    raise RuntimeError(error_msg)

                persisted_history = history_messages + [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": response_text,
                    },
                ]
                self._chat_histories[history_key] = (
                    self._trim_history(persisted_history)
                )
                break  # Success, and no print output requiring feedback. exit retry loop

            except openai.APIStatusError as e:
                print(f"Attempt {attempt} failed with API status error: {e}")
                # If the provider rejects response_format, disable it and retry.
                if (
                    response_mode == _LLMResponseMode.ACTIONS
                    and supports_response_format
                ):
                    msg = str(e).lower()
                    if (
                        "response_format" in msg
                        or "json_schema" in msg
                        or "invalid" in msg
                        and "response" in msg
                    ):
                        supports_response_format = False
                        attempt += 1
                        continue
                attempt += 1

            except Exception as e:
                print(f"Attempt {attempt} failed with error: {e}")
                if not (
                    isinstance(e, ValueError)
                    and "No valid Python code block found" in str(e)
                ):
                    attempt += 1
                if attempt < max_attempts - 1:
                    # Inform the error in the chat messages so the LLM can fix it
                    if "response_text" in locals():
                        messages.append(
                            {
                                "role": "assistant",
                                "content": response_text,
                            }
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"An error has occurred. Please fix the code and try again.\nDetails of the error:\n{str(e)}",
                        }
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": system_prompt,
                        }
                    )
                    if "response_text" in locals():
                        del response_text
                else:
                    print("Max retries reached. Stopping.")
            finally:
                time.sleep(2)
        self.clear_chat_history(response_mode, actions_version)

    def _build_code_system_prompt(self, context_str: str) -> str:
        return f"""
You are an advanced AI agent in charge of controlling a math animation scene using ManimGL (manimlib).
You have global access to the scene instance through the `scene` variable.

Currently, you have the following registered objects ready to be manipulated:
{context_str}

**STRICT RULES:**
1. Your only goal is to generate executable Python code using the `manimlib` library.
2. To animate something, you MUST use `scene.play(...)`. DO NOT use a simple `scene.add()`, assume you should make a smooth transition unless told otherwise.
3. Do not write ANY explanatory text before or after your code. Return ONLY the Python markdown code block.
4. You have access to all basic Python modules and `manimlib`. NEVER import any library. All the necessary libraries are already available in the environment.
5. Do NOT use `print()` or rely on stdout for feedback. Any stdout is ignored for performance.
6. All interactions affect the same scene, so you can build on top of previous code and changes. You can also modify previously registered objects or create new ones.
7. Don't do stuff with OpenGL as it may cause crashes. Stick to the manimlib API and the provided objects.
8. After creating objects, ALWAYS adjust positions and scales so there are no overlaps, every object is fully visible within the frame, and enough free frame space remains to add more objects later at a reasonable size.
"""

    def _build_actions_system_prompt(
        self, context_str: str, actions_version: int
    ) -> str:
        if actions_version == 2:
            mobject_lines = []
            for cls, schema in sorted(self._mobject_schemas.items()):
                keys = ", ".join(schema.properties.keys())
                mobject_lines.append(f"- {cls}: {keys}")
            animation_lines = []
            for cls, schema in sorted(
                self._animation_schemas.items()
            ):
                keys = ", ".join(schema.properties.keys())
                animation_lines.append(f"- {cls}: {keys}")
            mobjects_str = (
                "\n".join(mobject_lines)
                if mobject_lines
                else "(none)"
            )
            animations_str = (
                "\n".join(animation_lines)
                if animation_lines
                else "(none)"
            )
            mobject_methods_str = (
                self._format_method_registry_for_prompt(
                    self._mobject_method_schemas
                )
            )
            animation_methods_str = (
                self._format_method_registry_for_prompt(
                    self._animation_method_schemas
                )
            )
            custom_actions_str = (
                self._format_custom_actions_for_prompt()
            )
            return f"""
You are an advanced AI agent controlling a ManimGL scene (manimlib).

Registered objects you can reference by name:
{context_str}

**OUTPUT FORMAT (STRICT JSON ONLY):**
Return ONE JSON object (no markdown, no backticks, no extra text) with:
  - version: 2
  - actions: list of actions

Allowed mobject constructors for create (registered names; params keys):
{mobjects_str}

Allowed animation constructors for play (registered names; params keys):
{animations_str}

Registered mobject methods for typed call (method and params keys):
{mobject_methods_str}

Registered animation methods for typed call (method and params keys):
{animation_methods_str}

Registered custom action types (action and fields):
{custom_actions_str}

Allowed action types:
    - create: {{"type":"create","name":"...","class":"...","params":{{"entries":[{{"key":"param_name","value":...}}]}}}} (this doesn't display the object, it just creates it and stores it by name for later reference)
    - call: {{"type":"call","target":"...","method":"...","params":{{"entries":[{{"key":"param_name","value":...}}]}}}} (calls a method on a stored object; if the method is registered in the typed catalogs above, params must match the registered schema for that method)
    - play: {{"type":"play","animations":[{{"class":"...","params":{{"entries":[{{"key":"param_name","value":...}}]}}}}],"kwargs":{{"entries":[{{"key":"run_time","value":1.0}}]}}}} (calls scene.play with a list of animations; each animation must be from the registered animations catalog, and params must match the registered schema for that animation)
  - wait: {{"type":"wait","duration": 1.0}} (calls scene.wait with the specified duration in seconds)
  - add/remove: {{"type":"add","targets":["name1","name2"]}} or {{"type":"remove","targets":["name1","name2"]}} (calls scene.add or scene.remove with the specified stored objects)
  - custom registered actions listed above

Value encoding:
  - To reference an existing stored object, use {{"ref": "object_name"}}.
    - Free-form objects (like params/kwargs) use strict map encoding: {{"entries":[{{"key":"field","value":...}}]}}.
  - Vectors can be JSON arrays like [x, y, z] or [x, y].

Rules:
  - Prefer typed call with `params`.
  - If a method is registered in the typed catalogs above, `params` is required.
  - After any `create` actions, include layout-adjustment calls so objects do not overlap, all objects remain fully visible in-frame, and there is enough free frame area to add more objects later at a reasonable size.
  - When you create an object, you must `add` it to the scene or animate its creation with `play`. Don't `create` objects without displaying them.
"""

        # v1 prompt (generic args/kwargs)
        return f"""
You are an advanced AI agent controlling a ManimGL scene (manimlib).

Registered objects you can reference by name:
{context_str}

**OUTPUT FORMAT (STRICT JSON ONLY):**
Return ONE JSON object (no markdown, no backticks, no extra text) with:
  - version: 1
  - actions: list of actions

Allowed action types:
  - create: create a Mobject from manimlib and store it by name
  - call: call a method on a stored object
  - play: call scene.play with a list of animations
  - wait: call scene.wait(duration)
  - add: scene.add(objects...)
  - remove: scene.remove(objects...)

Value encoding:
  - To reference an existing stored object, use {{"ref": "object_name"}}.
  - Vectors can be JSON arrays like [x, y, z] or [x, y].

Layout rule:
    - After any `create` actions, include follow-up calls to adjust positions and scales so objects do not overlap, all objects are fully visible in-frame, and enough free frame area remains to add additional objects later at a reasonable size.
"""

    def _get_actions_response_format(
        self, actions_version: int
    ) -> dict:
        if actions_version == 2:
            return self._get_actions_response_format_v2()
        return self._get_actions_response_format_v1()

    def _get_actions_response_format_v1(self) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "manimgl_actions_v1",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "version": {"type": "integer", "const": 1},
                        "actions": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/action"},
                            "minItems": 1,
                        },
                    },
                    "required": ["version", "actions"],
                    "$defs": self._get_common_json_defs_v1(),
                },
            },
        }

    def _get_common_json_defs_v1(self) -> dict:
        # Common defs used by v1 (args/kwargs) schema
        return {
            "ref": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"ref": {"type": "string"}},
                "required": ["ref"],
            },
            "value_object_entry": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string"},
                    "value": {"$ref": "#/$defs/value"},
                },
                "required": ["key", "value"],
            },
            "value_object": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entries": {
                        "type": "array",
                        "items": {
                            "$ref": "#/$defs/value_object_entry"
                        },
                    }
                },
                "required": ["entries"],
            },
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                    {
                        "type": "array",
                        "items": {"$ref": "#/$defs/value"},
                    },
                    {"$ref": "#/$defs/ref"},
                    {"$ref": "#/$defs/value_object"},
                ]
            },
            "kwargs": {"$ref": "#/$defs/value_object"},
            "args": {
                "type": "array",
                "items": {"$ref": "#/$defs/value"},
            },
            "animation": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "class": {"type": "string"},
                    "args": {"$ref": "#/$defs/args"},
                    "kwargs": {"$ref": "#/$defs/kwargs"},
                },
                "required": ["class", "args", "kwargs"],
            },
            "action": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"const": "create"},
                            "name": {"type": "string"},
                            "class": {"type": "string"},
                            "args": {"$ref": "#/$defs/args"},
                            "kwargs": {"$ref": "#/$defs/kwargs"},
                        },
                        "required": [
                            "type",
                            "name",
                            "class",
                            "args",
                            "kwargs",
                        ],
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"const": "call"},
                            "target": {"type": "string"},
                            "method": {"type": "string"},
                            "args": {"$ref": "#/$defs/args"},
                            "kwargs": {"$ref": "#/$defs/kwargs"},
                        },
                        "required": [
                            "type",
                            "target",
                            "method",
                            "args",
                            "kwargs",
                        ],
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"const": "play"},
                            "animations": {
                                "type": "array",
                                "items": {
                                    "$ref": "#/$defs/animation"
                                },
                                "minItems": 1,
                            },
                            "kwargs": {"$ref": "#/$defs/kwargs"},
                        },
                        "required": [
                            "type",
                            "animations",
                            "kwargs",
                        ],
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"const": "wait"},
                            "duration": {
                                "type": "number",
                                "minimum": 0,
                            },
                        },
                        "required": ["type", "duration"],
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"const": "add"},
                            "targets": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        },
                        "required": ["type", "targets"],
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"const": "remove"},
                            "targets": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        },
                        "required": ["type", "targets"],
                    },
                ]
            },
        }

    def _get_actions_response_format_v2(self) -> dict:
        if not self._mobject_schemas or not self._animation_schemas:
            raise ValueError(
                "actions_version=2 requires at least one registered mobject schema and one registered animation schema"
            )

        defs = self._get_common_json_defs_v2()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "a2",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "version": {"type": "integer", "const": 2},
                        "actions": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/u"},
                        },
                    },
                    "required": ["version", "actions"],
                    "$defs": defs,
                },
            },
        }

    def _get_common_json_defs_v2(self) -> dict:
        # Compact strict schema: keep response_format small and enforce
        # class/param specifics locally with registered schemas.
        defs: dict = {
            "r": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"ref": {"type": "string"}},
                "required": ["ref"],
            },
            "e": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string"},
                    "value": {"$ref": "#/$defs/v"},
                },
                "required": ["key", "value"],
            },
            "o": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entries": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/e"},
                    }
                },
                "required": ["entries"],
            },
            "v": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                    {
                        "type": "array",
                        "items": {"$ref": "#/$defs/v"},
                    },
                    {"$ref": "#/$defs/r"},
                    {"$ref": "#/$defs/o"},
                ]
            },
            "a": {
                "type": "array",
                "items": {"$ref": "#/$defs/v"},
            },
        }

        action_variant_refs: list[dict[str, Any]] = [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "create"},
                    "name": {"type": "string"},
                    "class": {"type": "string"},
                    "params": {"$ref": "#/$defs/o"},
                },
                "required": ["type", "name", "class", "params"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "call"},
                    "target": {"type": "string"},
                    "method": {"type": "string"},
                    "params": {
                        "$ref": "#/$defs/o",
                        "nullable": True,
                    },
                    "args": {
                        "$ref": "#/$defs/a",
                        "nullable": True,
                    },
                    "kwargs": {
                        "$ref": "#/$defs/o",
                        "nullable": True,
                    },
                },
                "required": [
                    "type",
                    "target",
                    "method",
                    "params",
                    "args",
                    "kwargs",
                ],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "play"},
                    "animations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "class": {"type": "string"},
                                "params": {
                                    "$ref": "#/$defs/o"
                                },
                            },
                            "required": ["class", "params"],
                        },
                    },
                    "kwargs": {
                        "$ref": "#/$defs/o",
                        "nullable": True,
                    },
                },
                "required": ["type", "animations", "kwargs"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "wait"},
                    "duration": {"type": "number"},
                },
                "required": ["type", "duration"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"enum": ["add", "remove"]},
                    "targets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["type", "targets"],
            },
        ]

        schema_pool: dict[str, str] = {}
        for def_name, def_schema in defs.items():
            if isinstance(def_schema, dict):
                schema_pool[
                    self._schema_cache_key(def_schema)
                ] = def_name

        for action_type, action_schema in sorted(
            self._custom_action_schemas.items()
        ):
            action_props = {"type": {"const": action_type}}
            required_set = set(action_schema.required)
            for key, prop_schema in action_schema.properties.items():
                strict_prop = self._strictify_schema_for_response(
                    prop_schema
                )
                if key not in required_set:
                    strict_prop = self._make_nullable_schema(
                        strict_prop
                    )

                if not self._is_trivial_schema(strict_prop):
                    strict_prop = self._intern_schema_ref(
                        defs,
                        schema_pool,
                        strict_prop,
                        prefix=f"custom_action_{action_type}_{key}",
                    )

                action_props[key] = strict_prop

            # Strict-mode constrained decoding requires all fields to be listed
            # as required. Optional fields are represented as nullable.
            action_required = list(action_props.keys())
            action_variant_refs.append(
                self._intern_schema_ref(
                    defs,
                    schema_pool,
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": action_props,
                        "required": action_required,
                    },
                    prefix=f"a_{action_type}",
                )
            )

        defs["u"] = {"oneOf": action_variant_refs}

        return defs

    def _schema_allows_null(self, schema: dict) -> bool:
        if not isinstance(schema, dict):
            return False

        if schema.get("nullable") is True:
            return True

        stype = schema.get("type")
        if stype == "null":
            return True
        if isinstance(stype, list) and "null" in stype:
            return True

        if schema.get("const", object()) is None:
            return True
        if "enum" in schema and isinstance(schema["enum"], list):
            if None in schema["enum"]:
                return True

        for key in ("anyOf", "oneOf"):
            options = schema.get(key)
            if isinstance(options, list):
                for option in options:
                    if self._schema_allows_null(option):
                        return True

        return False

    def _make_nullable_schema(self, schema: dict) -> dict:
        result = copy.deepcopy(schema)
        # OpenAI strict-mode constrained decoding prefers keeping the original
        # schema shape and marking optional fields with nullable=true instead of
        # introducing a new anyOf wrapper.
        result["nullable"] = True
        return result

    @staticmethod
    def _is_trivial_schema(schema: dict) -> bool:
        if not isinstance(schema, dict):
            return True

        if set(schema.keys()) == {"$ref"}:
            return False

        if "const" in schema or "enum" in schema:
            return True

        keys = set(schema.keys())
        primitive_keys = {"type", "nullable"}
        primitive_types = {
            "string",
            "number",
            "integer",
            "boolean",
            "null",
        }
        if keys.issubset(primitive_keys):
            return schema.get("type") in primitive_types

        return False

    def _schema_for_registered_params(
        self,
        schema: _RegisteredCallableSchema,
        *,
        defs: Optional[dict] = None,
        schema_pool: Optional[dict[str, str]] = None,
        schema_ref_prefix: str = "params",
    ) -> dict:
        strict_properties: dict[str, dict] = {}
        required_set = set(schema.required)

        for key, prop_schema in schema.properties.items():
            strict_prop = self._strictify_schema_for_response(
                prop_schema
            )
            if key not in required_set:
                strict_prop = self._make_nullable_schema(
                    strict_prop
                )

            if (
                defs is not None
                and schema_pool is not None
                and not self._is_trivial_schema(strict_prop)
            ):
                strict_prop = self._intern_schema_ref(
                    defs,
                    schema_pool,
                    strict_prop,
                    prefix=f"{schema_ref_prefix}_prop_{key}",
                )

            strict_properties[key] = strict_prop

        # Strict-mode constrained decoding requires all object fields in
        # `required` and `additionalProperties: false`.
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": strict_properties,
            "required": list(strict_properties.keys()),
        }

    def _extract_actions(self, text: str) -> dict:
        payload = self._extract_json_object(text)
        self._validate_actions_payload(payload)
        return payload

    def _extract_json_object(self, text: str) -> dict:
        # Fast path: valid JSON as-is
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Fallback: find the first top-level JSON object substring.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Expected a single JSON object, but none was found."
            )
        candidate = text[start : end + 1]
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            raise ValueError(
                "Expected a JSON object at the top level."
            )
        return obj

    def _validate_actions_payload(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            raise ValueError(
                "Actions response must be a JSON object."
            )

        version = payload.get("version")
        if version == 1:
            self._validate_actions_payload_v1(payload)
            return
        if version == 2:
            self._validate_actions_payload_v2(payload)
            return
        raise ValueError(
            "Actions response must include version=1 or version=2."
        )

    def _validate_actions_payload_v1(self, payload: dict) -> None:
        allowed_top_keys = {"version", "actions"}
        extra_top_keys = set(payload.keys()) - allowed_top_keys
        if extra_top_keys:
            raise ValueError(
                f"Unexpected top-level keys: {sorted(extra_top_keys)}"
            )

        if payload.get("version") != 1:
            raise ValueError(
                "Actions response must include version=1."
            )
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(
                "Actions response must include a non-empty 'actions' list."
            )
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError(f"Action #{i} must be an object.")
            t = action.get("type")
            if t not in {
                "create",
                "call",
                "play",
                "wait",
                "add",
                "remove",
            }:
                raise ValueError(f"Action #{i} has invalid type: {t}")

            if t == "create":
                allowed_keys = {
                    "type",
                    "name",
                    "class",
                    "args",
                    "kwargs",
                }
                extra = set(action.keys()) - allowed_keys
                if extra:
                    raise ValueError(
                        f"Action #{i} (create) has unexpected keys: {sorted(extra)}"
                    )
                for key in ("name", "class"):
                    if not isinstance(
                        action.get(key), str
                    ) or not action.get(key):
                        raise ValueError(
                            f"Action #{i} (create) requires non-empty '{key}'."
                        )
                if "args" in action and not isinstance(
                    action.get("args"), list
                ):
                    raise ValueError(
                        f"Action #{i} (create) 'args' must be a list."
                    )
                if "kwargs" in action and not isinstance(
                    action.get("kwargs"), dict
                ):
                    raise ValueError(
                        f"Action #{i} (create) 'kwargs' must be an object."
                    )
            elif t == "call":
                allowed_keys = {
                    "type",
                    "target",
                    "method",
                    "args",
                    "kwargs",
                }
                extra = set(action.keys()) - allowed_keys
                if extra:
                    raise ValueError(
                        f"Action #{i} (call) has unexpected keys: {sorted(extra)}"
                    )
                for key in ("target", "method"):
                    if not isinstance(
                        action.get(key), str
                    ) or not action.get(key):
                        raise ValueError(
                            f"Action #{i} (call) requires non-empty '{key}'."
                        )
                if "args" in action and not isinstance(
                    action.get("args"), list
                ):
                    raise ValueError(
                        f"Action #{i} (call) 'args' must be a list."
                    )
                if "kwargs" in action and not isinstance(
                    action.get("kwargs"), dict
                ):
                    raise ValueError(
                        f"Action #{i} (call) 'kwargs' must be an object."
                    )
            elif t == "play":
                allowed_keys = {"type", "animations", "kwargs"}
                extra = set(action.keys()) - allowed_keys
                if extra:
                    raise ValueError(
                        f"Action #{i} (play) has unexpected keys: {sorted(extra)}"
                    )
                animations = action.get("animations")
                if not isinstance(animations, list) or not animations:
                    raise ValueError(
                        f"Action #{i} (play) requires a non-empty 'animations' list."
                    )
                for j, anim in enumerate(animations):
                    if not isinstance(anim, dict):
                        raise ValueError(
                            f"Action #{i} (play) animation #{j} must be an object."
                        )
                    allowed_anim_keys = {"class", "args", "kwargs"}
                    extra_anim = set(anim.keys()) - allowed_anim_keys
                    if extra_anim:
                        raise ValueError(
                            f"Action #{i} (play) animation #{j} has unexpected keys: {sorted(extra_anim)}"
                        )
                    if not isinstance(
                        anim.get("class"), str
                    ) or not anim.get("class"):
                        raise ValueError(
                            f"Action #{i} (play) animation #{j} requires non-empty 'class'."
                        )
                    if "args" in anim and not isinstance(
                        anim.get("args"), list
                    ):
                        raise ValueError(
                            f"Action #{i} (play) animation #{j} 'args' must be a list."
                        )
                    if "kwargs" in anim and not isinstance(
                        anim.get("kwargs"), dict
                    ):
                        raise ValueError(
                            f"Action #{i} (play) animation #{j} 'kwargs' must be an object."
                        )
                if "kwargs" in action and not isinstance(
                    action.get("kwargs"), (dict, type(None))
                ):
                    raise ValueError(
                        f"Action #{i} (play) 'kwargs' must be an object."
                    )
            elif t == "wait":
                allowed_keys = {"type", "duration"}
                extra = set(action.keys()) - allowed_keys
                if extra:
                    raise ValueError(
                        f"Action #{i} (wait) has unexpected keys: {sorted(extra)}"
                    )
                dur = action.get("duration")
                if not isinstance(dur, (int, float)) or dur < 0:
                    raise ValueError(
                        f"Action #{i} (wait) requires duration >= 0."
                    )
            elif t in ("add", "remove"):
                allowed_keys = {"type", "targets"}
                extra = set(action.keys()) - allowed_keys
                if extra:
                    raise ValueError(
                        f"Action #{i} ({t}) has unexpected keys: {sorted(extra)}"
                    )
                targets = action.get("targets")
                if (
                    not isinstance(targets, list)
                    or not targets
                    or not all(
                        isinstance(x, str) and x for x in targets
                    )
                ):
                    raise ValueError(
                        f"Action #{i} ({t}) requires a non-empty list of string targets."
                    )

    def _validate_actions_payload_v2(self, payload: dict) -> None:
        allowed_top_keys = {"version", "actions"}
        extra_top_keys = set(payload.keys()) - allowed_top_keys
        if extra_top_keys:
            raise ValueError(
                f"Unexpected top-level keys: {sorted(extra_top_keys)}"
            )
        if payload.get("version") != 2:
            raise ValueError(
                "Actions response must include version=2."
            )
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(
                "Actions response must include a non-empty 'actions' list."
            )

        allowed_custom_types = set(self._custom_action_schemas.keys())
        allowed_types = {
            "create",
            "call",
            "play",
            "wait",
            "add",
            "remove",
            *allowed_custom_types,
        }

        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError(f"Action #{i} must be an object.")
            t = action.get("type")
            if t not in allowed_types:
                raise ValueError(f"Action #{i} has invalid type: {t}")

            if t == "create":
                allowed_keys = {"type", "name", "class", "params"}
                extra = set(action.keys()) - allowed_keys
                if extra:
                    raise ValueError(
                        f"Action #{i} (create) has unexpected keys: {sorted(extra)}"
                    )
                name = action.get("name")
                class_name = action.get("class")
                params = action.get("params")
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"Action #{i} (create) requires non-empty 'name'."
                    )
                if not isinstance(class_name, str) or not class_name:
                    raise ValueError(
                        f"Action #{i} (create) requires non-empty 'class'."
                    )
                if class_name not in self._mobject_schemas:
                    raise ValueError(
                        f"Action #{i} (create) class '{class_name}' is not registered."
                    )
                fallback = getattr(manimlib, class_name, None)
                if class_name not in self._mobject_builders and (
                    fallback is None or not callable(fallback)
                ):
                    raise ValueError(
                        f"Action #{i} (create) class '{class_name}' has no registered builder."
                    )
                if self._is_strict_map_object(params):
                    params = self._unwrap_strict_map_once(params)
                    action["params"] = params
                if not isinstance(params, dict):
                    raise ValueError(
                        f"Action #{i} (create) requires 'params' object."
                    )
                self._validate_params_against_registered_schema(
                    params,
                    self._mobject_schemas[class_name],
                    where=f"Action #{i} (create) params",
                )

            elif t == "play":
                allowed_keys = {"type", "animations", "kwargs"}
                extra = set(action.keys()) - allowed_keys
                if extra:
                    raise ValueError(
                        f"Action #{i} (play) has unexpected keys: {sorted(extra)}"
                    )
                animations = action.get("animations")
                if not isinstance(animations, list) or not animations:
                    raise ValueError(
                        f"Action #{i} (play) requires a non-empty 'animations' list."
                    )
                for j, anim in enumerate(animations):
                    if not isinstance(anim, dict):
                        raise ValueError(
                            f"Action #{i} (play) animation #{j} must be an object."
                        )
                    allowed_anim_keys = {"class", "params"}
                    extra_anim = set(anim.keys()) - allowed_anim_keys
                    if extra_anim:
                        raise ValueError(
                            f"Action #{i} (play) animation #{j} has unexpected keys: {sorted(extra_anim)}"
                        )
                    class_name = anim.get("class")
                    params = anim.get("params")
                    if (
                        not isinstance(class_name, str)
                        or not class_name
                    ):
                        raise ValueError(
                            f"Action #{i} (play) animation #{j} requires non-empty 'class'."
                        )
                    if class_name not in self._animation_schemas:
                        raise ValueError(
                            f"Action #{i} (play) animation #{j} class '{class_name}' is not registered."
                        )
                    fallback = getattr(manimlib, class_name, None)
                    if (
                        class_name not in self._animation_builders
                        and (
                            fallback is None or not callable(fallback)
                        )
                    ):
                        raise ValueError(
                            f"Action #{i} (play) animation #{j} class '{class_name}' has no registered builder."
                        )
                    if self._is_strict_map_object(params):
                        params = self._unwrap_strict_map_once(params)
                        anim["params"] = params
                    if not isinstance(params, dict):
                        raise ValueError(
                            f"Action #{i} (play) animation #{j} requires 'params' object."
                        )
                    self._validate_params_against_registered_schema(
                        params,
                        self._animation_schemas[class_name],
                        where=f"Action #{i} (play) animation #{j} params",
                    )
                if "kwargs" in action and not isinstance(
                    action.get("kwargs"), dict
                ):
                    raise ValueError(
                        f"Action #{i} (play) 'kwargs' must be an object."
                    )

            elif t == "call":
                allowed_keys = {
                    "type",
                    "target",
                    "method",
                    "params",
                    "args",
                    "kwargs",
                }
                extra = set(action.keys()) - allowed_keys
                if extra:
                    raise ValueError(
                        f"Action #{i} (call) has unexpected keys: {sorted(extra)}"
                    )

                for key in ("target", "method"):
                    if not isinstance(
                        action.get(key), str
                    ) or not action.get(key):
                        raise ValueError(
                            f"Action #{i} (call) requires non-empty '{key}'."
                        )

                params_value = action.get("params", None)
                args_value = action.get("args", None)
                kwargs_value = action.get("kwargs", None)

                # In strict mode, nullable optional fields are still present.
                has_params = params_value is not None
                has_args = args_value is not None and not (
                    isinstance(args_value, list)
                    and len(args_value) == 0
                )
                has_kwargs = kwargs_value is not None and not (
                    (
                        isinstance(kwargs_value, dict)
                        and len(kwargs_value) == 0
                    )
                    or (
                        self._is_strict_map_object(kwargs_value)
                        and len(kwargs_value.get("entries", []))
                        == 0
                    )
                )

                if has_params and (has_args or has_kwargs):
                    raise ValueError(
                        f"Action #{i} (call) cannot mix 'params' with 'args'/'kwargs'."
                    )

                if has_params and not isinstance(params_value, dict):
                    raise ValueError(
                        f"Action #{i} (call) 'params' must be an object."
                    )

                if has_args and not isinstance(args_value, list):
                    raise ValueError(
                        f"Action #{i} (call) 'args' must be a list."
                    )

                if has_kwargs and not isinstance(kwargs_value, dict):
                    raise ValueError(
                        f"Action #{i} (call) 'kwargs' must be an object."
                    )

            elif t == "wait":
                allowed_keys = {"type", "duration"}
                extra = set(action.keys()) - allowed_keys
                if extra:
                    raise ValueError(
                        f"Action #{i} (wait) has unexpected keys: {sorted(extra)}"
                    )
                dur = action.get("duration")
                if not isinstance(dur, (int, float)) or dur < 0:
                    raise ValueError(
                        f"Action #{i} (wait) requires duration >= 0."
                    )

            elif t in ("add", "remove"):
                allowed_keys = {"type", "targets"}
                extra = set(action.keys()) - allowed_keys
                if extra:
                    raise ValueError(
                        f"Action #{i} ({t}) has unexpected keys: {sorted(extra)}"
                    )
                targets = action.get("targets")
                if (
                    not isinstance(targets, list)
                    or not targets
                    or not all(
                        isinstance(x, str) and x for x in targets
                    )
                ):
                    raise ValueError(
                        f"Action #{i} ({t}) requires a non-empty list of string targets."
                    )

            else:
                custom_schema = self._custom_action_schemas[t]
                self._validate_custom_action_payload(
                    action,
                    custom_schema,
                    where=f"Action #{i} ({t})",
                )

    def _validate_params_against_registered_schema(
        self,
        params: dict,
        schema: _RegisteredCallableSchema,
        *,
        where: str,
    ) -> None:
        for req in schema.required:
            if req not in params:
                raise ValueError(
                    f"{where}: missing required param '{req}'"
                )
        if not schema.allow_extra:
            extra = set(params.keys()) - set(schema.properties.keys())
            if extra:
                raise ValueError(
                    f"{where}: unexpected params {sorted(extra)}"
                )

        defs = self._get_validation_defs()
        for key, value in params.items():
            if key in schema.properties:
                if value is None and key not in schema.required:
                    continue
                self._validate_value_against_schema(
                    value,
                    schema.properties[key],
                    defs=defs,
                    where=f"{where}.{key}",
                )
            elif schema.allow_extra:
                self._validate_value_against_schema(
                    value,
                    defs["value"],
                    defs=defs,
                    where=f"{where}.{key}",
                )

    def _get_validation_defs(self) -> dict:
        # Minimal defs used by local validation.
        return {
            "ref": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"ref": {"type": "string"}},
                "required": ["ref"],
            },
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                    {
                        "type": "array",
                        "items": {"$ref": "#/$defs/value"},
                    },
                    {"$ref": "#/$defs/ref"},
                    {
                        "type": "object",
                        "additionalProperties": {
                            "$ref": "#/$defs/value"
                        },
                    },
                ]
            },
        }

    def _validate_value_against_schema(
        self,
        value: Any,
        schema: dict,
        *,
        defs: dict,
        where: str,
    ) -> None:
        if not isinstance(schema, dict) or not schema:
            return

        if schema.get("nullable") and value is None:
            return

        if "$ref" in schema:
            ref = schema["$ref"]
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.split("#/$defs/", 1)[1]
                if name not in defs:
                    raise ValueError(f"{where}: unknown $ref '{ref}'")
                self._validate_value_against_schema(
                    value, defs[name], defs=defs, where=where
                )
                return
            raise ValueError(f"{where}: unsupported $ref '{ref}'")

        if "const" in schema:
            if value != schema["const"]:
                raise ValueError(
                    f"{where}: expected const {schema['const']}"
                )
            return

        if "enum" in schema:
            if value not in schema["enum"]:
                raise ValueError(
                    f"{where}: expected one of {schema['enum']}"
                )
            return

        if "anyOf" in schema:
            errors = []
            for option in schema["anyOf"]:
                try:
                    self._validate_value_against_schema(
                        value, option, defs=defs, where=where
                    )
                    return
                except Exception as e:
                    errors.append(str(e))
            raise ValueError(
                f"{where}: value does not match anyOf ({'; '.join(errors)})"
            )

        if "oneOf" in schema:
            for option in schema["oneOf"]:
                try:
                    self._validate_value_against_schema(
                        value, option, defs=defs, where=where
                    )
                    return
                except Exception:
                    pass
            raise ValueError(f"{where}: value does not match oneOf")

        expected_type = schema.get("type")
        if expected_type == "string":
            if not isinstance(value, str):
                raise ValueError(f"{where}: expected string")
            return
        if expected_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{where}: expected boolean")
            return
        if expected_type == "null":
            if value is not None:
                raise ValueError(f"{where}: expected null")
            return
        if expected_type == "integer":
            if not (
                isinstance(value, int) and not isinstance(value, bool)
            ):
                raise ValueError(f"{where}: expected integer")
            return
        if expected_type == "number":
            if not (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                raise ValueError(f"{where}: expected number")
            return
        if expected_type == "array":
            if not isinstance(value, list):
                raise ValueError(f"{where}: expected array")
            if (
                "minItems" in schema
                and len(value) < schema["minItems"]
            ):
                raise ValueError(
                    f"{where}: expected at least {schema['minItems']} items"
                )
            if (
                "maxItems" in schema
                and len(value) > schema["maxItems"]
            ):
                raise ValueError(
                    f"{where}: expected at most {schema['maxItems']} items"
                )
            items_schema = schema.get("items")
            if isinstance(items_schema, dict):
                for idx, item in enumerate(value):
                    self._validate_value_against_schema(
                        item,
                        items_schema,
                        defs=defs,
                        where=f"{where}[{idx}]",
                    )
            return
        if expected_type == "object" or (
            expected_type is None
            and (
                "properties" in schema
                or "additionalProperties" in schema
            )
        ):
            if not isinstance(value, dict):
                raise ValueError(f"{where}: expected object")
            required = schema.get("required", [])
            for req in required:
                if req not in value:
                    raise ValueError(
                        f"{where}: missing required key '{req}'"
                    )
            properties = schema.get("properties", {})
            additional = schema.get("additionalProperties", True)
            for k, v in value.items():
                if k in properties and isinstance(
                    properties[k], dict
                ):
                    self._validate_value_against_schema(
                        v,
                        properties[k],
                        defs=defs,
                        where=f"{where}.{k}",
                    )
                elif additional is False:
                    raise ValueError(f"{where}: unexpected key '{k}'")
                elif isinstance(additional, dict):
                    self._validate_value_against_schema(
                        v,
                        additional,
                        defs=defs,
                        where=f"{where}.{k}",
                    )
            return

    def _is_strict_map_object(self, value: Any) -> bool:
        if not isinstance(value, dict) or set(value.keys()) != {"entries"}:
            return False
        entries = value.get("entries")
        if not isinstance(entries, list):
            return False
        for item in entries:
            if not isinstance(item, dict):
                return False
            if set(item.keys()) != {"key", "value"}:
                return False
            if not isinstance(item.get("key"), str):
                return False
        return True

    def _unwrap_strict_map_once(self, value: dict) -> dict[str, Any]:
        return {
            item["key"]: item["value"] for item in value["entries"]
        }

    def _coerce_value(self, value: Any) -> Any:
        # Convert JSON encodings into runtime values.
        # - {"ref": "name"} resolves to a registered object
        # - [x,y] / [x,y,z] become vectors (numpy arrays)
        if isinstance(value, dict) and set(value.keys()) == {"ref"}:
            ref_name = value.get("ref")
            if not isinstance(ref_name, str) or not ref_name:
                raise ValueError("Invalid ref value")
            if ref_name not in self.registered_objects:
                raise KeyError(f"Unknown ref: {ref_name}")
            return self.registered_objects[ref_name]

        if self._is_strict_map_object(value):
            unwrapped = self._unwrap_strict_map_once(value)
            return {
                k: self._coerce_value(v)
                for k, v in unwrapped.items()
            }

        if isinstance(value, list):
            coerced_list = [self._coerce_value(v) for v in value]
            if len(coerced_list) in (2, 3) and all(
                isinstance(x, (int, float)) for x in coerced_list
            ):
                if len(coerced_list) == 2:
                    coerced_list = [
                        coerced_list[0],
                        coerced_list[1],
                        0.0,
                    ]
                return np.array(coerced_list, dtype=float)
            return coerced_list

        if isinstance(value, dict):
            return {
                k: self._coerce_value(v) for k, v in value.items()
            }

        if isinstance(value, str):
            # Common direction constants
            direction_map = {
                "UP": manimlib.UP,
                "DOWN": manimlib.DOWN,
                "LEFT": manimlib.LEFT,
                "RIGHT": manimlib.RIGHT,
                "IN": manimlib.IN,
                "OUT": manimlib.OUT,
                "ORIGIN": manimlib.ORIGIN,
            }
            return direction_map.get(value, value)

        return value

    def _resolve_manimlib_callable(self, name: str) -> Any:
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name or ""):
            raise ValueError(f"Invalid callable name: {name}")
        if name.startswith("_"):
            raise ValueError(f"Invalid callable name: {name}")
        attr = getattr(manimlib, name, None)
        if attr is None or not callable(attr):
            raise AttributeError(
                f"manimlib has no callable named '{name}'"
            )
        return attr

    def _safe_callable_eval_globals(
        self,
        *,
        target: Any | None = None,
    ) -> dict[str, Any]:
        allowed: dict[str, Any] = {
            "np": np,
            "random": random,
            "math": math,
            "manimlib": manimlib,
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
            "linear": manimlib.linear,
            "smooth": manimlib.smooth,
            "double_smooth": manimlib.double_smooth,
            "there_and_back": manimlib.there_and_back,
            "wiggle": manimlib.wiggle,
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
        if target is not None:
            allowed["target"] = target
            allowed["mob"] = target
            allowed["mobject"] = target

        for name, obj in self.registered_objects.items():
            if name.isidentifier() and not name.startswith("_"):
                allowed[name] = obj
        return allowed

    @staticmethod
    def _validate_safe_callable_expression_ast(
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
            if isinstance(
                node, ast.Attribute
            ) and node.attr.startswith("__"):
                raise ValueError(
                    "Dunder attribute access is not allowed"
                )
            if isinstance(node, ast.Name):
                if node.id.startswith("__"):
                    raise ValueError("Dunder names are not allowed")
                if (
                    isinstance(node.ctx, ast.Load)
                    and node.id not in allowed_names
                    and node.id not in local_names
                ):
                    raise ValueError(
                        f"Name '{node.id}' is not allowed in callable expression"
                    )

    def _safe_eval_callable_expression(
        self,
        expression: str,
        *,
        target: Any | None = None,
    ) -> Any:
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(
                "Callable expression must be a non-empty string"
            )
        allowed = self._safe_callable_eval_globals(target=target)
        tree = ast.parse(expression, mode="eval")
        self._validate_safe_callable_expression_ast(
            tree,
            allowed_names=set(allowed.keys()),
        )
        code = compile(tree, "<llm_call_expr>", "eval")
        result = eval(code, {"__builtins__": {}, **allowed}, {})
        if not callable(result):
            raise TypeError(
                "Callable expression must evaluate to a callable"
            )
        return result

    def _coerce_typed_call_params(
        self,
        params: dict[str, Any],
        *,
        target: Any,
        method: Any,
        method_schema: _RegisteredCallableSchema,
    ) -> dict[str, Any]:
        callable_param_names = set(method_schema.callable_params)
        callable_param_names.update(
            self._infer_callable_params_from_callable(method)
        )
        if not callable_param_names:
            return dict(params)

        converted_params = dict(params)
        for name in callable_param_names:
            if name not in converted_params:
                continue
            value = converted_params[name]
            if isinstance(value, str):
                converted_params[name] = (
                    self._safe_eval_callable_expression(
                        value,
                        target=target,
                    )
                )
            elif name in method_schema.spread and isinstance(
                value, list
            ):
                converted_params[name] = [
                    self._safe_eval_callable_expression(
                        item,
                        target=target,
                    )
                    if isinstance(item, str)
                    else item
                    for item in value
                ]

        return converted_params

    def _coerce_call_callable_args(
        self,
        method_name: str,
        target: Any,
        method: Any,
        args: list[Any],
        kwargs: dict[str, Any],
        *,
        method_schema: Optional[_RegisteredCallableSchema] = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        converted_args = list(args)
        converted_kwargs = dict(kwargs)

        callable_param_names: set[str] = set()
        if method_schema is not None:
            callable_param_names.update(method_schema.callable_params)

        positional_param_names: list[str] = []
        var_positional_name: Optional[str] = None

        try:
            method_sig = inspect.signature(method)
        except (TypeError, ValueError):
            method_sig = None

        if method_sig is not None:
            callable_param_names.update(
                self._infer_callable_params_from_callable(
                    method,
                    sig=method_sig,
                )
            )
            for idx, param in enumerate(
                method_sig.parameters.values()
            ):
                if (
                    idx == 0
                    and param.name in {"self", "cls"}
                    and param.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ):
                    continue
                if param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    positional_param_names.append(param.name)
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    var_positional_name = param.name
        elif method_schema is not None:
            positional_param_names = list(method_schema.positional)
            if method_schema.spread:
                var_positional_name = next(iter(method_schema.spread))

        for idx, value in enumerate(converted_args):
            param_name: Optional[str] = None
            if idx < len(positional_param_names):
                param_name = positional_param_names[idx]
            elif var_positional_name is not None:
                param_name = var_positional_name

            if param_name in callable_param_names and isinstance(
                value, str
            ):
                converted_args[idx] = (
                    self._safe_eval_callable_expression(
                        value,
                        target=target,
                    )
                )

        for key, value in list(converted_kwargs.items()):
            if key in callable_param_names and isinstance(value, str):
                converted_kwargs[key] = (
                    self._safe_eval_callable_expression(
                        value,
                        target=target,
                    )
                )

        if method_name != "add_updater":
            return converted_args, converted_kwargs

        if converted_args and isinstance(converted_args[0], str):
            converted_args[0] = self._safe_eval_callable_expression(
                converted_args[0],
                target=target,
            )

        for key in (
            "update_function",
            "updater",
            "func",
            "function",
        ):
            value = converted_kwargs.get(key)
            if isinstance(value, str):
                converted_kwargs[key] = (
                    self._safe_eval_callable_expression(
                        value,
                        target=target,
                    )
                )

        return converted_args, converted_kwargs

    def _extract_code(self, text: str) -> Optional[str]:
        """Finds and extracts a Python code block using Regular Expressions."""
        # We use chr(96)*3 to generate ``` without breaking the markdown formatting
        ticks = chr(96) * 3
        pattern = f"{ticks}(?:python|py)?\n(.*?)\n{ticks}"

        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # If the model didn't put the backticks but returned code, assume everything is code
        if "scene.play" in text or "manimlib" in text:
            return text.strip()

        return None

    def _capture_execution_snapshot(self) -> dict[str, Any]:
        scene_state = manimlib.SceneState(self.scene)
        registered_snapshot: dict[str, Any] = {}
        detached_mobject_copies: dict[str, manimlib.Mobject] = {}

        scene_mobject_ids = {
            id(mob) for mob in scene_state.mobjects_to_copies.keys()
        }
        for name, obj in self.registered_objects.items():
            if isinstance(obj, manimlib.Mobject):
                registered_snapshot[name] = obj
                # Mobjects outside scene.mobjects still need manual restore.
                if id(obj) not in scene_mobject_ids:
                    detached_mobject_copies[name] = obj.copy()
            else:
                try:
                    registered_snapshot[name] = copy.deepcopy(obj)
                except Exception:
                    registered_snapshot[name] = obj

        return {
            "scene_state": scene_state,
            "registered_objects": registered_snapshot,
            "detached_mobject_copies": detached_mobject_copies,
        }

    def _restore_execution_snapshot(
        self, snapshot: dict[str, Any]
    ) -> None:
        scene_state = snapshot["scene_state"]
        scene_state.restore_scene(self.scene)

        self.registered_objects = dict(snapshot["registered_objects"])

        for name, saved_copy in snapshot[
            "detached_mobject_copies"
        ].items():
            obj = self.registered_objects.get(name)
            if isinstance(obj, manimlib.Mobject):
                obj.become(saved_copy, match_updaters=True)

    def _process_queue(self) -> None:
        """
        Runs on the scene main loop and executes one pending queued payload.
        """
        if self._is_processing_queue_item:
            return

        item_consumed = False
        result_queue = None

        try:
            self._is_processing_queue_item = True

            # Try to get code from the queue without blocking the thread
            item = self.execution_queue.get_nowait()
            item_consumed = True
            if isinstance(item, tuple) and len(item) == 2:
                payload, result_queue = item
                mode = _LLMResponseMode.CODE
            elif isinstance(item, tuple) and len(item) == 3:
                payload, result_queue, mode = item
            else:
                raise RuntimeError("Invalid queue item")

            snapshot = self._capture_execution_snapshot()

            if mode == _LLMResponseMode.ACTIONS:
                error_traceback, captured_output = (
                    self._execute_actions(payload)
                )
            else:
                error_traceback, captured_output = self._execute_code(
                    payload
                )

            if error_traceback:
                try:
                    self._restore_execution_snapshot(snapshot)
                except Exception:
                    error_traceback += (
                        "\n\nRollback failed with traceback:\n"
                        + traceback.format_exc()
                    )
                result_queue.put(
                    {
                        "status": "error",
                        "error": error_traceback,
                        "output": captured_output,
                    }
                )
            else:
                result_queue.put(
                    {"status": "success", "output": captured_output}
                )
        except queue.Empty:
            return
        except Exception:
            if result_queue is not None:
                try:
                    result_queue.put(
                        {
                            "status": "error",
                            "error": traceback.format_exc(),
                            "output": "",
                        }
                    )
                except Exception:
                    traceback.print_exc()
            else:
                traceback.print_exc()
        finally:
            if item_consumed:
                try:
                    self.execution_queue.task_done()
                except ValueError:
                    pass
            self._is_processing_queue_item = False

    def _play_scene(
        self, *proto_animations: Any, **kwargs: Any
    ) -> None:
        """Play animations with non-blocking defaults for interactive scenes."""
        if not proto_animations:
            return
        try:
            play_params = inspect.signature(
                self.scene.play
            ).parameters
        except (TypeError, ValueError):
            play_params = {}

        if "hold" in play_params:
            kwargs["hold"] = False

        self.scene.play(*proto_animations, **kwargs)

    def _wait_scene(
        self, duration: Optional[float] = None, **kwargs: Any
    ) -> None:
        """Wait with presenter-mode hold disabled by default."""
        try:
            wait_params = inspect.signature(
                self.scene.wait
            ).parameters
        except (TypeError, ValueError):
            wait_params = {}

        if "ignore_presenter_mode" in wait_params:
            kwargs["ignore_presenter_mode"] = True

        self.scene.wait(duration=duration, **kwargs)

    def _execute_actions(
        self, payload: dict
    ) -> tuple[Optional[str], str]:
        output_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buffer):
                version = payload.get("version", 1)
                actions = payload.get("actions", [])
                if version == 2:
                    self._execute_actions_v2(actions)
                else:
                    self._execute_actions_v1(actions)
            return None, output_buffer.getvalue()
        except Exception:
            return traceback.format_exc(), output_buffer.getvalue()

    def _execute_actions_v1(self, actions: list[dict]) -> None:
        for action in actions:
            t = action["type"]
            if t == "create":
                name = action["name"]
                cls_name = action["class"]
                args = [
                    self._coerce_value(v)
                    for v in action.get("args", [])
                ]
                raw_kwargs = action.get("kwargs", {})
                if raw_kwargs is None:
                    raw_kwargs = {}
                elif self._is_strict_map_object(raw_kwargs):
                    raw_kwargs = self._unwrap_strict_map_once(
                        raw_kwargs
                    )
                kwargs = {
                    k: self._coerce_value(v)
                    for k, v in raw_kwargs.items()
                }
                ctor = self._resolve_manimlib_callable(cls_name)
                obj = ctor(*args, **kwargs)
                if not isinstance(obj, manimlib.Mobject):
                    raise TypeError(
                        f"create '{cls_name}' did not return a Mobject"
                    )
                self.registered_objects[name] = obj

            elif t == "call":
                self._execute_call_action(action)

            elif t == "add":
                targets = [
                    self.registered_objects[n]
                    for n in action["targets"]
                ]
                self.scene.add(*targets)

            elif t == "remove":
                targets = [
                    self.registered_objects[n]
                    for n in action["targets"]
                ]
                self.scene.remove(*targets)

            elif t == "wait":
                self._wait_scene(action["duration"])

            elif t == "play":
                animations = []
                for anim_spec in action.get("animations", []):
                    anim_cls_name = anim_spec["class"]
                    anim_args = [
                        self._coerce_value(v)
                        for v in anim_spec.get("args", [])
                    ]
                    raw_anim_kwargs = anim_spec.get("kwargs", {})
                    if raw_anim_kwargs is None:
                        raw_anim_kwargs = {}
                    elif self._is_strict_map_object(
                        raw_anim_kwargs
                    ):
                        raw_anim_kwargs = (
                            self._unwrap_strict_map_once(
                                raw_anim_kwargs
                            )
                        )
                    anim_kwargs = {
                        k: self._coerce_value(v)
                        for k, v in raw_anim_kwargs.items()
                    }
                    anim_ctor = self._resolve_manimlib_callable(
                        anim_cls_name
                    )
                    animations.append(
                        anim_ctor(*anim_args, **anim_kwargs)
                    )
                raw_play_kwargs = action.get("kwargs", {})
                if raw_play_kwargs is None:
                    raw_play_kwargs = {}
                elif self._is_strict_map_object(raw_play_kwargs):
                    raw_play_kwargs = self._unwrap_strict_map_once(
                        raw_play_kwargs
                    )
                play_kwargs = {
                    k: self._coerce_value(v)
                    for k, v in raw_play_kwargs.items()
                }
                self._play_scene(*animations, **play_kwargs)

            else:
                raise ValueError(f"Unsupported action type: {t}")

    def _execute_actions_v2(self, actions: list[dict]) -> None:
        for action in actions:
            t = action["type"]
            if t == "create":
                name = action["name"]
                cls_name = action["class"]
                raw_params = action["params"]
                if self._is_strict_map_object(raw_params):
                    raw_params = self._unwrap_strict_map_once(
                        raw_params
                    )
                schema = self._mobject_schemas[cls_name]
                args, kwargs = self._args_kwargs_from_params(
                    raw_params, schema
                )
                coerced_params = {
                    k: self._coerce_value(v)
                    for k, v in raw_params.items()
                }
                builder = self._mobject_builders.get(cls_name)
                if builder is None:
                    builder = self._resolve_manimlib_callable(
                        cls_name
                    )
                obj = self._call_builder(
                    builder,
                    args,
                    kwargs,
                    params=coerced_params,
                    raw_params=raw_params,
                )
                if not isinstance(obj, manimlib.Mobject):
                    raise TypeError(
                        f"create '{cls_name}' did not return a Mobject"
                    )
                self.registered_objects[name] = obj

            elif t == "call":
                self._execute_call_action(action)

            elif t == "add":
                targets = [
                    self.registered_objects[n]
                    for n in action["targets"]
                ]
                self.scene.add(*targets)

            elif t == "remove":
                targets = [
                    self.registered_objects[n]
                    for n in action["targets"]
                ]
                self.scene.remove(*targets)

            elif t == "wait":
                self._wait_scene(action["duration"])

            elif t == "play":
                animations = []
                for anim_spec in action.get("animations", []):
                    anim_cls_name = anim_spec["class"]
                    raw_params = anim_spec["params"]
                    if self._is_strict_map_object(raw_params):
                        raw_params = self._unwrap_strict_map_once(
                            raw_params
                        )
                    schema = self._animation_schemas[anim_cls_name]
                    anim_args, anim_kwargs = (
                        self._args_kwargs_from_params(
                            raw_params, schema
                        )
                    )
                    coerced_params = {
                        k: self._coerce_value(v)
                        for k, v in raw_params.items()
                    }
                    anim_ctor = self._animation_builders.get(
                        anim_cls_name
                    )
                    if anim_ctor is None:
                        anim_ctor = self._resolve_manimlib_callable(
                            anim_cls_name
                        )
                    animations.append(
                        self._call_builder(
                            anim_ctor,
                            anim_args,
                            anim_kwargs,
                            params=coerced_params,
                            raw_params=raw_params,
                        )
                    )
                raw_play_kwargs = action.get("kwargs")
                if raw_play_kwargs is None:
                    raw_play_kwargs = {}
                else:
                    raw_play_kwargs = self._coerce_value(
                        raw_play_kwargs
                    )
                    if not isinstance(raw_play_kwargs, dict):
                        raise ValueError(
                            "play kwargs must be an object"
                        )
                play_kwargs = dict(raw_play_kwargs)
                self._play_scene(*animations, **play_kwargs)

            elif t in self._custom_action_schemas:
                self._execute_custom_action(action)

            else:
                raise ValueError(f"Unsupported action type: {t}")

    def _execute_call_action(self, action: dict) -> None:
        target_name = action["target"]
        method_name = action["method"]
        if method_name.startswith("_"):
            raise ValueError("Calling private methods is not allowed")
        if target_name not in self.registered_objects:
            raise KeyError(f"Unknown target: {target_name}")
        target = self.registered_objects[target_name]
        method = getattr(target, method_name, None)
        if method is None or not callable(method):
            raise AttributeError(
                f"Target '{target_name}' has no method '{method_name}'"
            )

        method_schema = self._resolve_registered_method_schema(
            target,
            method_name,
        )
        has_catalog = self._has_registered_method_catalog(target)
        has_typed_params = action.get("params") is not None

        if method_schema is not None:
            if not has_typed_params:
                raise ValueError(
                    f"Call '{type(target).__name__}.{method_name}' requires typed 'params'."
                )
            raw_params = action.get("params")
            if not isinstance(raw_params, dict):
                raise ValueError(
                    f"Call '{type(target).__name__}.{method_name}' requires 'params' object."
                )
            if self._is_strict_map_object(raw_params):
                raw_params = self._unwrap_strict_map_once(
                    raw_params
                )
            self._validate_params_against_registered_schema(
                raw_params,
                method_schema,
                where=f"call.{type(target).__name__}.{method_name}",
            )
            typed_params = self._coerce_typed_call_params(
                raw_params,
                target=target,
                method=method,
                method_schema=method_schema,
            )
            args, kwargs = self._args_kwargs_from_params(
                typed_params,
                method_schema,
            )
        else:
            if has_typed_params:
                raise ValueError(
                    f"Method '{type(target).__name__}.{method_name}' is not registered for typed params."
                )
            if has_catalog:
                raise ValueError(
                    f"Method '{type(target).__name__}.{method_name}' is not in the registered method catalog."
                )
            raw_args = action.get("args")
            if raw_args is None:
                raw_args = []
            raw_kwargs = action.get("kwargs")
            if raw_kwargs is None:
                raw_kwargs = {}
            else:
                raw_kwargs = self._coerce_value(raw_kwargs)
                if not isinstance(raw_kwargs, dict):
                    raise ValueError(
                        f"Call '{type(target).__name__}.{method_name}' kwargs must be an object."
                    )
            args = [
                self._coerce_value(v) for v in raw_args
            ]
            kwargs = {
                k: self._coerce_value(v)
                for k, v in raw_kwargs.items()
            }

        args, kwargs = self._coerce_call_callable_args(
            method_name,
            target,
            method,
            args,
            kwargs,
            method_schema=method_schema,
        )
        method(*args, **kwargs)

    def _execute_custom_action(self, action: dict) -> None:
        action_type = action["type"]
        schema = self._custom_action_schemas[action_type]

        self._validate_custom_action_payload(
            action,
            schema,
            where=f"custom action '{action_type}'",
        )

        raw_payload = {k: v for k, v in action.items() if k != "type"}
        payload = {
            k: self._coerce_value(v) for k, v in raw_payload.items()
        }

        self._call_action_builder(
            schema.builder,
            action_type=action_type,
            payload=payload,
            raw_action=dict(action),
        )

    def _args_kwargs_from_params(
        self, params: dict, schema: _RegisteredCallableSchema
    ) -> tuple[list[Any], dict[str, Any]]:
        args: list[Any] = []
        used_keys: set[str] = set()
        for name in schema.positional:
            if name not in params:
                if name in schema.required:
                    raise ValueError(
                        f"Missing positional param '{name}'"
                    )
                continue

            value = self._coerce_value(params[name])
            if value is None and name not in schema.required:
                # Strict-mode nullable optionals are represented as explicit null.
                # Treat them as omitted so builder defaults can apply.
                continue

            used_keys.add(name)
            if name in schema.spread:
                if not isinstance(value, list):
                    raise ValueError(
                        f"Param '{name}' must be an array to be spread"
                    )
                args.extend(value)
            else:
                args.append(value)

        kwargs: dict[str, Any] = {}
        for k, v in params.items():
            if k in used_keys:
                continue
            if v is None and k not in schema.required:
                continue
            kwargs[k] = self._coerce_value(v)
        return args, kwargs

    def _execute_code(self, code: str) -> tuple[Optional[str], str]:
        """
        Executes the code in a safe namespace (globals) that includes the scene,
        the manimlib library, and the registered objects.
        Returns a tuple of (traceback_string_if_error_else_None, captured_output).
        """
        controller = self

        class _SceneExecutionProxy:
            def __init__(self, scene: Any) -> None:
                self._scene = scene

            def __getattr__(self, name: str) -> Any:
                return getattr(self._scene, name)

            def play(
                self,
                *proto_animations: Any,
                **kwargs: Any,
            ) -> None:
                controller._play_scene(*proto_animations, **kwargs)

            def wait(
                self,
                duration: Optional[float] = None,
                **kwargs: Any,
            ) -> None:
                controller._wait_scene(duration=duration, **kwargs)

        exec_globals = {
            "scene": _SceneExecutionProxy(self.scene),
            "manimlib": manimlib,
            "inspect": inspect,
            "random": random,
            "math": math,
            "np": np,
            "sympy": sympy,
            # Disable prints from LLM-generated code for performance.
            "print": (lambda *args, **kwargs: None),
        }
        # Inject the registered objects as global variables
        exec_globals.update(self.registered_objects)

        output_buffer = io.StringIO()
        try:
            # exec runs sharing the same globals, redirecting stdout
            with contextlib.redirect_stdout(output_buffer):
                exec(code, exec_globals, exec_globals)
            return None, output_buffer.getvalue()
        except Exception:
            return traceback.format_exc(), output_buffer.getvalue()
