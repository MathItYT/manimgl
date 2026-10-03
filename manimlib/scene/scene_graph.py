from __future__ import annotations

import json
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def describe_callable(func):
    if func is None:
        return None
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if module is None or qualname is None or "<lambda>" in qualname:
        return {
            "serializable": False,
            "name": getattr(func, "__name__", None),
        }
    return {
        "serializable": True,
        "module": module,
        "qualname": qualname,
    }


def serialize_value(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return describe_callable(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    return repr(value)


class SceneGraphSerializer:
    """Serialize a Scene timeline and object graph without executing user code."""

    def __init__(self, scene):
        self.scene = scene

    def serialize_mobject(self, mob):
        parameters = getattr(mob, "_serialization_parameters", {})
        constructor = getattr(
            mob,
            "_serialization_constructor",
            mob.__class__.__qualname__,
        )
        name = getattr(
            mob,
            "_serialization_name",
            mob.__class__.__qualname__,
        )
        return {
            "id": str(id(mob)),
            "name": name,
            "constructor": constructor,
            "parameters": serialize_value(parameters),
            "children": [self.serialize_mobject(sm) for sm in mob.submobjects],
        }

    def serialize_animation(self, event):
        animation = event.animation
        result = {
            "name": event.name,
            "type": (
                animation.__class__.__qualname__
                if animation is not None
                else event.kind
            ),
            "t_start": event.t_start,
            "t_end": event.t_end,
        }
        if animation is not None:
            result["rate_func"] = describe_callable(animation.rate_func)
            result["object"] = {
                "id": str(id(animation.mobject)),
                "name": getattr(
                    animation.mobject,
                    "_serialization_name",
                    animation.mobject.__class__.__qualname__,
                ),
            }
        if event.metadata:
            result["metadata"] = serialize_value(event.metadata)
        return result

    def to_dict(self):
        return {
            "scene": {
                "name": self.scene.__class__.__qualname__,
                "duration": self.scene.time,
            },
            "objects": [
                self.serialize_mobject(mob)
                for mob in self.scene.get_top_level_mobjects()
            ],
            "animations": [
                self.serialize_animation(event)
                for event in self.scene.timeline
            ],
        }

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), indent=2, **kwargs)

    def to_yaml(self, **kwargs):
        if yaml is None:
            raise ImportError(
                "PyYAML is required for SceneGraphSerializer.to_yaml()"
            )
        return yaml.safe_dump(self.to_dict(), sort_keys=False, **kwargs)
