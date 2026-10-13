from __future__ import annotations

import base64
from copy import deepcopy
from pathlib import Path
import re
from urllib.parse import unquote, unquote_to_bytes, urlparse
from xml.etree import ElementTree as ET

import io
import numpy as np
import svgelements as se

from manimlib.constants import RIGHT
from manimlib.constants import TAU
from manimlib.logger import log
from manimlib.mobject.geometry import Circle
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Polygon
from manimlib.mobject.geometry import Polyline
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.geometry import RoundedRectangle
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.renderer.texture import ImageFile
from manimlib.renderer.texture import PixelArray
from manimlib.utils.bezier import quadratic_bezier_points_for_arc
from manimlib.utils.images import get_full_raster_image_path
from manimlib.utils.images import get_full_vector_image_path
from manimlib.utils.iterables import hash_obj
from manimlib.utils.space_ops import rotation_about_z

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manimlib.typing import ManimColor, Vect3Array


# A stroke_width of 100 spans one unit of a default scale frame, see
# STROKE_WIDTH_CONVERSION in shaders/stroke.wgsl
STROKE_WIDTHS_PER_UNIT: float = 100.0

SVG_HASH_TO_MOB_MAP: dict[int, list[VMobject]] = {}
PATH_TO_POINTS: dict[str, Vect3Array] = {}


def get_svg_content_height(svg_string: str) -> float:
    root = ET.fromstring(svg_string)
    # The browser Pango backend already measured the ink extents before
    # writing its SVG. Reuse that exact measurement instead of reparsing all
    # glyph paths through svgelements.
    measured_height = root.attrib.get("data-manim-content-height")
    if measured_height is not None:
        try:
            value = float(measured_height)
            if np.isfinite(value) and value > 0:
                return value
        except (TypeError, ValueError):
            pass

    # Strip root attributes to match SVGMobject.modify_xml_tree,
    # which avoids viewBox unit conversions (e.g. pt to px for dvisvgm)
    root = ET.fromstring(svg_string)
    root.attrib.clear()
    data_stream = io.BytesIO()
    ET.ElementTree(root).write(data_stream)
    data_stream.seek(0)
    svg = se.SVG.parse(data_stream)
    bbox = svg.bbox()
    if bbox is None:
        raise ValueError("SVG has no content to measure")
    return bbox[3] - bbox[1]


def _convert_point_to_3d(x: float, y: float) -> np.ndarray:
    return np.array([x, y, 0.0])


class SVGMobject(VMobject):
    file_name: str = ""
    height: float | None = None
    width: float | None = None

    def __init__(
        self,
        file_name: str = "",
        svg_string: str = "",
        should_center: bool = True,
        height: float | None = 2.0,
        width: float | None = None,
        # Style that overrides the original svg
        color: ManimColor = None,
        fill_color: ManimColor = None,
        fill_opacity: float | None = None,
        stroke_width: float | None = None,
        stroke_color: ManimColor = None,
        stroke_opacity: float | None = None,
        # Style that fills only when not specified
        # If None, regarded as default values from svg standard
        svg_default: dict = dict(
            color=None,
            opacity=None,
            fill_color=None,
            fill_opacity=None,
            stroke_width=None,
            stroke_color=None,
            stroke_opacity=None,
        ),
        path_string_config: dict = dict(),
        **kwargs
    ):
        self._svg_source_path: Path | None = None
        self._svg_root_attributes: dict[str, str] = {}
        self._svg_clip_paths: dict[str, ET.Element] = {}
        self._svg_clip_assignments: dict[str, str] = {}
        self._svg_clip_shape_ids: set[str] = set()
        self._svg_gradients: dict[str, ET.Element] = {}
        self._svg_gradient_assignments: dict[str, tuple[str, float]] = {}

        if svg_string != "":
            self.svg_string = svg_string
        elif file_name != "":
            self.svg_string = self.file_name_to_svg_string(file_name)
        elif self.file_name != "":
            self.svg_string = self.file_name_to_svg_string(self.file_name)
        else:
            raise Exception("Must specify either a file_name or svg_string SVGMobject")

        self.svg_default = dict(svg_default)
        self.path_string_config = dict(path_string_config)

        super().__init__(**kwargs)
        self.init_svg_mobject()
        self.ensure_positive_orientation()

        # Initialize position
        height = height or self.height
        width = width or self.width

        initial_height = self.get_height()

        if should_center:
            self.center()
        if height is not None:
            self.set_height(height)
        if width is not None:
            self.set_width(width)

        # Widths read from the file are in the svg's own user units, while the geometry
        # above has just been resized to the requested height or width. Converting them
        # keeps a stroke the thickness it looks in the file, whatever units that file
        # happens to be drawn in.
        if initial_height > 0:
            units_per_user_unit = self.get_height() / initial_height
            self.scale_stroke_widths(STROKE_WIDTHS_PER_UNIT * units_per_user_unit)

        # Rather than passing style into super().__init__
        # do it after svg has been taken in. Left until last so that a width asked for
        # here is the width drawn, rather than something the resizing above has scaled.
        self.set_style(
            fill_color=color or fill_color,
            fill_opacity=fill_opacity,
            stroke_color=color or stroke_color,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
        )

    def init_svg_mobject(self) -> None:
        # Clipping masks are attached to their SVG targets. Copying cached SVG submobjects
        # one at a time would leave Mask effects pointing at cached objects, so clipped SVGs
        # are parsed afresh instead of using the geometry cache.
        svg_root = ET.fromstring(self.svg_string)
        self._collect_svg_clip_metadata(svg_root)
        self._collect_svg_gradient_metadata(svg_root)
        hash_val = hash_obj(self.hash_seed)
        contains_images = any(
            element.tag.rsplit("}", 1)[-1] == "image"
            for element in svg_root.iter()
        )
        # A relative href can resolve differently for identical SVG markup loaded from two
        # directories, so SVGs containing images are not reused from the geometry cache.
        cacheable = (
            not contains_images
            and not self._svg_clip_paths
            and not self._svg_clip_assignments
        )
        if cacheable and hash_val in SVG_HASH_TO_MOB_MAP:
            submobs = [sm.copy() for sm in SVG_HASH_TO_MOB_MAP[hash_val]]
        else:
            submobs = self.mobjects_from_svg_string(self.svg_string)
            if cacheable:
                SVG_HASH_TO_MOB_MAP[hash_val] = [sm.copy() for sm in submobs]

        self.add(*submobs)
        self.flip(RIGHT)  # Flip y

    @property
    def hash_seed(self) -> tuple:
        # Returns data which can uniquely represent the result of `init_points`.
        # The hashed value of it is stored as a key in `SVG_HASH_TO_MOB_MAP`.
        return (
            self.__class__.__name__,
            self.svg_default,
            self.path_string_config,
            self.svg_string
        )

    def _collect_svg_clip_metadata(self, root: ET.Element) -> None:
        self._svg_root_attributes = dict(root.attrib)
        self._svg_clip_paths = {}
        self._svg_clip_assignments = {}
        self._svg_clip_shape_ids = set()

        def local_name(tag: str) -> str:
            return tag.rsplit("}", 1)[-1]

        for element in root.iter():
            if local_name(element.tag) != "clipPath":
                continue
            clip_id = element.attrib.get("id")
            if not clip_id:
                continue
            self._svg_clip_paths[clip_id] = deepcopy(element)
            for child in element.iter():
                child_id = child.attrib.get("id")
                if child_id:
                    self._svg_clip_shape_ids.add(child_id)

        graphic_tags = {
            "path", "rect", "image", "circle", "ellipse", "line",
            "polyline", "polygon", "use",
        }
        generated_id = 0

        def clip_ref(element: ET.Element) -> str | None:
            value = element.attrib.get("clip-path", "")
            style = element.attrib.get("style", "")
            match_style = re.search(r"(?:^|;)\s*clip-path\s*:\s*([^;]+)", style)
            if match_style:
                value = match_style.group(1)
            match = re.search(r"url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)", value)
            return match.group(1) if match else None

        def visit(element: ET.Element, inherited_clip: str | None = None) -> None:
            nonlocal generated_id
            name = local_name(element.tag)
            # Definitions are recorded above; they are not visible geometry and their
            # descendants must not inherit a clipping rule from the surrounding document.
            if name in ("clipPath", "defs"):
                return

            own_value = element.attrib.get("clip-path", "")
            own_style = element.attrib.get("style", "")
            has_style_clip = re.search(r"(?:^|;)\s*clip-path\s*:", own_style) is not None
            own_clip = clip_ref(element)
            active_clip = own_clip if own_clip is not None else (
                None if own_value == "none" or has_style_clip else inherited_clip
            )

            # svgelements exposes leaf graphic objects by their id, but SVG documents often
            # omit ids. Give clipped shapes private, deterministic ids so the generated
            # mobject can still be connected to its clip mask.
            if active_clip and name in graphic_tags:
                element_id = element.attrib.get("id")
                if not element_id:
                    generated_id += 1
                    element_id = f"__manim_clip_target_{generated_id:06d}"
                    element.set("id", element_id)
                self._svg_clip_assignments[element_id] = active_clip

            for child in element:
                visit(child, active_clip)

        visit(root)

    @staticmethod
    def _svg_style_attributes(element: ET.Element) -> dict[str, str]:
        """Return presentation attributes with inline style declarations applied last."""
        attrs = dict(element.attrib)
        for declaration in attrs.get("style", "").split(";"):
            name, separator, value = declaration.partition(":")
            if separator and name.strip():
                attrs[name.strip()] = value.replace("!important", "").strip()
        return attrs

    def _collect_svg_gradient_metadata(self, root: ET.Element) -> None:
        """Collect SVG gradient definitions and the shapes which reference them."""
        self._svg_gradients = {}
        self._svg_gradient_assignments = {}

        def local_name(element: ET.Element) -> str:
            return element.tag.rsplit("}", 1)[-1]

        for element in root.iter():
            if local_name(element) in ("linearGradient", "radialGradient"):
                gradient_id = element.attrib.get("id")
                if gradient_id:
                    self._svg_gradients[gradient_id] = deepcopy(element)

        if not self._svg_gradients:
            return

        graphic_tags = {
            "path", "rect", "image", "circle", "ellipse", "line",
            "polyline", "polygon", "use",
        }
        generated_id = 0

        def opacity_value(value: str | None, default: float = 1.0) -> float:
            if value is None:
                return default
            try:
                value = value.strip()
                result = float(value[:-1]) / 100.0 if value.endswith("%") else float(value)
                return float(np.clip(result, 0.0, 1.0))
            except (TypeError, ValueError):
                return default

        def gradient_ref(value: str | None) -> str | None:
            if not value:
                return None
            match = re.search(r"url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)", value)
            return match.group(1) if match else None

        def visit(
            element: ET.Element,
            inherited_gradient: str | None = None,
            inherited_fill_opacity: float = 1.0,
            inherited_opacity: float = 1.0,
        ) -> None:
            nonlocal generated_id
            name = local_name(element)
            # Definitions never paint by themselves.
            if name in ("defs", "clipPath", "linearGradient", "radialGradient", "mask", "pattern"):
                return

            attrs = self._svg_style_attributes(element)
            active_gradient = inherited_gradient
            if "fill" in attrs:
                active_gradient = gradient_ref(attrs.get("fill"))
                if active_gradient not in self._svg_gradients:
                    active_gradient = None

            current_fill_opacity = opacity_value(
                attrs.get("fill-opacity"),
                inherited_fill_opacity,
            )
            current_opacity = inherited_opacity * opacity_value(attrs.get("opacity"), 1.0)

            if name in graphic_tags and active_gradient is not None:
                element_id = attrs.get("id")
                if not element_id:
                    generated_id += 1
                    element_id = f"__manim_gradient_target_{generated_id:06d}"
                    element.set("id", element_id)
                self._svg_gradient_assignments[element_id] = (
                    active_gradient,
                    current_fill_opacity * current_opacity,
                )

            for child in element:
                visit(child, active_gradient, current_fill_opacity, current_opacity)

        visit(root)

    def _resolve_svg_gradient(
        self,
        gradient_id: str,
        visited: set[str] | None = None,
    ) -> dict | None:
        """Resolve href inheritance without depending on svgelements' paint parser."""
        if visited is None:
            visited = set()
        if gradient_id in visited:
            log.warning("Circular SVG gradient reference: %s", gradient_id)
            return None
        element = self._svg_gradients.get(gradient_id)
        if element is None:
            return None
        visited = {*visited, gradient_id}
        href = element.attrib.get("href") or element.attrib.get(
            "{http://www.w3.org/1999/xlink}href"
        )
        base_id = href[1:] if href and href.startswith("#") else None
        base = self._resolve_svg_gradient(base_id, visited) if base_id else None

        attrs = {} if base is None else dict(base["attrs"])
        attrs.update({
            key: value
            for key, value in element.attrib.items()
            if key not in ("id", "href", "{http://www.w3.org/1999/xlink}href")
        })
        own_stops = [
            deepcopy(child)
            for child in list(element)
            if child.tag.rsplit("}", 1)[-1] == "stop"
        ]
        stops = own_stops if own_stops else ([] if base is None else base["stops"])
        return {
            "type": element.tag.rsplit("}", 1)[-1],
            "attrs": attrs,
            "stops": stops,
        }

    def _apply_svg_gradient(
        self,
        mob: VMobject,
        shape_id: str,
        shape: se.GraphicObject,
    ) -> None:
        assignment = self._svg_gradient_assignments.get(shape_id)
        if assignment is None:
            return
        gradient_id, inherited_opacity = assignment
        gradient = self._resolve_svg_gradient(gradient_id)
        if not gradient or not gradient["stops"]:
            log.warning("SVG gradient %r has no usable color stops", gradient_id)
            return

        attrs = gradient["attrs"]
        is_radial = gradient["type"] == "radialGradient"
        units = attrs.get("gradientUnits", "objectBoundingBox")
        object_bbox = units != "userSpaceOnUse"
        bbox = mob.get_bounding_box()
        lower, upper = bbox[0], bbox[2]
        size = upper - lower

        view_box = self._svg_root_attributes.get("viewBox", "").replace(",", " ").split()
        if len(view_box) == 4:
            try:
                view_min_x, view_min_y, view_width, view_height = map(float, view_box)
            except ValueError:
                view_min_x, view_min_y, view_width, view_height = 0.0, 0.0, size[0], size[1]
        else:
            view_min_x, view_min_y, view_width, view_height = 0.0, 0.0, size[0], size[1]
        if view_width <= 0:
            view_width = size[0]
        if view_height <= 0:
            view_height = size[1]

        number_pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")

        def parse_number(value: str | None, default: float = 0.0) -> float:
            if value is None:
                return default
            match = number_pattern.search(value.strip())
            if not match:
                return default
            try:
                return float(match.group(0))
            except ValueError:
                return default

        def coordinate(value: str | None, axis: int, default: str) -> float:
            value = default if value is None else value.strip()
            fraction = value.endswith("%")
            number = parse_number(value, 0.0)
            if object_bbox:
                return number / 100.0 if fraction else number
            if fraction:
                origin = view_min_x if axis == 0 else view_min_y
                extent = view_width if axis == 0 else view_height
                return origin + extent * number / 100.0
            return number

        def opacity(value: str | None, default: float = 1.0) -> float:
            if value is None:
                return default
            result = parse_number(value, default)
            if value.strip().endswith("%"):
                result /= 100.0
            return float(np.clip(result, 0.0, 1.0))

        gradient_matrix = None
        transform_text = attrs.get("gradientTransform")
        if transform_text:
            try:
                gradient_matrix = se.Matrix(transform_text)
            except Exception as error:
                log.warning("Ignoring unsupported gradientTransform %r: %s", transform_text, error)

        def apply_matrix(x: float, y: float, matrix) -> tuple[float, float]:
            if matrix is None:
                return x, y
            return (
                matrix.a * x + matrix.c * y + matrix.e,
                matrix.b * x + matrix.d * y + matrix.f,
            )

        def to_point(x: float, y: float) -> np.ndarray:
            # gradientTransform acts in the gradient's coordinate system. objectBoundingBox
            # coordinates are then expanded into the target's local bounding box.
            x, y = apply_matrix(x, y, gradient_matrix)
            if object_bbox:
                x = lower[0] + x * size[0]
                y = lower[1] + y * size[1]
            return np.array([x, y, 0.0], dtype=float)

        colors: list[str] = []
        offsets: list[float] = []
        opacities: list[float] = []
        for index, stop in enumerate(gradient["stops"]):
            stop_attrs = self._svg_style_attributes(stop)
            offset_value = stop_attrs.get("offset", "0")
            offset = parse_number(offset_value, 0.0)
            if offset_value.strip().endswith("%"):
                offset /= 100.0
            color = stop_attrs.get("stop-color", "black").strip()
            if color.lower() == "currentcolor":
                color = "black"
            colors.append(color)
            offsets.append(float(np.clip(offset, 0.0, 1.0)))
            opacities.append(
                opacity(stop_attrs.get("stop-opacity"), 1.0) * inherited_opacity
            )
        offsets = np.maximum.accumulate(np.asarray(offsets, dtype=float)).tolist()

        spread_method = attrs.get("spreadMethod", "pad").lower()
        if spread_method not in ("pad", "repeat", "reflect"):
            spread_method = "pad"

        if not is_radial:
            x1 = coordinate(attrs.get("x1"), 0, "0%")
            y1 = coordinate(attrs.get("y1"), 1, "0%")
            x2 = coordinate(attrs.get("x2"), 0, "100%")
            y2 = coordinate(attrs.get("y2"), 1, "0%")
            mob.set_fill_gradient(
                colors,
                offsets=offsets,
                opacities=opacities,
                gradient_type="linear",
                start=to_point(x1, y1),
                end=to_point(x2, y2),
                spread_method=spread_method,
                recurse=False,
            )
            return

        cx = coordinate(attrs.get("cx"), 0, "50%")
        cy = coordinate(attrs.get("cy"), 1, "50%")
        radius_value = attrs.get("r", "50%").strip()
        radius_fraction = radius_value.endswith("%")
        radius_amount = parse_number(radius_value, 0.5)
        if radius_fraction:
            radius_amount /= 100.0
        if object_bbox:
            radius_amount = radius_amount
        elif radius_fraction:
            radius_amount *= float(np.sqrt((view_width * view_width + view_height * view_height) / 2.0))
        # Keep the focal point and both outer-radius axes in the same transformed
        # coordinate system, so the shader can solve the ray/circle intersection.
        fx = coordinate(attrs.get("fx"), 0, attrs.get("cx", "50%"))
        fy = coordinate(attrs.get("fy"), 1, attrs.get("cy", "50%"))
        center = to_point(cx, cy)
        focus = to_point(fx, fy)
        radial_x = to_point(cx + radius_amount, cy)
        radial_y = to_point(cx, cy + radius_amount)
        mob.set_fill_gradient(
            colors,
            offsets=offsets,
            opacities=opacities,
            gradient_type="radial",
            center=center,
            focus=focus,
            radial_axis_x=radial_x,
            radial_axis_y=radial_y,
            spread_method=spread_method,
            recurse=False,
        )

    def mobjects_from_svg_string(self, svg_string: str) -> list[VMobject]:
        original_root = ET.fromstring(svg_string)
        self._collect_svg_clip_metadata(original_root)
        self._collect_svg_gradient_metadata(original_root)

        # Keep clipping/gradient definitions in our metadata, but do not let the generic
        # element iterator emit their definition children as visible shapes.
        parse_root = deepcopy(original_root)
        def remove_clip_path_definitions(parent: ET.Element) -> None:
            for child in list(parent):
                if child.tag.rsplit("}", 1)[-1] in ("clipPath", "linearGradient", "radialGradient"):
                    parent.remove(child)
                else:
                    remove_clip_path_definitions(child)
        remove_clip_path_definitions(parse_root)

        element_tree = ET.ElementTree(parse_root)
        new_tree = self.modify_xml_tree(element_tree)

        # New svg based on tree contents
        data_stream = io.BytesIO()
        new_tree.write(data_stream)
        data_stream.seek(0)
        svg = se.SVG.parse(data_stream)
        data_stream.close()

        return self.mobjects_from_svg(svg)

    def file_name_to_svg_string(self, file_name: str) -> str:
        path = Path(get_full_vector_image_path(file_name))
        self._svg_source_path = path
        return path.read_text()

    def modify_xml_tree(self, element_tree: ET.ElementTree) -> ET.ElementTree:
        config_style_attrs = self.generate_config_style_dict()
        style_keys = (
            "fill",
            "fill-opacity",
            "stroke",
            "stroke-opacity",
            "stroke-width",
            "style"
        )
        root = element_tree.getroot()
        style_attrs = {
            k: v
            for k, v in root.attrib.items()
            if k in style_keys
        }

        # Ignore other attributes in case that svgelements cannot parse them
        SVG_XMLNS = "{http://www.w3.org/2000/svg}"
        new_root = ET.Element("svg")
        config_style_node = ET.SubElement(new_root, f"{SVG_XMLNS}g", config_style_attrs)
        root_style_node = ET.SubElement(config_style_node, f"{SVG_XMLNS}g", style_attrs)
        root_style_node.extend(root)
        return ET.ElementTree(new_root)

    def generate_config_style_dict(self) -> dict[str, str]:
        keys_converting_dict = {
            "fill": ("color", "fill_color"),
            "fill-opacity": ("opacity", "fill_opacity"),
            "stroke": ("color", "stroke_color"),
            "stroke-opacity": ("opacity", "stroke_opacity"),
            "stroke-width": ("stroke_width",)
        }
        svg_default_dict = self.svg_default
        result = {}
        for svg_key, style_keys in keys_converting_dict.items():
            for style_key in style_keys:
                if svg_default_dict[style_key] is None:
                    continue
                result[svg_key] = str(svg_default_dict[style_key])
        return result

    def mobjects_from_svg(
        self,
        svg: se.SVG,
        apply_clip_paths: bool = True,
    ) -> list[VMobject]:
        result: list[VMobject] = []
        mobjects_by_id: dict[str, VMobject] = {}
        image_type = getattr(se, "Image", getattr(se, "SVGImage", None))

        for shape in svg.elements():
            shape_id = (
                getattr(shape, "id", None)
                or getattr(shape, "values", {}).get("id")
            )
            if shape_id is not None:
                shape_id = str(shape_id)
            if apply_clip_paths and shape_id and shape_id in self._svg_clip_shape_ids:
                # Clip geometry is used as a mask, not as a visible SVG object.
                continue
            if isinstance(shape, (se.Group, se.Use)):
                continue

            is_image = image_type is not None and isinstance(shape, image_type)
            if is_image:
                try:
                    mob = self.image_to_mobject(shape)
                except (OSError, ValueError) as error:
                    log.warning("Could not load SVG <image> %r: %s", shape_id, error)
                    continue
            elif isinstance(shape, se.Path):
                mob = self.path_to_mobject(shape, svg)
            elif isinstance(shape, se.SimpleLine):
                mob = self.line_to_mobject(shape)
            elif isinstance(shape, se.Rect):
                mob = self.rect_to_mobject(shape)
            elif isinstance(shape, (se.Circle, se.Ellipse)):
                mob = self.ellipse_to_mobject(shape)
            elif isinstance(shape, se.Polygon):
                mob = self.polygon_to_mobject(shape)
            elif isinstance(shape, se.Polyline):
                mob = self.polyline_to_mobject(shape)
            elif type(shape) == se.SVGElement:
                continue
            else:
                log.warning("Unsupported element type: %s", type(shape))
                continue

            if not mob.has_points():
                continue
            if isinstance(shape, se.GraphicObject) and not is_image:
                self.apply_style_to_mobject(mob, shape)
            # Set gradient coordinates before applying the element transform. The gradient's
            # point uniforms then follow the very same transform as the actual VMobject.
            if not is_image and shape_id in self._svg_gradient_assignments:
                self._apply_svg_gradient(mob, shape_id, shape)
            if isinstance(shape, se.Transformable) and shape.apply:
                self.handle_transform(mob, shape.transform)
            result.append(mob)
            if shape_id:
                mobjects_by_id[shape_id] = mob

        if apply_clip_paths and self._svg_clip_assignments:
            for shape_id, clip_id in self._svg_clip_assignments.items():
                target = mobjects_by_id.get(shape_id)
                if target is None or clip_id not in self._svg_clip_paths:
                    continue
                mask = self._clip_path_to_mobject(clip_id)
                if mask is None or not len(mask.get_all_points()):
                    log.warning(
                        "Could not build SVG clipPath %r for element %r",
                        clip_id, shape_id,
                    )
                    continue
                clip_path = self._svg_clip_paths[clip_id]
                if clip_path.attrib.get("clipPathUnits", "userSpaceOnUse") == "objectBoundingBox":
                    bbox = target.get_bounding_box()
                    lower, upper = bbox[0], bbox[2]
                    size = upper - lower
                    mask.scale([size[0], size[1], 1.0], about_point=np.zeros(3))
                    mask.shift(lower)
                mask.set_fill(color="#FFFFFF", opacity=1.0)
                mask.set_stroke(width=0.0, opacity=0.0)
                # Keep the mask transform-linked to its target without making it a
                # child of the textured object. A child mask becomes part of the target's
                # rendered layer, so the Mask post-process receives the unmasked rectangle
                # (and the mask can be drawn into the same layer). A wrapper keeps both
                # objects under the same parent transforms while the mask stays a sibling.
                target.add_mask(mask)
                wrapper = VGroup(target, mask)
                for index, mob in enumerate(result):
                    if mob is target:
                        result[index] = wrapper
                        break
        return result

    def image_to_mobject(self, image) -> Rectangle:
        width = float(image.width)
        height = float(image.height)
        mob = Rectangle(width=width, height=height)
        mob.shift(_convert_point_to_3d(
            float(image.x) + width / 2,
            float(image.y) + height / 2,
        ))
        mob.set_fill(color="#FFFFFF", opacity=1.0)
        image_opacity = getattr(image, "opacity", None)
        if image_opacity is not None:
            mob.set_fill(opacity=float(image_opacity))
        mob.set_stroke(width=0.0, opacity=0.0)
        mob.set_texture(self._texture_source_from_href(self._get_image_href(image)), flip_y=False)
        return mob

    @staticmethod
    def _get_image_href(image) -> str:
        href = getattr(image, "href", None)
        if not href:
            href = getattr(image, "url", None)
        if not href:
            values = getattr(image, "values", {})
            href = values.get("href") or values.get("xlink:href")
        if not href:
            raise ValueError("SVG <image> has no href or xlink:href")
        return str(href)

    def _texture_source_from_href(self, href: str) -> ImageFile | PixelArray:
        href = href.strip()
        if href.startswith("data:"):
            header, separator, payload = href.partition(",")
            if not separator:
                raise ValueError("Malformed SVG image data URI")
            raw = base64.b64decode(payload) if ";base64" in header.lower() else unquote_to_bytes(payload)
            try:
                from PIL import Image as PILImage
                with PILImage.open(io.BytesIO(raw)) as image:
                    pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            except ImportError as error:
                raise RuntimeError("Pillow is required to decode embedded SVG images") from error
            except Exception as error:
                raise ValueError(f"Unsupported embedded SVG image format: {error}") from error
            return PixelArray(pixels)

        parsed = urlparse(href)
        if parsed.scheme in ("http", "https") or href.startswith("//"):
            raise ValueError("Remote SVG images must be embedded as a data URI or bundled as a local file")
        if parsed.scheme == "file":
            path = Path(unquote(parsed.path))
        else:
            path = Path(unquote(parsed.path or href))

        if not path.is_absolute() and self._svg_source_path is not None:
            relative_path = self._svg_source_path.parent / path
            if relative_path.is_file():
                path = relative_path
        if not path.is_file() and not parsed.scheme:
            try:
                path = Path(get_full_raster_image_path(str(path)))
            except Exception:
                pass
        if not path.is_file():
            raise FileNotFoundError(f"SVG image resource was not found: {href}")
        return ImageFile(str(path))

    def _clip_path_to_mobject(self, clip_id: str) -> VGroup | None:
        clip_path = self._svg_clip_paths.get(clip_id)
        if clip_path is None:
            return None

        # Parse the clipping geometry as ordinary SVG shapes while retaining the same user
        # coordinate space. The clipPath itself is not copied into the temporary SVG, so this
        # does not recursively apply the same clipping rule.
        root = ET.Element("svg")
        group_attrs = {}
        if clip_path.attrib.get("transform"):
            group_attrs["transform"] = clip_path.attrib["transform"]
        group = ET.SubElement(root, "g", group_attrs)
        for child in list(clip_path):
            group.append(deepcopy(child))
        stream = io.BytesIO(ET.tostring(root))
        try:
            clip_svg = se.SVG.parse(stream)
        finally:
            stream.close()

        shapes = self.mobjects_from_svg(clip_svg, apply_clip_paths=False)
        shapes = [
            shape for shape in shapes
            if any(member.has_points() for member in shape.get_family())
        ]
        if not shapes:
            return None
        return VGroup(*shapes)

    @staticmethod
    def handle_transform(mob: VMobject, matrix: se.Matrix) -> VMobject:
        mat = np.array([
            [matrix.a, matrix.c],
            [matrix.b, matrix.d]
        ])
        vec = np.array([matrix.e, matrix.f, 0.0])
        mob.apply_matrix(mat)
        mob.shift(vec)
        return mob

    @staticmethod
    def apply_style_to_mobject(
        mob: VMobject,
        shape: se.GraphicObject
    ) -> VMobject:
        # svgelements hands back a stroke width of 1.0 even for a shape painted with
        # `stroke: none`, so the width is only taken when a stroke color came with it.
        # Otherwise it is zeroed, which is what an unstroked shape should draw as.
        stroke = getattr(shape, "stroke", None)
        fill = getattr(shape, "fill", None)
        stroke_color = getattr(stroke, "hexrgb", None)
        fill_color = getattr(fill, "hexrgb", None)
        has_stroke = stroke is not None and stroke_color is not None
        mob.set_style(
            stroke_width=shape.stroke_width if has_stroke else 0.0,
            stroke_color=stroke_color if has_stroke else None,
            stroke_opacity=getattr(stroke, "opacity", None) if has_stroke else None,
            fill_color=fill_color,
            fill_opacity=getattr(fill, "opacity", None),
        )
        return mob

    def path_to_mobject(self, path: se.Path, svg: se.SVG) -> VMobjectFromSVGPath:
        if path.id in svg.objects:
            # If this path reuses a referenced definition (<use>), build the mobject from
            # the original geometry.
            # We apply the transform ourselves so we (1) keep the full precision of the 
            # reference and (2) only store one entry in PATH_TO_POINTS.
            ref_path = svg.objects[path.id]
            mob = VMobjectFromSVGPath(ref_path, **self.path_string_config)
            if 'transform' in path.values:
                matrix = se.Matrix(path.values['transform'])
                rotation = np.array([[matrix.a, matrix.b],
                                     [matrix.c, matrix.d]])
                translation = np.array([[matrix.e, matrix.f]])
                mob.apply_points_function(
                    lambda points: np.concatenate([points[:, :2] @ rotation + translation,
                                                   points[:, [2]]],
                                                  axis=1),
                    about_point=None,
                    about_edge=None,
                    works_on_bounding_box=False)
            return mob
        else:
            return VMobjectFromSVGPath(path, **self.path_string_config)

    def line_to_mobject(self, line: se.SimpleLine) -> Line:
        return Line(
            start=_convert_point_to_3d(line.x1, line.y1),
            end=_convert_point_to_3d(line.x2, line.y2)
        )

    def rect_to_mobject(self, rect: se.Rect) -> Rectangle:
        if rect.rx == 0 or rect.ry == 0:
            mob = Rectangle(
                width=rect.width,
                height=rect.height,
            )
        else:
            mob = RoundedRectangle(
                width=rect.width,
                height=rect.height * rect.rx / rect.ry,
                corner_radius=rect.rx
            )
            mob.stretch_to_fit_height(rect.height)
        mob.shift(_convert_point_to_3d(
            rect.x + rect.width / 2,
            rect.y + rect.height / 2
        ))
        return mob

    def ellipse_to_mobject(self, ellipse: se.Circle | se.Ellipse) -> Circle:
        mob = Circle(radius=ellipse.rx)
        mob.stretch_to_fit_height(2 * ellipse.ry)
        mob.shift(_convert_point_to_3d(
            ellipse.cx, ellipse.cy
        ))
        return mob

    def polygon_to_mobject(self, polygon: se.Polygon) -> Polygon:
        points = [
            _convert_point_to_3d(*point)
            for point in polygon
        ]
        return Polygon(*points)

    def polyline_to_mobject(self, polyline: se.Polyline) -> Polyline:
        points = [
            _convert_point_to_3d(*point)
            for point in polyline
        ]
        return Polyline(*points)

    def text_to_mobject(self, text: se.Text):
        pass


class VMobjectFromSVGPath(VMobject):
    def __init__(
        self,
        path_obj: se.Path,
        **kwargs
    ):
        # caches (transform.inverse(), rot, shift)
        self.transform_cache: tuple[se.Matrix, np.ndarray, np.ndarray] | None = None

        self.path_obj = path_obj
        super().__init__(**kwargs)

    def init_points(self) -> None:
        # After a given svg_path has been converted into points, the result
        # will be saved so that future calls for the same pathdon't need to
        # retrace the same computation.
        path_string = self.path_obj.d()
        if path_string not in PATH_TO_POINTS:
            self.handle_commands()
            # Save for future use
            PATH_TO_POINTS[path_string] = self.get_points().copy()
        else:
            points = PATH_TO_POINTS[path_string]
            self.set_points(points)

    def handle_commands(self) -> None:
        segment_class_to_func_map = {
            se.Move: (self.start_new_path, ("end",)),
            se.Close: (self.close_path, ()),
            se.Line: (lambda p: self.add_line_to(p, allow_null_line=False), ("end",)),
            se.QuadraticBezier: (lambda c, e: self.add_quadratic_bezier_curve_to(c, e, allow_null_curve=False), ("control", "end")),
            se.CubicBezier: (self.add_cubic_bezier_curve_to, ("control1", "control2", "end"))
        }
        for segment in self.path_obj:
            segment_class = segment.__class__
            if segment_class is se.Arc:
                self.handle_arc(segment)
            else:
                func, attr_names = segment_class_to_func_map[segment_class]
                points = [
                    _convert_point_to_3d(*segment.__getattribute__(attr_name))
                    for attr_name in attr_names
                ]
                func(*points)

        # Get rid of the side effect of trailing "Z M" commands.
        if self.has_new_path_started():
            self.resize_points(self.get_num_points() - 2)

    def handle_arc(self, arc: se.Arc) -> None:
        if self.transform_cache is not None:
            transform, rot, shift = self.transform_cache
        else:
            # The transform obtained in this way considers the combined effect
            # of all parent group transforms in the SVG.
            # Therefore, the arc can be transformed inversely using this transform
            # to correctly compute the arc path before transforming it back.
            transform = se.Matrix(self.path_obj.values.get('transform', ''))
            rot = np.array([
                [transform.a, transform.c],
                [transform.b, transform.d]
            ])
            shift = np.array([transform.e, transform.f, 0])
            transform.inverse()
            self.transform_cache = (transform, rot, shift)

        # Apply inverse transformation to the arc so that its path can be correctly computed
        arc *= transform

        # The value of n_components is chosen based on the implementation of VMobject.arc_to
        n_components = int(np.ceil(8 * abs(arc.sweep) / TAU))

        # Obtain the required angular segments on the unit circle
        arc_points = quadratic_bezier_points_for_arc(arc.sweep, n_components)
        arc_points @= np.array(rotation_about_z(arc.get_start_t())).T

        # Transform to an ellipse, considering rotation and translating the ellipse center
        arc_points[:, 0] *= arc.rx
        arc_points[:, 1] *= arc.ry
        arc_points @= np.array(rotation_about_z(arc.get_rotation().as_radians)).T
        arc_points += [*arc.center, 0]

        # Transform back
        arc_points[:, :2] @= rot.T
        arc_points += shift

        self.append_points(arc_points[1:])
