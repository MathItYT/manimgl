from __future__ import annotations

import base64
from copy import deepcopy
from pathlib import Path
import re
from urllib.parse import unquote, unquote_to_bytes, urlparse
from xml.etree import ElementTree as ET

import numpy as np
import svgelements as se
import io
from PIL import Image as PILImage

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
        # Clipping masks refer to sibling mobjects. Copying cached SVG submobjects one at a
        # time would leave Mask effects pointing at the cached siblings, so clipped SVGs are
        # parsed afresh instead of using the geometry cache.
        self._collect_svg_clip_metadata(ET.fromstring(self.svg_string))
        hash_val = hash_obj(self.hash_seed)
        cacheable = not self._svg_clip_paths and not self._svg_clip_assignments
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

        def clip_ref(element: ET.Element) -> str | None:
            value = element.attrib.get("clip-path", "")
            style = element.attrib.get("style", "")
            match_style = re.search(r"(?:^|;)\\s*clip-path\\s*:\\s*([^;]+)", style)
            if match_style:
                value = match_style.group(1)
            match = re.search(r"url\\(\\s*['\\"]?#([^)'\\"]+)['\\"]?\\s*\\)", value)
            return match.group(1) if match else None

        def visit(element: ET.Element, inherited_clip: str | None = None) -> None:
            own_value = element.attrib.get("clip-path", "")
            own_style = element.attrib.get("style", "")
            has_style_clip = re.search(r"(?:^|;)\\s*clip-path\\s*:", own_style) is not None
            own_clip = clip_ref(element)
            active_clip = own_clip if own_clip is not None else (None if own_value == "none" or has_style_clip else inherited_clip)
            element_id = element.attrib.get("id")
            if element_id and active_clip:
                self._svg_clip_assignments[element_id] = active_clip
            for child in element:
                visit(child, active_clip)

        visit(root)

    def mobjects_from_svg_string(self, svg_string: str) -> list[VMobject]:
        original_root = ET.fromstring(svg_string)
        self._collect_svg_clip_metadata(original_root)
        element_tree = ET.ElementTree(original_root)
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
        mobjects_by_id: dict[str, VMobject] = []
        image_type = getattr(se, "Image", None)

        for shape in svg.elements():
            shape_id = getattr(shape, "id", None)
            if shape_id and shape_id in self._svg_clip_shape_ids:
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
            if isinstance(shape, se.Transformable) and shape.apply:
                self.handle_transform(mob, shape.transform)
            result.append(mob)
            if shape_id:
                mobjects_by_id[shape_id] = mob

        if apply_clip_paths and self._svg_clip_assignments:
            clip_mobjects: list[VMobject] = []
            for shape_id, clip_id in self._svg_clip_assignments.items():
                target = mobjects_by_id.get(shape_id)
                if target is None or clip_id not in self._svg_clip_paths:
                    continue
                mask = self._clip_path_to_mobject(clip_id)
                if mask is None or not mask.has_points():
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
                target.add_mask(mask)
                # Keep masks in the SVG family (hidden) so parent flips, shifts and scales
                # also transform them; Mask's postprocessor renders them separately.
                clip_mobjects.append(mask)
            result.extend(clip_mobjects)
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
                with PILImage.open(io.BytesIO(raw)) as image:
                    pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
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
        shapes = [shape for shape in shapes if shape.has_points()]
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
        has_stroke = shape.stroke is not None and shape.stroke.hexrgb is not None
        mob.set_style(
            stroke_width=shape.stroke_width if has_stroke else 0.0,
            stroke_color=shape.stroke.hexrgb if has_stroke else None,
            stroke_opacity=shape.stroke.opacity if has_stroke else None,
            fill_color=shape.fill.hexrgb,
            fill_opacity=shape.fill.opacity
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
