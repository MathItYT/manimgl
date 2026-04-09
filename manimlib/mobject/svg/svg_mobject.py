from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np
import svgelements as se
import io
from pathlib import Path

from manimlib.constants import RIGHT, UP, BLACK
from manimlib.constants import TAU
from manimlib.logger import log
from manimlib.mobject.geometry import Circle
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Polygon
from manimlib.mobject.geometry import Polyline
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.geometry import RoundedRectangle
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.bezier import quadratic_bezier_points_for_arc
from manimlib.utils.images import get_full_vector_image_path
from manimlib.utils.iterables import hash_obj
from manimlib.utils.space_ops import rotation_about_z

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manimlib.typing import ManimColor, Vect3Array


SVG_HASH_TO_MOB_MAP: dict[int, list[VMobject]] = {}
PATH_TO_POINTS: dict[str, Vect3Array] = {}


def get_svg_content_height(svg_string: str) -> float:
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
    height: float | None = 2.0
    width: float | None = None

    def __init__(
        self,
        file_name: str = "",
        svg_string: str = "",
        should_center: bool = True,
        height: float | None = None,
        width: float | None = None,
        unbatch: bool | str = False,
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
        needs_flip: bool = True,
        **kwargs
    ):
        self.needs_flip = needs_flip
        self.unbatch = unbatch
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

        # Rather than passing style into super().__init__
        # do it after svg has been taken in
        self.set_style(
            fill_color=color or fill_color,
            fill_opacity=fill_opacity,
            stroke_color=color or stroke_color,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
        )

        # Control batching behaviour for complex SVGs.
        #
        # - unbatch=False: overlap-safe batching (fast in the common case where
        #   shapes are spatially disjoint, while automatically splitting into a
        #   few batches when overlaps would cause winding/blending artifacts).
        # - unbatch=True: fully unbatch each VMobject (safest / correct, but
        #   can be significantly slower due to many draw calls).
        # - unbatch='style': split batching by (approximate) style signature
        #   (mainly fill/stroke), which prevents cross-layer winding/blending
        #   interference while still batching within layers.
        # - unbatch='overlap'/'auto'/'safe': explicit alias for overlap-safe batching.
        # - unbatch='max'/'raw': disable uid-based splitting (maximum batching; may show artifacts).
        self._apply_unbatch_mode()

        # Initialize position
        height = height or self.height
        width = width or self.width

        if should_center:
            self.center()
        if height is not None:
            self.set_height(height)
        if width is not None:
            self.set_width(width)

    def init_svg_mobject(self) -> None:
        hash_val = hash_obj(self.hash_seed)
        if hash_val in SVG_HASH_TO_MOB_MAP:
            submobs = [sm.copy() for sm in SVG_HASH_TO_MOB_MAP[hash_val]]
        else:
            submobs = self.mobjects_from_svg_string(self.svg_string)
            SVG_HASH_TO_MOB_MAP[hash_val] = [sm.copy() for sm in submobs]

        self.add(*submobs)
        if self.needs_flip:
            self.flip(RIGHT)  # Flip y

    @staticmethod
    def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (
            (ax1 < bx0) or
            (ax0 > bx1) or
            (ay1 < by0) or
            (ay0 > by1)
        )

    @staticmethod
    def _rect_to_cell_range(
        rect: tuple[float, float, float, float],
        cell_size: float,
    ) -> tuple[int, int, int, int]:
        import math

        x0, y0, x1, y1 = rect
        inv = 1.0 / cell_size
        ix0 = int(math.floor(x0 * inv))
        ix1 = int(math.floor(x1 * inv))
        iy0 = int(math.floor(y0 * inv))
        iy1 = int(math.floor(y1 * inv))
        return ix0, ix1, iy0, iy1

    def _apply_overlap_safe_batching(self, mobs: list[VMobject]) -> None:
        # Assign `_svg_uid` so that any two point mobjects whose
        # axis-aligned bounding boxes overlap end up in different batches.
        #
        # This preserves correctness for ManimGL's winding-based fill pass
        # while keeping batching maximal when shapes are spatially disjoint
        # (common for text glyphs).

        rects: list[tuple[float, float, float, float]] = []
        sizes: list[float] = []

        for mob in mobs:
            bb = mob.get_bounding_box()
            x0, y0 = float(bb[0, 0]), float(bb[0, 1])
            x1, y1 = float(bb[2, 0]), float(bb[2, 1])
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

            # Expand slightly to account for AA and curve hull slack
            data = mob.data if mob.get_num_points() > 0 else mob._data_defaults
            stroke_w = float(data["stroke_width"][0, 0]) if "stroke_width" in data.dtype.names else 0.0
            buff = 1e-3 + 0.5 * max(0.0, stroke_w)
            x0 -= buff
            y0 -= buff
            x1 += buff
            y1 += buff

            rects.append((x0, y0, x1, y1))
            sizes.append(max(x1 - x0, y1 - y0))

        if not rects:
            return

        # Choose a hashing grid size around the median bbox size.
        cell_size = float(np.median(sizes)) if sizes else 1.0
        cell_size = max(cell_size, 1e-2)

        # Each group tracks a spatial hash for overlap queries.
        groups: list[dict[str, object]] = []

        for mob, rect in zip(mobs, rects):
            placed = False
            ix0, ix1, iy0, iy1 = self._rect_to_cell_range(rect, cell_size)

            for gid, group in enumerate(groups):
                cell_map = group["cell_map"]  # type: ignore[assignment]
                group_rects = group["rects"]  # type: ignore[assignment]

                # Gather candidates from occupied cells
                candidates: set[int] = set()
                for ix in range(ix0, ix1 + 1):
                    for iy in range(iy0, iy1 + 1):
                        candidates.update(cell_map.get((ix, iy), ()))  # type: ignore[arg-type]

                if any(self._rects_overlap(rect, group_rects[i]) for i in candidates):  # type: ignore[index]
                    continue

                # Fits in this group
                idx = len(group_rects)  # type: ignore[arg-type]
                group_rects.append(rect)  # type: ignore[union-attr]
                for ix in range(ix0, ix1 + 1):
                    for iy in range(iy0, iy1 + 1):
                        cell_map.setdefault((ix, iy), []).append(idx)  # type: ignore[union-attr]

                mob.uniforms["_svg_uid"] = float(gid)
                placed = True
                break

            if placed:
                continue

            # New group
            new_group: dict[str, object] = {"rects": [rect], "cell_map": {}}
            new_map = new_group["cell_map"]
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    new_map.setdefault((ix, iy), []).append(0)  # type: ignore[union-attr]
            groups.append(new_group)
            mob.uniforms["_svg_uid"] = float(len(groups) - 1)

    @staticmethod
    def _quantize_rgba(rgba: np.ndarray) -> tuple[int, int, int, int]:
        # rgba expected in [0, 1]
        vals = [int(np.clip(round(255 * float(x)), 0, 255)) for x in rgba]
        # Ensure length 4 even if input isn't
        if len(vals) < 4:
            vals += [255] * (4 - len(vals))
        return (vals[0], vals[1], vals[2], vals[3])

    @classmethod
    def _style_signature(cls, mob: VMobject) -> tuple:
        # Build a lightweight signature for batching that is stable within a run.
        # We purposefully only use the first entry of each style array, since
        # most SVG paths have constant style. This is used ONLY for batching.
        data = mob.data if mob.get_num_points() > 0 else mob._data_defaults
        fill = cls._quantize_rgba(data["fill_rgba"][0])
        stroke = cls._quantize_rgba(data["stroke_rgba"][0])
        stroke_width = int(round(1000 * float(data["stroke_width"][0, 0])))
        fill_border_width = int(round(1000 * float(data["fill_border_width"][0, 0])))
        stroke_behind = int(getattr(mob, "stroke_behind", False))
        flat_stroke = int(getattr(mob, "get_flat_stroke", lambda: False)())
        joint_type = int(round(float(mob.uniforms.get("joint_type", 0.0))))
        anti_alias_width = int(round(1000 * float(mob.uniforms.get("anti_alias_width", 1.5))))
        shading = tuple(int(round(1000 * float(v))) for v in mob.get_shading())
        return (
            fill,
            stroke,
            stroke_width,
            fill_border_width,
            stroke_behind,
            flat_stroke,
            joint_type,
            anti_alias_width,
            shading,
        )

    def _apply_unbatch_mode(self) -> None:
        mode = self.unbatch

        # Iterate only over point-bearing mobjects; those are the ones that
        # produce shader wrappers.
        mobs = self.family_members_with_points()
        for mob in mobs:
            if hasattr(mob, "uniforms"):
                mob.uniforms.pop("_svg_uid", None)

        if mode is False or mode is None:
            # Default: keep batching as large as possible, but split batches
            # when shapes overlap to avoid fill winding artifacts.
            self._apply_overlap_safe_batching(mobs)
            return
        if mode is True:
            for mob in mobs:
                if hasattr(mob, "uniforms"):
                    mob.uniforms["_svg_uid"] = float(id(mob))
            return

        if isinstance(mode, str):
            key = mode.strip().lower()
            if key in {"max", "raw", "none", "off", "unsafe"}:
                # Leave `_svg_uid` cleared for maximum batching.
                return
            if key in {"style", "by_style", "style_groups"}:
                # Use a 24-bit id so it is exactly representable as a float32
                # uniform (shader wrapper ids should still differ reliably).
                for mob in mobs:
                    if not hasattr(mob, "uniforms"):
                        continue
                    sig = self._style_signature(mob)
                    uid = hash_obj(sig) & 0xFFFFFF
                    mob.uniforms["_svg_uid"] = float(uid)
                return

            if key in {"overlap", "auto", "safe"}:
                self._apply_overlap_safe_batching(mobs)
                return

        raise ValueError(
            "Invalid `unbatch` value. Expected False, True, 'style', or 'overlap'."
        )

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

    def mobjects_from_svg_string(self, svg_string: str) -> list[VMobject]:
        element_tree = ET.ElementTree(ET.fromstring(svg_string))
        new_tree = self.modify_xml_tree(element_tree)

        # New svg based on tree contents
        data_stream = io.BytesIO()
        new_tree.write(data_stream)
        data_stream.seek(0)
        svg = se.SVG.parse(data_stream)
        data_stream.close()

        return self.mobjects_from_svg(svg)

    def file_name_to_svg_string(self, file_name: str) -> str:
        return Path(get_full_vector_image_path(file_name)).read_text()

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

    def mobjects_from_svg(self, svg: se.SVG) -> list[VMobject]:
        result = []
        for shape in svg.elements():
            if isinstance(shape, (se.Group, se.Use)):
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
            # elif isinstance(shape, se.Text):
            #     mob = self.text_to_mobject(shape)
            elif isinstance(shape, se.Image):
                mob = self.image_to_mobject(shape)
                if mob is None:
                    log.warning("Only SVG images with embedded base64 data are supported. Skipping image element.")
                    continue
            elif type(shape) == se.SVGElement:
                continue
            else:
                log.warning("Unsupported element type: %s", type(shape))
                continue
            if not mob.has_points() and not isinstance(mob, SVGMobject):
                continue
            if isinstance(shape, se.GraphicObject) and not isinstance(mob, SVGMobject):
                self.apply_style_to_mobject(mob, shape)
            if isinstance(shape, se.Transformable) and shape.apply:
                self.handle_transform(mob, shape.transform)
            result.append(mob)
        return result

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
        mob.set_style(
            stroke_width=0 if not shape.stroke.hexrgb else shape.stroke_width,
            stroke_color=shape.stroke.hexrgb or BLACK,
            stroke_opacity=0 if not shape.stroke.hexrgb else shape.stroke.opacity,
            fill_color=shape.fill.hexrgb or BLACK,
            fill_opacity=0 if not shape.fill.hexrgb else shape.fill.opacity,
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

    def image_to_mobject(self, image: se.Image) -> SVGMobject | None:
        try:
            svg_string = image.data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        # Preserve the parent's batching policy for embedded SVG images.
        mob = SVGMobject(svg_string=svg_string, needs_flip=False, unbatch=self.unbatch)
        if image.height:
            mob.set_height(image.height)
        if image.width:
            mob.set_width(image.width)
        mob.shift((image.x or 0) * RIGHT + (image.width or 0) * RIGHT / 2)
        mob.shift((image.y or 0) * UP + (image.height or 0) * UP / 2)
        return mob

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

    @staticmethod
    def _polygon_signed_area(poly: np.ndarray) -> float:
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    @staticmethod
    def _point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
        # Ray casting point-in-polygon test
        x, y = float(point[0]), float(point[1])
        n = len(poly)
        if n < 3:
            return False
        inside = False
        xj, yj = float(poly[-1, 0]), float(poly[-1, 1])
        for i in range(n):
            xi, yi = float(poly[i, 0]), float(poly[i, 1])
            if (yi > y) != (yj > y):
                denom = (yj - yi)
                if denom != 0:
                    x_intersect = (xj - xi) * (y - yi) / denom + xi
                    if x < x_intersect:
                        inside = not inside
            xj, yj = xi, yi
        return inside

    @classmethod
    def _find_point_inside_polygon(cls, poly: np.ndarray) -> np.ndarray:
        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)
        w = float(xmax - xmin)
        h = float(ymax - ymin)
        cx = float(0.5 * (xmin + xmax))
        cy = float(0.5 * (ymin + ymax))
        candidates = [
            (cx, cy),
            (float(xmin + 0.25 * w), float(ymin + 0.25 * h)),
            (float(xmin + 0.75 * w), float(ymin + 0.25 * h)),
            (float(xmin + 0.25 * w), float(ymin + 0.75 * h)),
            (float(xmin + 0.75 * w), float(ymin + 0.75 * h)),
        ]
        for px, py in candidates:
            p = np.array([px, py], dtype=np.float32)
            if cls._point_in_polygon(p, poly):
                return p
        # Fallback: midpoint of first edge (may be on boundary)
        return 0.5 * (poly[0] + poly[1])

    def _normalize_subpath_orientations(self) -> None:
        # ManimGL's fill treats oppositely-oriented subpaths as subtractive.
        # SVG's non-zero fill rule fills disjoint subpaths regardless of
        # orientation.
        #
        # Be conservative: only normalize winding when the path consists of
        # disjoint contours. If any contour appears nested (typical for true
        # holes like inside glyphs), leave the original winding untouched.
        subpaths = self.get_subpaths()
        if len(subpaths) <= 1:
            return

        # Compute a cheap bbox per contour using all points (anchors + handles)
        # so curved outlines don't get under-approximated.
        areas: list[float] = []
        bboxes: list[tuple[float, float, float, float] | None] = []
        for sp in subpaths:
            if len(sp) == 0:
                areas.append(0.0)
                bboxes.append(None)
                continue

            pts2d = sp[:, :2].astype(np.float32)
            xmin, ymin = pts2d.min(axis=0)
            xmax, ymax = pts2d.max(axis=0)
            bboxes.append((float(xmin), float(ymin), float(xmax), float(ymax)))

            anchors = sp[::2]
            poly = anchors[:, :2].astype(np.float32)
            if len(poly) >= 2 and np.allclose(poly[0], poly[-1], atol=1e-6):
                poly = poly[:-1]
            if len(poly) < 3:
                areas.append(0.0)
            else:
                areas.append(self._polygon_signed_area(poly))

        # If any contour's bbox is fully contained in another, assume the path
        # intentionally encodes holes and don't touch winding.
        tol = 1e-6
        for i, bbi in enumerate(bboxes):
            if bbi is None:
                continue
            xi0, yi0, xi1, yi1 = bbi
            for j, bbj in enumerate(bboxes):
                if i == j or bbj is None:
                    continue
                xj0, yj0, xj1, yj1 = bbj
                if (
                    (xi0 >= xj0 - tol) and (yi0 >= yj0 - tol) and
                    (xi1 <= xj1 + tol) and (yi1 <= yj1 + tol) and
                    (
                        (xi0 > xj0 + tol) or (yi0 > yj0 + tol) or
                        (xi1 < xj1 - tol) or (yi1 < yj1 - tol)
                    )
                ):
                    return

        # No nesting detected: unify all non-degenerate contours to the winding
        # of the largest contour.
        if not areas:
            return
        target_area = max(areas, key=lambda a: abs(a))
        if abs(target_area) < 1e-8:
            return
        desired_positive = (target_area > 0)

        new_subpaths: list[np.ndarray] = []
        needs_rebuild = False
        for sp, area in zip(subpaths, areas):
            if abs(area) < 1e-8:
                new_subpaths.append(sp)
                continue
            if (area > 0) != desired_positive:
                new_subpaths.append(sp[::-1].copy())
                needs_rebuild = True
            else:
                new_subpaths.append(sp)

        if not needs_rebuild:
            return

        self.clear_points()
        for sp in new_subpaths:
            self.add_subpath(sp)

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
        
        # Get rid of the side effect of trailing 'Z M' commands.
        if self.has_new_path_started():
            self.resize_points(self.get_num_points() - 2)

        self._normalize_subpath_orientations()

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
