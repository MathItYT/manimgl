from __future__ import annotations

import sys
import numpy as np

from manimlib.mobject.types.vectorized_mobject import VMobject

if sys.platform != "emscripten":
    import pathops


def _convert_vmobject_to_svg_path(vmobject: VMobject) -> str:
    commands = []
    for submob in vmobject.family_members_with_points():
        for subpath in submob.get_subpaths():
            if len(subpath) == 0:
                continue
            quads = vmobject.get_bezier_tuples_from_points(subpath)
            start = subpath[0]
            commands.append(f"M {start[0]} {start[1]}")
            for p0, p1, p2 in quads:
                commands.append(f"Q {p1[0]} {p1[1]} {p2[0]} {p2[1]}")
            if vmobject.consider_points_equal(subpath[0], subpath[-1]):
                commands.append("Z")
    return " ".join(commands)


def _native_convert_vmobject_to_skia_path(vmobject: VMobject) -> pathops.Path:
    path = pathops.Path()
    for submob in vmobject.family_members_with_points():
        for subpath in submob.get_subpaths():
            quads = vmobject.get_bezier_tuples_from_points(subpath)
            start = subpath[0]
            path.moveTo(*start[:2])
            for p0, p1, p2 in quads:
                path.quadTo(*p1[:2], *p2[:2])
            if vmobject.consider_points_equal(subpath[0], subpath[-1]):
                path.close()
    return path


def _native_convert_skia_path_to_vmobject(path: pathops.Path, vmobject: VMobject) -> VMobject:
    PathVerb = pathops.PathVerb
    current_path_start = np.array([0.0, 0.0, 0.0])
    for path_verb, points in path:
        if path_verb == PathVerb.CLOSE:
            vmobject.add_line_to(current_path_start)
        else:
            points = np.hstack((np.array(points), np.zeros((len(points), 1))))
            if path_verb == PathVerb.MOVE:
                for point in points:
                    current_path_start = point
                    vmobject.start_new_path(point)
            elif path_verb == PathVerb.CUBIC:
                vmobject.add_cubic_bezier_curve_to(*points)
            elif path_verb == PathVerb.LINE:
                vmobject.add_line_to(points[0])
            elif path_verb == PathVerb.QUAD:
                vmobject.add_quadratic_bezier_curve_to(*points)
            else:
                raise Exception(f"Unsupported: {path_verb}")
    return vmobject.reverse_points()


def _browser_boolean(target: VMobject, vmobjects: list[VMobject], operation: str) -> VMobject:
    from manimlib.mobject.svg.svg_mobject import SVGMobject
    from manimlib.utils.browser_pathops import browser_pathops

    svg_paths = [_convert_vmobject_to_svg_path(mob) for mob in vmobjects]
    d = browser_pathops.combine(svg_paths, operation)
    result = SVGMobject(svg_string=f'<svg xmlns="http://www.w3.org/2000/svg"><path d="{d}"/></svg>')
    target.become(result)
    return target


class Union(VMobject):
    def __init__(self, *vmobjects: VMobject, **kwargs):
        if len(vmobjects) < 2:
            raise ValueError("At least 2 mobjects needed for Union.")
        super().__init__(**kwargs)
        if sys.platform == "emscripten":
            _browser_boolean(self, list(vmobjects), "UNION")
            return
        outpen = pathops.Path()
        paths = [_native_convert_vmobject_to_skia_path(vmobject) for vmobject in vmobjects]
        pathops.union(paths, outpen.getPen())
        _native_convert_skia_path_to_vmobject(outpen, self)


class Difference(VMobject):
    def __init__(self, subject: VMobject, clip: VMobject, **kwargs):
        super().__init__(**kwargs)
        if sys.platform == "emscripten":
            _browser_boolean(self, [subject, clip], "DIFFERENCE")
            return
        outpen = pathops.Path()
        pathops.difference([_native_convert_vmobject_to_skia_path(subject)], [_native_convert_vmobject_to_skia_path(clip)], outpen.getPen())
        _native_convert_skia_path_to_vmobject(outpen, self)


class Intersection(VMobject):
    def __init__(self, *vmobjects: VMobject, **kwargs):
        if len(vmobjects) < 2:
            raise ValueError("At least 2 mobjects needed for Intersection.")
        super().__init__(**kwargs)
        if sys.platform == "emscripten":
            _browser_boolean(self, list(vmobjects), "INTERSECT")
            return
        outpen = pathops.Path()
        pathops.intersection([_native_convert_vmobject_to_skia_path(vmobjects[0])], [_native_convert_vmobject_to_skia_path(vmobjects[1])], outpen.getPen())
        for _i in range(2, len(vmobjects)):
            new_outpen = pathops.Path()
            pathops.intersection([outpen], [_native_convert_vmobject_to_skia_path(vmobjects[_i])], new_outpen.getPen())
            outpen = new_outpen
        _native_convert_skia_path_to_vmobject(outpen, self)


class Exclusion(VMobject):
    def __init__(self, *vmobjects: VMobject, **kwargs):
        if len(vmobjects) < 2:
            raise ValueError("At least 2 mobjects needed for Exclusion.")
        super().__init__(**kwargs)
        if sys.platform == "emscripten":
            _browser_boolean(self, list(vmobjects), "XOR")
            return
        outpen = pathops.Path()
        pathops.xor([_native_convert_vmobject_to_skia_path(vmobjects[0])], [_native_convert_vmobject_to_skia_path(vmobjects[1])], outpen.getPen())
        for _i in range(2, len(vmobjects)):
            new_outpen = pathops.Path()
            pathops.xor([outpen], [_native_convert_vmobject_to_skia_path(vmobjects[_i])], new_outpen.getPen())
            outpen = new_outpen
        _native_convert_skia_path_to_vmobject(outpen, self)
