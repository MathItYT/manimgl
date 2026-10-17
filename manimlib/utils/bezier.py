from __future__ import annotations

import numpy as np
from scipy import linalg
from fontTools.cu2qu.cu2qu import curve_to_quadratic

from manimlib.logger import log
from manimlib.utils.simple_functions import choose
from manimlib.utils.space_ops import cross2d
from manimlib.utils.space_ops import cross
from manimlib.utils.space_ops import find_intersection
from manimlib.utils.space_ops import midpoint
from manimlib.utils.space_ops import get_norm
from manimlib.utils.space_ops import z_to_vector

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Sequence, TypeVar, Tuple
    from manimlib.typing import VectN, FloatArray, VectNArray, Vect3Array

    Scalable = TypeVar("Scalable", float, FloatArray)


CLOSED_THRESHOLD = 0.001


def bezier(
    points: Sequence[float | FloatArray] | VectNArray
) -> Callable[[float], float | FloatArray]:
    n = len(points)
    if n == 0:
        raise Exception("bezier cannot be calld on an empty list")
    if n == 3:
        p0, p1, p2 = points[0], points[1], points[2]
        def quad_bezier(t: float) -> float | FloatArray:
            omt = 1.0 - t
            return (omt * omt) * p0 + (2.0 * omt * t) * p1 + (t * t) * p2
        return quad_bezier
    elif n == 4:
        p0, p1, p2, p3 = points[0], points[1], points[2], points[3]
        def cubic_bezier(t: float) -> float | FloatArray:
            omt = 1.0 - t
            omt2 = omt * omt
            t2 = t * t
            return (omt2 * omt) * p0 + (3.0 * omt2 * t) * p1 + (3.0 * omt * t2) * p2 + (t2 * t) * p3
        return cubic_bezier
    elif n == 2:
        p0, p1 = points[0], points[1]
        def linear_bezier(t: float) -> float | FloatArray:
            return (1.0 - t) * p0 + t * p1
        return linear_bezier
    elif n == 1:
        p0 = points[0]
        return lambda t: p0

    deg = n - 1

    def result(t: float) -> float | FloatArray:
        return sum(
            ((1 - t)**(deg - k)) * (t**k) * choose(deg, k) * point
            for k, point in enumerate(points)
        )

    return result


def partial_bezier_points(
    points: Sequence[Scalable],
    a: float,
    b: float
) -> list[Scalable]:
    """
    Given an list of points which define
    a bezier curve, and two numbers 0<=a<b<=1,
    return an list of the same size, which
    describes the portion of the original bezier
    curve on the interval [a, b].

    This algorithm is pretty nifty, and pretty dense.
    """
    n = len(points)
    if a == 1:
        return [points[-1]] * n
    if n == 3:
        return partial_quadratic_bezier_points(points, a, b)
    if n == 4:
        p0, p1, p2, p3 = points[0], points[1], points[2], points[3]
        oma = 1.0 - a
        q0 = oma * p0 + a * p1
        q1 = oma * p1 + a * p2
        q2 = oma * p2 + a * p3
        r0 = oma * q0 + a * q1
        r1 = oma * q1 + a * q2
        s0 = oma * r0 + a * r1

        end_prop = (b - a) / (1.0 - a)
        ome = 1.0 - end_prop
        c0 = s0
        c1 = ome * s0 + end_prop * r1
        c2_0 = ome * r1 + end_prop * q2
        c2 = ome * c1 + end_prop * c2_0
        c3_0 = ome * q2 + end_prop * p3
        c3_1 = ome * c2_0 + end_prop * c3_0
        c3 = ome * c2 + end_prop * c3_1
        return [c0, c1, c2, c3]

    a_to_1 = [
        bezier(points[i:])(a)
        for i in range(len(points))
    ]
    end_prop = (b - a) / (1. - a)
    return [
        bezier(a_to_1[:i + 1])(end_prop)
        for i in range(len(points))
    ]


# Shortened version of partial_bezier_points just for quadratics,
# since this is called a fair amount
def partial_quadratic_bezier_points(
    points: Sequence[VectN] | VectNArray,
    a: float,
    b: float
) -> list[VectN]:
    p0, p1, p2 = points[0], points[1], points[2]
    if a == 1:
        return [p2, p2, p2]

    if a > 0:
        oma = 1.0 - a
        h0 = (oma * oma) * p0 + (2.0 * oma * a) * p1 + (a * a) * p2
    else:
        h0 = p0

    if b < 1:
        omb = 1.0 - b
        h2 = (omb * omb) * p0 + (2.0 * omb * b) * p1 + (b * b) * p2
    else:
        h2 = p2

    h1_prime = (1.0 - a) * p1 + a * p2
    end_prop = (b - a) / (1.0 - a)
    h1 = (1.0 - end_prop) * h0 + end_prop * h1_prime
    return [h0, h1, h2]


# Linear interpolation variants


def interpolate(start: Scalable, end: Scalable, alpha: float | VectN) -> Scalable:
    try:
        return (1 - alpha) * start + alpha * end
    except TypeError:
        log.debug(f"`start` parameter with type `{type(start)}` and dtype `{start.dtype}`")
        log.debug(f"`end` parameter with type `{type(end)}` and dtype `{end.dtype}`")
        log.debug(f"`alpha` parameter with value `{alpha}`")
        import sys
        sys.exit(2)


def outer_interpolate(
    start: Scalable,
    end: Scalable,
    alpha: Scalable,
) -> np.ndarray:
    result = np.outer(1 - alpha, start) + np.outer(alpha, end)
    return result.reshape((*np.shape(alpha), *np.shape(start)))


def set_array_by_interpolation(
    arr: np.ndarray,
    arr1: np.ndarray,
    arr2: np.ndarray,
    alpha: float,
    interp_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = interpolate
) -> np.ndarray:
    arr[:] = interp_func(arr1, arr2, alpha)
    return arr


def integer_interpolate(
    start: int,
    end: int,
    alpha: float
) -> tuple[int, float]:
    """
    alpha is a float between 0 and 1.  This returns
    an integer between start and end (inclusive) representing
    appropriate interpolation between them, along with a
    "residue" representing a new proportion between the
    returned integer and the next one of the
    list.

    For example, if start=0, end=10, alpha=0.46, This
    would return (4, 0.6).
    """
    if alpha >= 1:
        return (end - 1, 1.0)
    if alpha <= 0:
        return (start, 0)
    value = int(interpolate(start, end, alpha))
    residue = ((end - start) * alpha) % 1
    return (value, residue)


def mid(start: Scalable, end: Scalable) -> Scalable:
    return (start + end) / 2.0


def inverse_interpolate(start: Scalable, end: Scalable, value: Scalable) -> np.ndarray:
    return np.true_divide(value - start, end - start)


def match_interpolate(
    new_start: Scalable,
    new_end: Scalable,
    old_start: Scalable,
    old_end: Scalable,
    old_value: Scalable
) -> Scalable:
    return interpolate(
        new_start, new_end,
        inverse_interpolate(old_start, old_end, old_value)
    )


def quadratic_bezier_points_for_arc(angle: float, n_components: int = 8):
    n_points = 2 * n_components + 1
    angles = np.linspace(0, angle, n_points)
    points = np.array([np.cos(angles), np.sin(angles), np.zeros(n_points)]).T
    # Adjust handles
    theta = angle / n_components
    points[1::2] /= np.cos(theta / 2)
    return points


def approx_smooth_quadratic_bezier_handles(
    points: FloatArray
) -> FloatArray:
    """
    Figuring out which bezier curves most smoothly connect a sequence of points.

    Given three successive points, P0, P1 and P2, you can compute that by defining
    h = (1/4) P0 + P1 - (1/4)P2, the bezier curve defined by (P0, h, P1) will pass
    through the point P2.

    So for a given set of four successive points, P0, P1, P2, P3, if we want to add
    a handle point h between P1 and P2 so that the quadratic bezier (P1, h, P2) is
    part of a smooth curve passing through all four points, we calculate one solution
    for h that would produce a parbola passing through P3, call it smooth_to_right, and
    another that would produce a parabola passing through P0, call it smooth_to_left,
    and use the midpoint between the two.
    """
    if len(points) == 1:
        return points[0]
    elif len(points) == 2:
        return midpoint(*points)
    smooth_to_right, smooth_to_left = [
        0.25 * ps[0:-2] + ps[1:-1] - 0.25 * ps[2:]
        for ps in (points, points[::-1])
    ]
    if np.isclose(points[0], points[-1]).all():
        last_str = 0.25 * points[-2] + points[-1] - 0.25 * points[1]
        last_stl = 0.25 * points[1] + points[0] - 0.25 * points[-2]
    else:
        last_str = smooth_to_left[0]
        last_stl = smooth_to_right[0]
    handles = 0.5 * np.vstack([smooth_to_right, [last_str]])
    handles += 0.5 * np.vstack([last_stl, smooth_to_left[::-1]])
    return handles


def smooth_quadratic_path(anchors: Vect3Array) -> Vect3Array:
    """
    Returns a path defining a smooth quadratic bezier spline
    through anchors.
    """
    if len(anchors) < 2:
        return anchors
    elif len(anchors) == 2:
        return np.array([anchors[0], anchors.mean(0), anchors[1]])

    is_flat = (anchors[:, 2] == 0).all()
    if not is_flat:
        normal = cross(anchors[2] - anchors[1], anchors[1] - anchors[0])
        rot = z_to_vector(normal)
        anchors = np.dot(anchors, rot)
        shift = anchors[0, 2]
        anchors[:, 2] -= shift
    h1s, h2s = get_smooth_cubic_bezier_handle_points(anchors)
    quads = [anchors[0, :2]]
    for cub_bs in zip(anchors[:-1], h1s, h2s, anchors[1:]):
        # Try to use fontTools curve_to_quadratic
        new_quads = curve_to_quadratic(
            [b[:2] for b in cub_bs],
            max_err=0.1 * get_norm(cub_bs[3] - cub_bs[0])
        )
        # Otherwise fall back on home baked solution
        if new_quads is None or len(new_quads) % 2 == 0:
            new_quads = get_quadratic_approximation_of_cubic(*cub_bs)[:, :2]
        quads.extend(new_quads[1:])
    new_path = np.zeros((len(quads), 3))
    new_path[:, :2] = quads
    if not is_flat:
        new_path[:, 2] += shift
        new_path = np.dot(new_path, rot.T)
    return new_path


def get_smooth_cubic_bezier_handle_points(
    points: Sequence[VectN] | VectNArray
) -> tuple[FloatArray, FloatArray]:
    points = np.array(points)
    num_handles = len(points) - 1
    dim = points.shape[1]
    if num_handles < 1:
        return np.zeros((0, dim)), np.zeros((0, dim))
    # Must solve 2*num_handles equations to get the handles.
    # l and u are the number of lower an upper diagonal rows
    # in the matrix to solve.
    l, u = 2, 1
    # diag is a representation of the matrix in diagonal form
    # See https://www.particleincell.com/2012/bezier-splines/
    # for how to arrive at these equations
    diag = np.zeros((l + u + 1, 2 * num_handles))
    diag[0, 1::2] = -1
    diag[0, 2::2] = 1
    diag[1, 0::2] = 2
    diag[1, 1::2] = 1
    diag[2, 1:-2:2] = -2
    diag[3, 0:-3:2] = 1
    # last
    diag[2, -2] = -1
    diag[1, -1] = 2
    # This is the b as in Ax = b, where we are solving for x,
    # and A is represented using diag.  However, think of entries
    # to x and b as being points in space, not numbers
    b = np.zeros((2 * num_handles, dim))
    b[1::2] = 2 * points[1:]
    b[0] = points[0]
    b[-1] = points[-1]

    use_closed_solve_function = is_closed(points)
    if use_closed_solve_function:
        # Get equations to relate first and last points
        matrix = diag_to_matrix((l, u), diag)
        # last row handles second derivative
        matrix[-1, [0, 1, -2, -1]] = [2, -1, 1, -2]
        # first row handles first derivative
        matrix[0, :] = 0
        matrix[0, [0, -1]] = [1, 1]
        b[0] = 2 * points[0]
        b[-1] = 0
        handle_pairs = linalg.solve(matrix, b)
    else:
        handle_pairs = linalg.solve_banded((l, u), diag, b)

    return handle_pairs[0::2], handle_pairs[1::2]


def diag_to_matrix(
    l_and_u: tuple[int, int], 
    diag: np.ndarray
) -> np.ndarray:
    """
    Converts array whose rows represent diagonal
    entries of a matrix into the matrix itself.
    See scipy.linalg.solve_banded
    """
    l, u = l_and_u
    dim = diag.shape[1]
    matrix = np.zeros((dim, dim))
    for i in range(l + u + 1):
        np.fill_diagonal(
            matrix[max(0, i - u):, max(0, u - i):],
            diag[i, max(0, u - i):]
        )
    return matrix


def is_closed(points: FloatArray) -> bool:
    return np.allclose(points[0], points[-1])


# Given 4 control points for a cubic bezier curve (or arrays of such)
# return control points for 2 quadratics (or 2n quadratics) approximating them.
def get_quadratic_approximation_of_cubic(
    a0: FloatArray,
    h0: FloatArray,
    h1: FloatArray,
    a1: FloatArray
) -> FloatArray:
    a0 = np.array(a0, ndmin=2)
    h0 = np.array(h0, ndmin=2)
    h1 = np.array(h1, ndmin=2)
    a1 = np.array(a1, ndmin=2)
    # Tangent vectors at the start and end.
    T0 = h0 - a0
    T1 = a1 - h1

    # Search for inflection points.  If none are found, use the
    # midpoint as a cut point.
    # Based on http://www.caffeineowl.com/graphics/2d/vectorial/cubic-inflexion.html
    has_infl = np.ones(len(a0), dtype=bool)

    p = h0 - a0
    q = h1 - 2 * h0 + a0
    r = a1 - 3 * h1 + 3 * h0 - a0

    a = cross2d(q, r)
    b = cross2d(p, r)
    c = cross2d(p, q)

    disc = b * b - 4 * a * c
    has_infl &= (disc > 0)
    sqrt_disc = np.sqrt(np.abs(disc))
    settings = np.seterr(all='ignore')
    ti_bounds = []
    for sgn in [-1, +1]:
        ti = (-b + sgn * sqrt_disc) / (2 * a)
        ti[a == 0] = (-c / b)[a == 0]
        ti[(a == 0) & (b == 0)] = 0
        ti_bounds.append(ti)
    ti_min, ti_max = ti_bounds
    np.seterr(**settings)
    ti_min_in_range = has_infl & (0 < ti_min) & (ti_min < 1)
    ti_max_in_range = has_infl & (0 < ti_max) & (ti_max < 1)

    # Choose a value of t which starts at 0.5,
    # but is updated to one of the inflection points
    # if they lie between 0 and 1

    t_mid = 0.5 * np.ones(len(a0))
    t_mid[ti_min_in_range] = ti_min[ti_min_in_range]
    t_mid[ti_max_in_range] = ti_max[ti_max_in_range]

    m, n = a0.shape
    t_mid = t_mid.repeat(n).reshape((m, n))

    # Compute bezier point and tangent at the chosen value of t
    mid = bezier([a0, h0, h1, a1])(t_mid)
    Tm = bezier([h0 - a0, h1 - h0, a1 - h1])(t_mid)

    # Intersection between tangent lines at end points
    # and tangent in the middle
    i0 = find_intersection(a0, T0, mid, Tm)
    i1 = find_intersection(a1, T1, mid, Tm)

    m, n = np.shape(a0)
    result = np.zeros((5 * m, n))
    result[0::5] = a0
    result[1::5] = i0
    result[2::5] = mid
    result[3::5] = i1
    result[4::5] = a1
    return result


def get_smooth_quadratic_bezier_path_through(
    points: Sequence[VectN]
) -> np.ndarray:
    # TODO
    h0, h1 = get_smooth_cubic_bezier_handle_points(points)
    a0 = points[:-1]
    a1 = points[1:]
    return get_quadratic_approximation_of_cubic(a0, h0, h1, a1)
