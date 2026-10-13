from __future__ import annotations

import numpy as np

from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.svg.svg_mobject import SVGMobject


def test_vmobject_accepts_multi_stop_linear_gradient() -> None:
    mob = Rectangle(width=4, height=2)
    mob.set_fill_gradient(
        ["#ff0000", "#00ff00", "#0000ff"],
        offsets=[0.0, 0.25, 1.0],
        direction=[1.0, 0.0, 0.0],
    )

    assert mob.uniforms["gradient_mode"] == 1.0
    assert mob.uniforms["gradient_count"] == 3.0
    np.testing.assert_allclose(
        mob.uniforms["gradient_stops"][:3, 0],
        [0.0, 0.25, 1.0],
    )
    np.testing.assert_allclose(
        mob.uniforms["gradient_colors"][:3, 3],
        [1.0, 1.0, 1.0],
    )


def test_vmobject_accepts_radial_gradient_and_focus() -> None:
    mob = Rectangle(width=4, height=2)
    mob.set_fill_gradient(
        ["#ffffff", "#000000"],
        gradient_type="radial",
        center=[0.0, 0.0, 0.0],
        focus=[0.25, 0.0, 0.0],
        radius=2.0,
        radius_y=1.0,
    )

    assert mob.uniforms["gradient_mode"] == 2.0
    assert mob.uniforms["gradient_interpolation"] == 0.0  # SVG's initial value is sRGB.
    np.testing.assert_allclose(
        mob.uniforms["gradient_focus"],
        [0.25, 0.0, 0.0],
    )


def test_svg_linear_gradient_is_applied_through_vmobject_api() -> None:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="100" height="50" viewBox="0 0 100 50">
      <defs>
        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="red"/>
          <stop offset="40%" stop-color="rgb(0, 255, 0)" stop-opacity="0.5"/>
          <stop offset="100%" stop-color="#0000ff"/>
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="100" height="50" fill="url(#gradient)"/>
    </svg>
    """
    mob = SVGMobject(svg_string=svg, should_center=False, height=1.0)

    assert len(mob.submobjects) == 1
    shape = mob.submobjects[0]
    assert shape.uniforms["gradient_mode"] == 1.0
    assert shape.uniforms["gradient_count"] == 3.0
    np.testing.assert_allclose(
        shape.uniforms["gradient_colors"][:3, 3],
        [1.0, 0.5, 1.0],
    )


def test_svg_radial_gradient_uses_its_focal_point() -> None:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
      <defs>
        <radialGradient id="gradient" cx="50%" cy="50%" r="50%" fx="35%" fy="40%">
          <stop offset="0%" stop-color="#ffffff"/>
          <stop offset="100%" stop-color="#101820"/>
        </radialGradient>
      </defs>
      <circle cx="40" cy="40" r="35" fill="url(#gradient)"/>
    </svg>
    """
    mob = SVGMobject(svg_string=svg, should_center=False, height=1.0)

    assert len(mob.submobjects) == 1
    shape = mob.submobjects[0]
    assert shape.uniforms["gradient_mode"] == 2.0
    assert shape.uniforms["gradient_count"] == 2.0
    assert not np.allclose(
        shape.uniforms["gradient_start"],
        shape.uniforms["gradient_focus"],
    )



def test_vmobject_supports_linear_rgb_interpolation() -> None:
    mob = Rectangle(width=2, height=1)
    mob.set_fill_gradient(
        ["#000000", "#ffffff"],
        gradient_type="radial",
        color_interpolation="linearRGB",
    )

    assert mob.uniforms["gradient_interpolation"] == 1.0


def test_svg_gradient_honors_color_interpolation() -> None:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"
         viewBox="0 0 80 80" color-interpolation="linearRGB">
      <defs>
        <radialGradient id="gradient" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#ffffff"/>
          <stop offset="100%" stop-color="#101820"/>
        </radialGradient>
      </defs>
      <circle cx="40" cy="40" r="35" fill="url(#gradient)"/>
    </svg>
    """
    mob = SVGMobject(svg_string=svg, should_center=False, height=1.0)

    assert len(mob.submobjects) == 1
    shape = mob.submobjects[0]
    assert shape.uniforms["gradient_mode"] == 2.0
    assert shape.uniforms["gradient_interpolation"] == 1.0



def test_svg_gradient_accepts_its_own_color_interpolation_attribute() -> None:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"
         viewBox="0 0 80 80">
      <defs>
        <linearGradient id="gradient" color-interpolation="linearRGB">
          <stop offset="0%" stop-color="#000000"/>
          <stop offset="100%" stop-color="#ffffff"/>
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="80" height="80" fill="url(#gradient)"/>
    </svg>
    """
    mob = SVGMobject(svg_string=svg, should_center=False, height=1.0)

    assert len(mob.submobjects) == 1
    assert mob.submobjects[0].uniforms["gradient_interpolation"] == 1.0
