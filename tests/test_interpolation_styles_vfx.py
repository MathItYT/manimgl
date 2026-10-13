from __future__ import annotations

import numpy as np

from manimlib.mobject.geometry import Rectangle
from manimlib.utils.vfx_presets import Grain


def interpolate_at(mobject, start, target, alpha: float):
    mobject.prepare_interpolation(start, target)
    mobject.interpolate(start, target, alpha)
    return mobject


def test_solid_fill_morphs_into_a_gradient() -> None:
    start = Rectangle(width=3, height=1)
    start.set_fill("#ff0000", opacity=1)
    target = start.copy()
    target.set_fill_gradient(
        ["#ff0000", "#0000ff"],
        direction=[1, 0, 0],
    )
    animated = start.copy()

    interpolate_at(animated, start, target, 0.5)

    assert animated.uniforms["gradient_mode"] == 1.0
    assert animated.uniforms["gradient_count"] == 2.0
    np.testing.assert_allclose(
        animated.uniforms["gradient_colors"][:2, :3],
        [[1.0, 0.5, 0.0], [0.5, 0.0, 0.5]],
        atol=1e-5,
    )


def test_gradient_morphs_into_a_solid_fill() -> None:
    start = Rectangle(width=3, height=1)
    start.set_fill_gradient(
        ["#ff0000", "#0000ff"],
        direction=[1, 0, 0],
    )
    target = start.copy()
    target.set_fill("#00ff00", opacity=1)
    animated = start.copy()

    interpolate_at(animated, start, target, 0.5)

    assert animated.uniforms["gradient_mode"] == 1.0
    np.testing.assert_allclose(
        animated.uniforms["gradient_colors"][:2, :3],
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]],
        atol=1e-5,
    )


def test_gradient_morph_aligns_different_stops_and_gradient_types() -> None:
    start = Rectangle(width=3, height=1)
    start.set_fill_gradient(
        ["#ff0000", "#0000ff"],
        offsets=[0.0, 1.0],
        direction=[1, 0, 0],
    )
    target = start.copy()
    target.set_fill_gradient(
        ["#ffffff", "#55aaff", "#000000"],
        offsets=[0.0, 0.3, 1.0],
        gradient_type="radial",
        center=[0, 0, 0],
        radius=1.5,
        radius_y=0.5,
    )
    animated = start.copy()

    interpolate_at(animated, start, target, 0.5)

    assert animated.uniforms["gradient_count"] == 3.0
    assert 1.0 < animated.uniforms["gradient_mode"] < 2.0
    np.testing.assert_allclose(
        animated.uniforms["gradient_stops"][:3, 0],
        [0.0, 0.3, 1.0],
        atol=1e-5,
    )


def test_vfx_is_faded_in_when_only_the_target_has_it() -> None:
    start = Rectangle()
    target = start.copy()
    target.add_vfx(Grain(intensity=0.4))
    animated = start.copy()

    interpolate_at(animated, start, target, 0.5)

    effects = [effect for effect in animated.get_vfx_list() if effect.name == "Grain"]
    assert len(effects) == 1
    assert abs(effects[0].intensity - 0.2) < 1e-6

    interpolate_at(animated, start, target, 1.0)
    assert abs(animated.get_vfx_list()[0].intensity - 0.4) < 1e-6


def test_vfx_is_faded_out_when_only_the_start_has_it() -> None:
    start = Rectangle()
    start.add_vfx(Grain(intensity=0.6))
    target = Rectangle()
    animated = start.copy()

    interpolate_at(animated, start, target, 0.5)

    effects = [effect for effect in animated.get_vfx_list() if effect.name == "Grain"]
    assert len(effects) == 1
    assert abs(effects[0].intensity - 0.3) < 1e-6

    interpolate_at(animated, start, target, 1.0)
    assert not animated.get_vfx_list()


def test_matching_vfx_parameters_are_interpolated() -> None:
    start = Rectangle()
    start.add_vfx(Grain(intensity=0.2, speed=10))
    target = Rectangle()
    target.add_vfx(Grain(intensity=0.8, speed=30))
    animated = start.copy()

    interpolate_at(animated, start, target, 0.5)

    effect = next(effect for effect in animated.get_vfx_list() if effect.name == "Grain")
    assert abs(effect.intensity - 0.5) < 1e-6
    assert abs(effect.speed - 20.0) < 1e-6
