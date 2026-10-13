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
        [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5]],
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

def test_linear_to_radial_keeps_a_stable_radial_basis() -> None:
    start = Rectangle(width=3, height=1)
    start.set_fill_gradient(
        ["#ff0000", "#0000ff"],
        direction=[1, 0, 0],
    )
    target = start.copy()
    target.set_fill_gradient(
        ["#ff0000", "#0000ff"],
        gradient_type="radial",
        center=[0, 0, 0],
        radius=1.5,
        radius_y=0.6,
    )
    animated = start.copy()

    interpolate_at(animated, start, target, 0.01)
    uniforms = animated.uniforms
    center = uniforms["gradient_radial_center"][:2]
    axis_x = uniforms["gradient_radial_axis_x"][:2] - center
    axis_y = uniforms["gradient_radial_axis_y"][:2] - center
    determinant = axis_x[0] * axis_y[1] - axis_x[1] * axis_y[0]
    assert abs(determinant) > 1e-4


def test_gradient_color_interpolation_and_spread_modes_morph_continuously() -> None:
    start = Rectangle(width=3, height=1)
    start.set_fill_gradient(
        ["#ff0000", "#0000ff"],
        direction=[1, 0, 0],
        spread_method="pad",
        color_interpolation="sRGB",
    )
    target = start.copy()
    target.set_fill_gradient(
        ["#ff0000", "#0000ff"],
        direction=[1, 0, 0],
        spread_method="reflect",
        color_interpolation="linearRGB",
    )
    animated = start.copy()

    interpolate_at(animated, start, target, 0.5)

    assert abs(animated.uniforms["gradient_spread"] - 1.0) < 1e-6
    assert abs(animated.uniforms["gradient_interpolation"] - 0.5) < 1e-6


def test_new_vfx_radius_and_strength_both_fade_in() -> None:
    from manimlib.utils.vfx_presets import Glow

    start = Rectangle()
    target = start.copy()
    target.add_vfx(Glow(radius=32, intensity=1.6))
    animated = start.copy()

    interpolate_at(animated, start, target, 0.5)

    glow = next(effect for effect in animated.get_vfx_list() if effect.name == "Glow")
    assert abs(glow.radius - 16.0) < 1e-6
    assert abs(glow.intensity - 0.8) < 1e-6

def test_matching_glow_interpolates_default_tint_to_explicit_tint() -> None:
    from manimlib.utils.vfx_presets import Glow

    start = Rectangle()
    start.add_vfx(Glow(color=None, intensity=1.0))
    target = Rectangle()
    target.add_vfx(Glow(color="#0000ff", intensity=1.0))
    animated = start.copy()

    interpolate_at(animated, start, target, 0.5)

    glow = next(effect for effect in animated.get_vfx_list() if effect.name == "Glow")
    np.testing.assert_allclose(glow.color, [0.5, 0.5, 1.0, 1.0], atol=1e-5)

def test_shallow_copy_has_independent_vfx_instances_and_bookkeeping() -> None:
    source = Rectangle()
    source.add_vfx(Grain(intensity=0.4, speed=12))
    source_effect = source.get_vfx_list()[0]

    clone = source.copy()
    clone_effect = clone.get_vfx_list()[0]

    assert clone.get_vfx_list() is not source.get_vfx_list()
    assert clone_effect is not source_effect
    assert clone_effect.mobject is clone
    assert source_effect.mobject is source
    assert clone._vfx_group is clone
    assert clone._vfx_local is clone

    clone.clear_vfx()
    assert len(source.get_vfx_list()) == 1
    assert not clone.get_vfx_list()


def test_deep_copy_has_independent_vfx_instances() -> None:
    source = Rectangle()
    source.add_vfx(Grain(intensity=0.35, speed=7))
    source_effect = source.get_vfx_list()[0]

    clone = source.copy(deep=True)
    clone_effect = clone.get_vfx_list()[0]

    assert clone_effect is not source_effect
    assert clone_effect.mobject is clone
    assert source_effect.mobject is source
    clone_effect.intensity = 0.9
    assert abs(source_effect.intensity - 0.35) < 1e-6


def test_transform_endpoints_keep_independent_vfx_after_copy() -> None:
    source = Rectangle()
    source.add_vfx(Grain(intensity=0.2))
    target = source.copy()
    target.clear_vfx()
    target.add_vfx(Grain(intensity=0.8))

    assert len(source.get_vfx_list()) == 1
    assert len(target.get_vfx_list()) == 1
    assert source.get_vfx_list()[0] is not target.get_vfx_list()[0]

    animated = source.copy()
    interpolate_at(animated, source, target, 0.5)
    effect = next(effect for effect in animated.get_vfx_list() if effect.name == "Grain")
    assert abs(effect.intensity - 0.5) < 1e-6
