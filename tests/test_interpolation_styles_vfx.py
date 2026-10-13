from __future__ import annotations

import base64
from io import BytesIO

import numpy as np

from manimlib.animation.transform import Transform
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.svg.svg_mobject import SVGMobject
from manimlib.renderer.texture import PixelArray
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

def test_copy_remaps_mask_vfx_to_the_copied_family() -> None:
    from manimlib.mobject.mobject import Group

    source = Group(Rectangle(), Rectangle())
    source_mask = source.submobjects[1]
    source.add_mask(source_mask)

    clone = source.copy()
    clone_mask = clone.submobjects[1]
    clone_effect = next(effect for effect in clone.get_vfx_list() if effect.name == "Mask")

    assert clone_effect is not source.get_vfx_list()[0]
    assert clone_effect.mask is clone_mask
    assert clone_effect.mask is not source_mask
    assert source.get_vfx_list()[0].mask is source_mask

def make_test_texture() -> PixelArray:
    pixels = np.array([
        [[255, 40, 80, 255], [40, 120, 255, 255]],
        [[255, 210, 40, 255], [20, 230, 160, 255]],
    ], dtype=np.uint8)
    return PixelArray(pixels)


def test_gradient_to_svg_image_interpolates_texture_contribution() -> None:
    gradient = Rectangle(width=3, height=2)
    gradient.set_fill_gradient(["#ff0000", "#0000ff"], direction=[1, 0, 0])

    image = Rectangle(width=3, height=2)
    image.set_fill("#ffffff", opacity=1)
    image.set_texture(make_test_texture(), flip_y=False)

    animated = gradient.copy()
    interpolate_at(animated, gradient, image, 0.25)

    assert "Texture" in animated.textures
    assert animated.textures["Texture"] is image.textures["Texture"]
    assert abs(animated.uniforms["texture_mix"] - 0.25) < 1e-6

    interpolate_at(animated, gradient, image, 1.0)
    assert "Texture" in animated.textures
    assert abs(animated.uniforms["texture_mix"] - 1.0) < 1e-6


def test_svg_image_to_gradient_interpolates_texture_contribution() -> None:
    image = Rectangle(width=3, height=2)
    image.set_fill("#ffffff", opacity=1)
    image.set_texture(make_test_texture(), flip_y=False)

    gradient = Rectangle(width=3, height=2)
    gradient.set_fill_gradient(["#ff0000", "#0000ff"], direction=[1, 0, 0])

    animated = image.copy()
    interpolate_at(animated, image, gradient, 0.25)

    assert "Texture" in animated.textures
    assert animated.textures["Texture"] is image.textures["Texture"]
    assert abs(animated.uniforms["texture_mix"] - 0.75) < 1e-6

    interpolate_at(animated, image, gradient, 1.0)
    assert not animated.textures
    assert abs(animated.uniforms["texture_mix"] - 0.0) < 1e-6

def interpolate_family_at(mobject, start, target, alpha: float):
    mobject.prepare_interpolation(start, target)
    for current_member, start_member, target_member in zip(
        mobject.get_family(),
        start.get_family(),
        target.get_family(),
    ):
        current_member.interpolate(start_member, target_member, alpha)
    return mobject


def make_png_data_uri(color: tuple[int, int, int, int] = (40, 120, 255, 255)) -> str:
    from PIL import Image

    with BytesIO() as stream:
        Image.new("RGBA", (2, 2), color).save(stream, format="PNG")
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_svg_gradient_and_svg_image_transform_blends_texture_both_ways() -> None:
    gradient_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"
         viewBox="0 0 100 50">
      <defs>
        <linearGradient id="gradient">
          <stop offset="0%" stop-color="#ff0000"/>
          <stop offset="100%" stop-color="#0000ff"/>
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="100" height="50" fill="url(#gradient)"/>
    </svg>
    """
    image_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"
         viewBox="0 0 100 50">
      <image x="0" y="0" width="100" height="50"
             href="{make_png_data_uri()}"/>
    </svg>
    """
    gradient = SVGMobject(svg_string=gradient_svg, should_center=False, height=1.0)
    image = SVGMobject(svg_string=image_svg, should_center=False, height=1.0)
    assert len(gradient.submobjects) == len(image.submobjects) == 1

    # Gradient -> image: the target texture is bound from frame zero, but its
    # contribution starts at zero and grows continuously.
    animated = gradient.copy()
    interpolate_family_at(animated, gradient, image, 0.25)
    animated_shape = animated.submobjects[0]
    image_shape = image.submobjects[0]
    assert "Texture" in animated_shape.textures
    assert animated_shape.textures["Texture"] is image_shape.textures["Texture"]
    assert abs(animated_shape.uniforms["texture_mix"] - 0.25) < 1e-6

    # Image -> gradient: keep the original texture until the final frame while
    # fading it out to reveal the gradient fill underneath.
    animated = image.copy()
    interpolate_family_at(animated, image, gradient, 0.25)
    animated_shape = animated.submobjects[0]
    assert "Texture" in animated_shape.textures
    assert abs(animated_shape.uniforms["texture_mix"] - 0.75) < 1e-6

    interpolate_family_at(animated, image, gradient, 1.0)
    assert not animated.submobjects[0].textures
    assert abs(animated.submobjects[0].uniforms["texture_mix"]) < 1e-6



def test_svg_image_to_image_transform_blends_sampled_pixels() -> None:
    source_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"
         viewBox="0 0 100 50">
      <image x="0" y="0" width="100" height="50"
             href="{make_png_data_uri((255, 0, 0, 255))}"/>
    </svg>
    """
    target_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"
         viewBox="0 0 100 50">
      <image x="0" y="0" width="100" height="50"
             href="{make_png_data_uri((0, 0, 255, 255))}"/>
    </svg>
    """
    source = SVGMobject(svg_string=source_svg, should_center=False, height=1.0)
    target = SVGMobject(svg_string=target_svg, should_center=False, height=1.0)
    animated = source.copy()

    interpolate_family_at(animated, source, target, 0.5)
    image = animated.submobjects[0]
    assert tuple(image.textures) == ("Texture", "TransformTargetTexture")
    assert image.textures["Texture"] is source.submobjects[0].textures["Texture"]
    assert image.textures["TransformTargetTexture"] is target.submobjects[0].textures["Texture"]
    assert image.uniforms["texture_mode"] == 3.0
    assert abs(image.uniforms["texture_mix"] - 0.5) < 1e-6

    interpolate_family_at(animated, source, target, 1.0)
    image = animated.submobjects[0]
    assert tuple(image.textures) == ("Texture",)
    assert image.textures["Texture"] is target.submobjects[0].textures["Texture"]
    assert image.uniforms["texture_mode"] == 0.0


def test_gradient_image_pixel_blend_keeps_endpoint_paint_colors() -> None:
    gradient = Rectangle(width=3, height=2)
    gradient.set_fill_gradient(["#ff0000", "#0000ff"], direction=[1, 0, 0])
    image = Rectangle(width=3, height=2)
    image.set_fill("#ffffff", opacity=1)
    image.set_texture(make_test_texture(), flip_y=False)

    animated = gradient.copy()
    interpolate_at(animated, gradient, image, 0.4)

    assert animated.uniforms["texture_mode"] == 1.0
    assert animated.uniforms["gradient_mode"] == 1.0
    np.testing.assert_allclose(
        animated.uniforms["gradient_colors"][:2, :3],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-5,
    )

    animated = image.copy()
    interpolate_at(animated, image, gradient, 0.4)
    assert animated.uniforms["texture_mode"] == 2.0
    assert animated.uniforms["gradient_mode"] == 1.0


def test_transform_aligns_svg_submobjects_by_id_not_document_order() -> None:
    source_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"
         viewBox="0 0 100 50">
      <rect id="left-shape" x="0" y="0" width="20" height="30" fill="#ff0000"/>
      <rect id="right-shape" x="75" y="0" width="20" height="30" fill="#00ff00"/>
    </svg>
    """
    target_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"
         viewBox="0 0 100 50">
      <rect id="right-shape" x="65" y="0" width="20" height="30" fill="#00ff00"/>
      <rect id="left-shape" x="10" y="0" width="20" height="30" fill="#ff0000"/>
    </svg>
    """
    source = SVGMobject(svg_string=source_svg, should_center=False, height=1.0)
    target = SVGMobject(svg_string=target_svg, should_center=False, height=1.0)
    original_target_order = [mob._svg_element_id for mob in target.submobjects]

    animation = Transform(source, target)
    animation.begin()

    assert original_target_order == ["right-shape", "left-shape"]
    assert [mob._svg_element_id for mob in animation.target_copy.submobjects] == [
        "left-shape", "right-shape",
    ]
    np.testing.assert_allclose(
        animation.target_copy.submobjects[0].get_center(),
        target.submobjects[1].get_center(),
        atol=1e-6,
    )
