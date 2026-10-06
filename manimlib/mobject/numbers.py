from __future__ import annotations
from functools import lru_cache

import numpy as np

from manimlib.constants import DOWN, LEFT, RIGHT, UP
from manimlib.constants import DEFAULT_MOBJECT_COLOR
from manimlib.mobject.svg.tex_mobject import Tex
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.types.vectorized_mobject import VGroup, VMobject
from manimlib.utils.paths import straight_path
from manimlib.utils.bezier import interpolate

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeVar, Callable
    from manimlib.mobject.mobject import Mobject
    from manimlib.typing import ManimColor, Vect3, Self

    T = TypeVar("T", bound=VMobject)


@lru_cache()
def char_to_cahced_mob(char: str, **text_config):
    if "\\" in char or char == "i":
        # This is for when the "character" is a LaTeX command
        # like ^\circ or \dots
        return Tex(char, **text_config)
    else:
        return Text(char, **text_config)


class DecimalNumber(VMobject):
    def __init__(
        self,
        number: float | complex = 0,
        color: ManimColor = DEFAULT_MOBJECT_COLOR,
        stroke_width: float = 0,
        fill_opacity: float = 1.0,
        fill_border_width: float = 0.5,
        num_decimal_places: int = 2,
        min_total_width: Optional[int] = 0,
        include_sign: bool = False,
        group_with_commas: bool = True,
        digit_buff_per_font_unit: float = 0.001,
        show_ellipsis: bool = False,
        unit: str | None = None,  # Aligned to bottom unless it starts with "^"
        include_background_rectangle: bool = False,
        hide_zero_components_on_complex: bool = True,
        edge_to_fix: Vect3 = LEFT,
        font_size: float = 48,
        text_config: dict = dict(),  # Do not pass in font_size here
        **kwargs
    ):
        self.num_decimal_places = num_decimal_places
        self.include_sign = include_sign
        self.group_with_commas = group_with_commas
        self.min_total_width = min_total_width
        self.digit_buff_per_font_unit = digit_buff_per_font_unit
        self.show_ellipsis = show_ellipsis
        self.unit = unit
        self.include_background_rectangle = include_background_rectangle
        self.hide_zero_components_on_complex = hide_zero_components_on_complex
        self.edge_to_fix = edge_to_fix
        self.font_size = font_size
        self.text_config = dict(text_config)
        skip_text_render = kwargs.pop("_skip_text_render", False)

        super().__init__(
            color=color,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
            fill_border_width=fill_border_width,
            **kwargs
        )

        if not skip_text_render:
            self.set_submobjects_from_number(number)
        self.init_colors()
        self.draw_fills_together_if_disjoint()

    def set_submobjects_from_number(self, number: float | complex) -> None:
        # Create the submobject list
        self.number = number
        self.num_string = self.get_num_string(number)

        # Submob_templates will be a list of cached Tex and Text mobjects,
        # with the intent of calling .copy or .become on them
        submob_templates = list(map(self.char_to_mob, self.num_string))
        if self.show_ellipsis:
            dots = self.char_to_mob("...")
            dots.arrange(RIGHT, buff=2 * dots[0].get_width())
            submob_templates.append(dots)
        if self.unit is not None:
            submob_templates.append(self.char_to_mob(self.unit))

        # Set internals
        font_size = self.get_font_size()
        if len(submob_templates) == len(self.submobjects):
            for sm, smt in zip(self.submobjects, submob_templates):
                sm.become(smt)
                sm.scale(font_size / smt.font_size)
        else:
            self.set_submobjects([
                smt.copy().scale(font_size / smt.font_size)
                for smt in submob_templates
            ])

        digit_buff = self.digit_buff_per_font_unit * font_size
        self.arrange(RIGHT, buff=digit_buff, aligned_edge=DOWN)

        # Handle alignment of special characters
        for i, c in enumerate(self.num_string):
            if c == "–" and len(self.num_string) > i + 1:
                self[i].align_to(self[i + 1], UP)
                self[i].shift(self[i + 1].get_height() * DOWN / 2)
            elif c == ",":
                self[i].shift(self[i].get_height() * DOWN / 2)
        if self.unit and self.unit.startswith("^"):
            self[-1].align_to(self, UP)

        if self.include_background_rectangle:
            self.add_background_rectangle()

    @classmethod
    async def create(cls, number: float | complex = 0, **kwargs):
        """Asynchronously construct a DecimalNumber in browser/Pyodide."""
        import sys
        if sys.platform != "emscripten":
            return cls(number, **kwargs)

        obj = cls(number, _skip_text_render=True, **kwargs)
        await obj.set_submobjects_from_number_async(number)
        obj.draw_fills_together_if_disjoint()
        return obj

    async def set_submobjects_from_number_async(self, number: float | complex) -> None:
        """Browser-safe counterpart of set_submobjects_from_number.

        Pyodide cannot use the synchronous Text/Tex renderers here because
        their SVG backend is asynchronous in the browser. Typst.create()
        provides the same awaitable construction path used by the rest of
        the browser text backend and defaults to New Computer Modern.
        """
        self.number = number
        self.num_string = self.get_num_string(number)

        from manimlib.mobject.svg.typst_mobject import Typst

        font_size = self.get_font_size()
        isolate = tuple(self.num_string)
        number_mob = await Typst.create(
            self.num_string,
            isolate=isolate,
            font_size=font_size,
            **self.text_config,
        )

        # Keep the complete number in a single Typst expression so that
        # punctuation uses its natural glyph metrics.  Do not render the
        # characters independently: that would normalize the decimal point
        # to the same nominal size as a digit.
        submob_templates = []
        for index in range(len(self.num_string)):
            indices = number_mob.find_submobject_indices_by_span(
                (index, index + 1)
            )
            if indices:
                submob_templates.append(
                    VGroup(*(number_mob.submobjects[i] for i in indices))
                )

        if self.show_ellipsis:
            dots = await Typst.create(
                "...",
                font_size=font_size,
                **self.text_config,
            )
            dots.arrange(RIGHT, buff=2 * dots[0].get_width())
            submob_templates.append(dots)

        if self.unit is not None:
            unit = await Typst.create(
                self.unit,
                font_size=font_size,
                **self.text_config,
            )
            submob_templates.append(unit)

        self.set_submobjects([
            smt.copy()
            for smt in submob_templates
        ])

        # Typst glyphs carry their own default fill color (black).  The
        # DecimalNumber color must still be inherited by the extracted
        # glyphs, just like the synchronous implementation does.
        self.set_color(self.color)

        digit_buff = self.digit_buff_per_font_unit * font_size
        self.arrange(RIGHT, buff=digit_buff, aligned_edge=DOWN)

        for i, c in enumerate(self.num_string):
            if c == "–" and len(self.num_string) > i + 1:
                self[i].align_to(self[i + 1], UP)
                self[i].shift(self[i + 1].get_height() * DOWN / 2)
            elif c == ",":
                self[i].shift(self[i].get_height() * DOWN / 2)
        if self.unit and self.unit.startswith("^"):
            self[-1].align_to(self, UP)
        if self.include_background_rectangle:
            self.add_background_rectangle()

    def get_num_string(self, number: float | complex) -> str:
        if isinstance(number, complex):
            if self.hide_zero_components_on_complex and number.imag == 0:
                number = number.real
                formatter = self.get_formatter()
            elif self.hide_zero_components_on_complex and number.real == 0:
                number = number.imag
                formatter = self.get_formatter() + "i"
            else:
                formatter = self.get_complex_formatter()
        else:
            formatter = self.get_formatter()
        if self.num_decimal_places == 0 and isinstance(number, float):
            number = int(number)
        num_string = formatter.format(number)

        rounded_num = np.round(number, self.num_decimal_places)
        if num_string.startswith("-") and rounded_num == 0:
            if self.include_sign:
                num_string = "+" + num_string[1:]
            else:
                num_string = num_string[1:]
        num_string = num_string.replace("-", "–")
        return num_string

    def char_to_mob(self, char: str) -> Text:
        return char_to_cahced_mob(char, **self.text_config)

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = straight_path
    ) -> Self:
        super().interpolate(mobject1, mobject2, alpha, path_func)
        if hasattr(mobject1, "font_size") and hasattr(mobject2, "font_size"):
            self.font_size = interpolate(mobject1.font_size, mobject2.font_size, alpha)

    def get_font_size(self) -> float:
        return self.font_size

    def get_formatter(self, **kwargs) -> str:
        """
        Configuration is based first off instance attributes,
        but overwritten by any kew word argument.  Relevant
        key words:
        - include_sign
        - group_with_commas
        - num_decimal_places
        - field_name (e.g. 0 or 0.real)
        """
        config = dict([
            (attr, getattr(self, attr))
            for attr in [
                "include_sign",
                "group_with_commas",
                "num_decimal_places",
                "min_total_width",
            ]
        ])
        config.update(kwargs)
        ndp = config["num_decimal_places"]
        return "".join([
            "{",
            config.get("field_name", ""),
            ":",
            "+" if config["include_sign"] else "",
            "0" + str(config.get("min_total_width", "")) if config.get("min_total_width") else "",
            "," if config["group_with_commas"] else "",
            f".{ndp}f" if ndp > 0 else "d",
            "}",
        ])

    def get_complex_formatter(self, **kwargs) -> str:
        return "".join([
            self.get_formatter(field_name="0.real"),
            self.get_formatter(field_name="0.imag", include_sign=True),
            "i"
        ])

    def get_tex(self):
        return self.num_string

    def set_value(self, number: float | complex) -> Self:
        move_to_point = self.get_edge_center(self.edge_to_fix)
        style = self.family_members_with_points()[0].get_style()
        self.set_submobjects_from_number(number)
        self.move_to(move_to_point, self.edge_to_fix)
        self.set_style(**style)
        for submob in self.get_family():
            submob.uniforms.update(self.uniforms)
        self.draw_fills_together_if_disjoint()
        return self

    async def set_value_async(self, number: float | complex) -> Self:
        """Browser-safe asynchronous counterpart of set_value."""
        move_to_point = self.get_edge_center(self.edge_to_fix)
        style = self.family_members_with_points()[0].get_style()
        await self.set_submobjects_from_number_async(number)
        self.move_to(move_to_point, self.edge_to_fix)
        self.set_style(**style)
        for submob in self.get_family():
            submob.uniforms.update(self.uniforms)
        self.draw_fills_together_if_disjoint()
        return self

    def _handle_scale_side_effects(self, scale_factor: float) -> Self:
        self.font_size *= scale_factor
        return self

    def get_value(self) -> float | complex:
        return self.number

    def increment_value(self, delta_t: float | complex = 1) -> Self:
        self.set_value(self.get_value() + delta_t)
        return self


class Integer(DecimalNumber):
    def __init__(
        self,
        number: int = 0,
        num_decimal_places: int = 0,
        **kwargs,
    ):
        super().__init__(number, num_decimal_places=num_decimal_places, **kwargs)

    def get_value(self) -> int:
        return int(np.round(super().get_value()))
