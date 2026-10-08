from __future__ import annotations

from functools import lru_cache
import re
import sys
from typing import Any, TYPE_CHECKING

from manimlib.config import manim_config
from manimlib.constants import WHITE
from manimlib.mobject.svg.string_mobject import StringMobject
from manimlib.mobject.svg.svg_mobject import SVGMobject
from manimlib.mobject.types.vectorized_mobject import VGroup, VMobject
from manimlib.utils.color import color_to_hex

from manimlib.utils.typst_file_writing import typst_to_svg

if sys.platform == 'emscripten':
    from manimlib.utils.browser_typst import typst_to_svg_async

if TYPE_CHECKING:
    from manimlib.typing import ManimColor, Selector, Span


@lru_cache(maxsize=1)
def get_typst_mob_scale_factor() -> float:
    if sys.platform == "emscripten":
        raise RuntimeError(
            "get_typst_mob_scale_factor() is not available in Pyodide. "
            "Use get_typst_mob_scale_factor_async() instead."
        )
    # Render a reference "0" and calibrate so that font_size_for_unit_height
    # gives a height of 1 manim unit. Compensates for platform dvisvgm differences.
    font_size_for_unit_height = manim_config.tex.font_size_for_unit_height
    svg_string = typst_to_svg("#set text(size: 10pt)\n0")
    svg_height = SVGMobject(svg_string=svg_string).get_height()
    return 1.0 / (font_size_for_unit_height * svg_height)


TYPST_MOB_SCALE_FACTOR_ASYNC = None


async def get_typst_mob_scale_factor_async() -> float:
    if sys.platform != "emscripten":
        raise RuntimeError(
            "get_typst_mob_scale_factor_async() is only available in Pyodide. "
            "Use get_typst_mob_scale_factor() instead."
        )
    font_size_for_unit_height = manim_config.tex.font_size_for_unit_height
    svg_string = await typst_to_svg_async("#set text(size: 10pt)\n0")
    svg_height = SVGMobject(svg_string=svg_string).get_height()
    return 1.0 / (font_size_for_unit_height * svg_height)


def to_hex(color: Any) -> str:
    if isinstance(color, str):
        if color.startswith("#"):
            return color.lower()
        try:
            return color_to_hex(color).lower()
        except Exception:
            return f"#{color}".lower()
    try:
        return color_to_hex(color).lower()
    except Exception:
        return "#ffffff"


class SingleStringTypst(StringMobject):
    def __init__(
        self,
        typst_string: str,
        base_color: ManimColor = WHITE,
        color: ManimColor | None = None,
        isolate: Any = (),
        protect: Any = (),
        math_mode: bool = True,
        preamble: str = "",
        font: str | None = "New Computer Modern",
        code_font: str | None = "JetBrains Mono",
        code_theme_file: str | None = None,
        font_size: float = 48.0,
        t2c: dict[Any, ManimColor] | None = None,
        typst_to_color_map: dict[Any, ManimColor] | None = None,
        **kwargs,
    ):
        self.font_size = font_size
        self.math_mode = math_mode
        self.preamble = preamble
        self.font = font
        self.code_theme_file = code_theme_file
        self.code_font = code_font
        self.base_color = color if color is not None else base_color
        self.t2c = t2c or typst_to_color_map or {}

        isolate = () if isolate is None else isolate
        protect = () if protect is None else protect

        cleaned_string = typst_string.strip()
        if self.math_mode:
            if cleaned_string.startswith("$") and cleaned_string.endswith("$") and len(cleaned_string) >= 2:
                cleaned_string = cleaned_string[1:-1].strip()

        super().__init__(
            cleaned_string,
            base_color=None,
            fill_color=None,
            stroke_color=None,
            isolate=isolate,
            protect=protect,
            stroke_width=None,
            **kwargs,
        )

        if ("height" not in kwargs or kwargs["height"] is None) and ("width" not in kwargs or kwargs["width"] is None):
            # Use exactly the same font-size -> Manim-unit calibration as the
            # native Typst path. Browser Typst SVGs can use different document
            # units, so calibrating from their raw SVG height produces a
            # different scale than native Manim.
            if sys.platform != "emscripten":
                scale = get_typst_mob_scale_factor() * self.font_size
                self.scale(scale)
                self.scale_stroke_widths(scale)
        self._char_to_submob_map = self._build_char_to_submob_map()

    @classmethod
    async def create(cls, typst_string: str, **kwargs):
        """Asynchronously construct a Typst mobject in Pyodide."""
        global TYPST_MOB_SCALE_FACTOR_ASYNC
        import sys
        if sys.platform != "emscripten":
            return cls(typst_string, **kwargs)
        # First build a lightweight parser instance to obtain the exact Typst source,
        # including Manim's color labels and document preamble. The actual geometry is
        # then constructed from the SVG returned by typst-wasm.
        # The StringMobject constructor normally renders a second, labelled
        # SVG for span matching. That path is synchronous, while browser
        # Typst compilation is asynchronous. Build the labelled document
        # directly and tell the final object to use it, so no synchronous
        # Typst compilation is attempted inside __init__.
        probe = cls(
            typst_string,
            _svg_override='<svg xmlns="http://www.w3.org/2000/svg"/>',
            use_labelled_svg=True,
            **kwargs,
        )
        content = probe.get_content(is_labelled=True)
        svg = await typst_to_svg_async(content)
        if not isinstance(svg, str):
            raise TypeError(
                f"Browser Typst compiler returned {type(svg).__name__}; expected SVG text."
            )
        obj = cls(
            typst_string,
            _svg_override=svg,
            **kwargs,
        )
        if ("height" not in kwargs or kwargs["height"] is None) and ("width" not in kwargs or kwargs["width"] is None):
            if TYPST_MOB_SCALE_FACTOR_ASYNC is None:
                TYPST_MOB_SCALE_FACTOR_ASYNC = await get_typst_mob_scale_factor_async()
            scale = TYPST_MOB_SCALE_FACTOR_ASYNC * obj.font_size
            obj.scale(scale)
            obj.scale_stroke_widths(scale)
        return obj

    def _build_char_to_submob_map(self) -> list[list[int]]:
        mapping: list[list[int]] = [[] for _ in range(len(self.string))]
        total_submobs = len(self.submobjects)
        submob_idx = 0

        func_paren_indices = set()
        for func_pat in (r"sqrt\s*\(", r"root\s*\("):
            for m in re.finditer(func_pat, self.string):
                open_paren = m.end() - 1
                func_paren_indices.add(open_paren)
                depth = 1
                for j in range(open_paren + 1, len(self.string)):
                    if self.string[j] == "(":
                        depth += 1
                    elif self.string[j] == ")":
                        depth -= 1
                        if depth == 0:
                            func_paren_indices.add(j)
                            break

        token_regex = re.compile(
            r"(?P<sqrt>sqrt)|"
            r"(?P<root>root)|"
            r"(?P<syntax>"
            r"\\(?=[*$`_])"
            r"|\*"
            r"|(?<!\w)_(?!\w)"
            r"|```(?:[a-zA-Z0-9]+)?"
            r"|`"
            r"|\$"
            r"|\^|_"
            r")|"
            r"(?P<multi>"
            r"\b(?:arrow|vec|mat|cases|binom|exists|forall|or|and|"
            r"alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|"
            r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|"
            r"sum|product|integral|lim|"
            r"oo|infinity|RR|QQ|ZZ|NN|CC|EE|PP|HH|"
            r"notin|in|times|div|cdot|approx|equiv|subset|supset)\b|"
            r"<->|<=>|->|=>|<-|<=|!=|>="
            r")|"
            r"(?P<space>\s+)|"
            r"(?P<char>.)",
            re.DOTALL
        )

        in_math = self.math_mode

        for match in token_regex.finditer(self.string):
            start, end = match.span()

            if match.group("syntax") == "$":
                in_math = not in_math
                for k in range(start, end):
                    mapping[k] = []
                continue

            if match.group("sqrt") or match.group("root"):
                assigned = []
                for _ in range(2):
                    if submob_idx < total_submobs:
                        assigned.append(submob_idx)
                        submob_idx += 1
                for k in range(start, end):
                    mapping[k] = assigned

            elif match.group("syntax") or match.group("space"):
                for k in range(start, end):
                    mapping[k] = []

            elif match.group("multi") and in_math:
                assigned = []
                if submob_idx < total_submobs:
                    assigned.append(submob_idx)
                    submob_idx += 1
                for k in range(start, end):
                    mapping[k] = assigned

            else:
                if start in func_paren_indices:
                    mapping[start] = []
                else:
                    assigned = []
                    if submob_idx < total_submobs:
                        assigned.append(submob_idx)
                        submob_idx += 1
                    mapping[start] = assigned

        return mapping

    def find_submobject_indices_by_span(self, span: Span) -> list[int]:
        submob_indices = []
        for i in range(span[0], span[1]):
            if i < len(self._char_to_submob_map):
                for idx in self._char_to_submob_map[i]:
                    if idx < len(self.submobjects) and idx not in submob_indices:
                        submob_indices.append(idx)
        return submob_indices

    def find_submobject_indices_by_spans(self, spans: list[Span]) -> list[int]:
        submob_indices = []
        for span in spans:
            for idx in self.find_submobject_indices_by_span(span):
                if idx not in submob_indices:
                    submob_indices.append(idx)
        return submob_indices

    # --- Métodos abstractos de StringMobject en ManimGL ---

    def get_command_flag(self, match: re.Match) -> bool:
        return match.group() == "}}"

    def get_command_string(
        self,
        attr_dict: dict[str, str],
        is_end: bool = False,
        label_hex: str | None = None,
        label: int | None = None,
        **kwargs,
    ) -> str:
        if is_end:
            return "$]" if self.math_mode else "]"

        if label_hex is not None:
            color = label_hex
        elif label is not None:
            color = f"#{label:06x}"
        elif "color" in attr_dict:
            color = attr_dict["color"]
        else:
            color = ""

        if self.math_mode:
            return f'#_mc("{color}")[$'
        return f'#_mc("{color}")['

    def get_command_matches(self, string: str) -> list[re.Match]:
        return list(re.finditer(r"\{\{|\}\}", string))

    def replace_for_content(self, match: re.Match) -> str:
        return ""

    def replace_for_matching(self, match: re.Match) -> str:
        return ""

    def get_attr_dict_from_command_pair(
        self,
        open_match: re.Match,
        close_match: re.Match,
    ) -> dict[str, str]:
        return {}

    def get_configured_items(self) -> list[tuple[Selector, dict[str, str]]]:
        return [
            (selector, {"color": to_hex(color)})
            for selector, color in self.t2c.items()
        ]

    def get_content_prefix_and_suffix(self, is_labelled: bool) -> tuple[str, str]:
        hex_color = to_hex(self.base_color)
        font_rule = f'#set text(font: "{self.font}")\n' if self.font else ""
        code_font_rule = f'#show raw: set text(font: "{self.code_font}")\n' if self.code_font else ""
        code_theme_rule = f'#set raw(theme: "{self.code_theme_file}")\n' if self.code_theme_file else ""

        doc_head = (
            f'#set page(width: auto, height: auto, margin: 0pt, fill: none)\n'
            f'#set text(fill: rgb("{hex_color}"), size: 10pt, ligatures: false)\n'
            f'{font_rule}'
            f'{code_theme_rule}'
            f'{code_font_rule}'
            f'#show raw: set text(ligatures: false)\n'
            f'#show raw: set text(features: (calt: 0))\n'
            f'#show math.equation: set text(ligatures: false)\n'
            f'#let _mc(c, it) = if c != "" {{ text(fill: rgb(c), it) }} else {{ it }}\n'
            f'{self.preamble}'
        ).strip()

        if self.math_mode:
            return f"{doc_head}\n$ ", " $\n"
        return f"{doc_head}\n", "\n"

    def get_svg_string_by_content(self, content: str) -> str:
        if self._svg_override is not None:
            return self._svg_override
        return typst_to_svg(content)

    def select_parts(self, selector: Selector) -> VGroup:
        spans = self.find_spans_by_selector(selector)
        # Cada span (palabra) se empaqueta en su propio VGroup independiente
        return VGroup(*(
            VGroup(*(self.submobjects[i] for i in self.find_submobject_indices_by_span(span)))
            for span in spans
        ))

    def get_parts_by_string(self, substr: str) -> VGroup:
        return self.select_parts(substr)

    def set_color_by_string(self, substr: str, color: ManimColor):
        self.get_parts_by_string(substr).set_color(color)
        return self

    def set_parts_color(self, selector: Selector, color: ManimColor):
        return self.set_color_by_string(selector, color)

    def get_parts_by_typst(self, typst_str: str) -> VGroup:
        return self.get_parts_by_string(typst_str)

    def get_part_by_typst(self, typst_str: str, index: int = 0) -> VMobject:
        parts = self.get_parts_by_typst(typst_str)
        return parts[index]

    def set_color_by_typst(self, typst_str: str, color: ManimColor):
        return self.set_color_by_string(typst_str, color)

    def __getitem__(self, value):
        if isinstance(value, str):
            return self.get_parts_by_string(value)
        return super().__getitem__(value)


class Typst(SingleStringTypst):
    def __init__(
        self,
        *typst_strings: str,
        arg_separator: str = " ",
        isolate: Any = (),
        protect: Any = (),
        math_mode: bool = True,
        **kwargs,
    ):
        isolate = () if isolate is None else isolate
        protect = () if protect is None else protect

        if len(typst_strings) > 1 and not isolate:
            isolate = tuple(s.strip() for s in typst_strings if s.strip())

        full_string = arg_separator.join(typst_strings) if typst_strings else ""
        super().__init__(
            full_string,
            isolate=isolate,
            protect=protect,
            math_mode=math_mode,
            **kwargs,
        )


class TypstText(SingleStringTypst):
    def __init__(
        self,
        *typst_strings: str,
        arg_separator: str = " ",
        isolate: Any = (),
        protect: Any = (),
        math_mode: bool = False,
        **kwargs,
    ):
        isolate = () if isolate is None else isolate
        protect = () if protect is None else protect

        if len(typst_strings) > 1 and not isolate:
            isolate = tuple(s.strip() for s in typst_strings if s.strip())

        full_string = arg_separator.join(typst_strings) if typst_strings else ""
        super().__init__(
            full_string,
            isolate=isolate,
            protect=protect,
            math_mode=math_mode,
            **kwargs,
        )