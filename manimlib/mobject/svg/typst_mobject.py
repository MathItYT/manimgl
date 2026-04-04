from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
import re
import emoji

from manimlib.config import manim_config
from manimlib.mobject.svg.string_mobject import StringMobject
from manimlib.mobject.svg.svg_mobject import get_svg_content_height
from manimlib.utils.cache import cache_on_disk
from manimlib.utils.color import color_to_hex
from manimlib.utils.simple_functions import hash_string
from manimlib.utils.typst import num_typst_symbols, find_pattern_matches, extract_grapheme_clusters

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manimlib.typing import Span


def _typst_executable() -> str:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError(
            "Typst CLI was not found. Install it first, for example with `winget install --id Typst.Typst`."
        )
    return typst


def _typst_document_prefix(font_size_pt: int, *, text_font: str = "", math_font: str = "") -> str:
    lines = [
        "#set page(width: auto, height: auto, margin: 0pt, fill: none)",
        f"#set text(size: {font_size_pt}pt)",
        f"#show math.equation: set text(size: {font_size_pt}pt)",
    ]
    if text_font:
        text_font = json.dumps(text_font, ensure_ascii=False) if isinstance(text_font, str) else "(" + ", ".join(json.dumps(f, ensure_ascii=False) for f in text_font) + ")"
        lines.append(f"#set text(font: {text_font})")
    if math_font:
        math_font = json.dumps(math_font, ensure_ascii=False) if isinstance(math_font, str) else "(" + ", ".join(json.dumps(f, ensure_ascii=False) for f in math_font) + ")"
        lines.append(f"#show math.equation: set text(font: {math_font})")
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=128)
@cache_on_disk
def typst_to_svg(document: str) -> str:
    typst = _typst_executable()
    temp_dir = Path(tempfile.gettempdir()) / f"manimgl_typst_{hash_string(document)}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    source_path = temp_dir / "input.typ"
    output_path = temp_dir / "output.svg"
    source_path.write_text(document, encoding="utf-8")

    command = [
        typst,
        "compile",
        "--format",
        "svg",
        str(source_path),
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Typst compilation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError("Typst did not produce an SVG output file.")

    return output_path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def get_typst_text_mob_scale_factor() -> float:
    ref_size = 48
    font_size_for_unit_height = manim_config.typst.font_size_for_unit_height
    document = _typst_document_prefix(ref_size) + "0"
    svg_string = typst_to_svg(document)
    svg_height = get_svg_content_height(svg_string)
    return 1.0 / (font_size_for_unit_height * svg_height)


@lru_cache(maxsize=1)
def get_typst_math_mob_scale_factor() -> float:
    ref_size = 48
    font_size_for_unit_height = manim_config.typst.font_size_for_unit_height
    document = _typst_document_prefix(ref_size) + "$ 0 $"
    svg_string = typst_to_svg(document)
    svg_height = get_svg_content_height(svg_string)
    return 1.0 / (font_size_for_unit_height * svg_height)


class _BaseTypstMobject(StringMobject):
    height = None

    def __init__(
        self,
        content: str,
        *,
        font_size: int = 48,
        text_font: str | list[str] = "",
        math_font: str | list[str] = "",
        typst_to_color_map: dict = dict(),
        t2c: dict = dict(),
        isolate=(),
        protect=(),
        use_labelled_svg: bool = False,
        color=None,
        fill_color=None,
        fill_opacity: float | None = None,
        stroke_width: float | None = None,
        stroke_color=None,
        stroke_opacity: float | None = None,
        **kwargs,
    ):
        self.content = content
        self.font_size = font_size
        self.text_font = text_font
        self.math_font = math_font
        self.typst_to_color_map = dict(**t2c, **typst_to_color_map)
        super().__init__(
            content,
            fill_color=fill_color or color,
            stroke_color=stroke_color or color,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
            stroke_opacity=stroke_opacity,
            isolate=isolate,
            protect=protect,
            use_labelled_svg=use_labelled_svg,
            **kwargs,
        )

    def get_svg_string_by_content(self, content: str) -> str:
        raise NotImplementedError

    @staticmethod
    def get_command_matches(string: str) -> list[re.Match]:
        # Typst commands injected by this class are generated during reconstruction,
        # not part of the original content string. So there are no intrinsic command matches.
        return []

    @staticmethod
    def get_command_flag(match_obj: re.Match) -> int:
        return 0

    @staticmethod
    def replace_for_content(match_obj: re.Match) -> str:
        return match_obj.group()

    @staticmethod
    def replace_for_matching(match_obj: re.Match) -> str:
        return match_obj.group()

    @staticmethod
    def get_attr_dict_from_command_pair(
        open_command: re.Match, close_command: re.Match
    ) -> dict[str, str] | None:
        return None

    def get_configured_items(self) -> list[tuple[Span, dict[str, str]]]:
        return [
            (span, {})
            for selector in self.typst_to_color_map
            for span in self.find_spans_by_selector(selector)
        ]

    @staticmethod
    def get_command_string(
        attr_dict: dict[str, str], is_end: bool, label_hex: str | None
    ) -> str:
        if label_hex is None:
            return ""
        if is_end:
            return "]"
        return f"#text(fill: rgb(\"{label_hex}\"))["

    def get_content_prefix_and_suffix(
        self, is_labelled: bool
    ) -> tuple[str, str]:
        if is_labelled:
            return "", ""
        # Match Tex/StringMobject behavior: for the non-labelled render, apply base color globally.
        return f"#text(fill: rgb(\"{color_to_hex(self.base_color)}\"))[", "]"

    # Typst-specific selection and coloring methods
    def get_parts_by_typst(self, selector: str) -> list:
        """
        Get parts of the mobject by selector (substring or pattern).
        Alias for select_parts() matching TexMobject API.
        
        Supports:
        - Exact string matching: get_parts_by_typst("hello")
        - Regex patterns: get_parts_by_typst(r"\\d+")  # matches numbers
        - Case-sensitive substring search
        
        Args:
            selector: String or regex pattern to search for
            
        Returns:
            VGroup of matching parts
        """
        return self.select_parts(selector)

    def get_part_by_typst(self, selector: str, index: int = 0):
        """
        Get a specific part of the mobject by selector and index.
        Alias for select_part() matching TexMobject API.
        
        Args:
            selector: String or regex pattern to search for
            index: Which occurrence to select (0 = first, 1 = second, etc.)
            
        Returns:
            Single VMobject at the specified index
        """
        return self.select_part(selector, index)

    def substr_to_path_count(self, substr: str) -> int:
        """
        Count the number of visible symbols in a Typst substring.
        Override of StringMobject's method for Typst-specific counting.
        
        This respects Typst's symbol counting rules where markup characters
        like *, _, #, {}, [], etc. don't count as visible symbols.
        
        Args:
            substr: Typst substring to count
            
        Returns:
            Number of visible symbols (grapheme clusters that are not markup)
        """
        return num_typst_symbols(substr)

    def get_symbol_substrings(self):
        """
        Return list of symbol substrings in this Typst mobject.
        
        For Typst, symbols are contiguous sequences of non-whitespace,
        non-structural characters. This respects Typst's structure:
        - Math expressions like "x^2" stay together
        - Operators like "+" are separate
        - Markup markers are excluded
        
        Returns:
            List of "symbol" substrings (as Typst parses them)
        """
        # In Typst (from syntax docs), symbols are roughly:
        # - Contiguous alphanumerics and common operators
        # - Math subscripts/superscripts stay with their base
        # Exclude pure markup characters: {}[]#$@<>*_`&
        pattern = r"[^\s{}\[\]#$@<>*_`&]+"
        return re.findall(pattern, self.string)

    def find_all_typst_matches(self, pattern: str | re.Pattern) -> list[dict]:
        """
        Find all matches of a pattern in the Typst content.
        
        Mimics Typst's str.matches() API which returns list of match dictionaries
        with 'start', 'end', 'text', and 'captures' fields.
        
        Args:
            pattern: String (exact match) or regex pattern
            
        Returns:
            List of match dictionaries with structure:
                {
                    'start': byte_index,
                    'end': byte_index,
                    'text': matched_string,
                    'captures': [captured_groups]
                }
        """
        matches = []
        text = self.string
        
        if isinstance(pattern, str):
            # Exact string matching
            pattern_escaped = re.escape(pattern)
            regex = re.compile(pattern_escaped)
        else:
            regex = pattern
        
        for match in regex.finditer(text):
            match_dict = {
                'start': match.start(),
                'end': match.end(),
                'text': match.group(),
                'captures': list(match.groups()) if match.groups() else []
            }
            matches.append(match_dict)
        
        return matches
    
    def select_parts_case_insensitive(self, selector: str):
        """
        Select parts using case-insensitive matching.
        
        Useful for "hello"/"Hello"/"HELLO" all matching.
        
        Args:
            selector: String to search for (case-insensitive)
            
        Returns:
            VGroup of matching parts
        """
        pattern = re.compile(re.escape(selector), re.IGNORECASE)
        return self.select_parts(pattern)


class TypstMobject(_BaseTypstMobject):
    """Render Typst math as an SVG mobject."""

    def __init__(
        self,
        *typst_strings: str,
        font_size: int = 48,
        text_font: str | list[str] = ["New Computer Modern", "Twitter Color Emoji"],
        math_font: str | list[str] = ["New Computer Modern Math", "Twitter Color Emoji"],
        **kwargs,
    ):
        typst_string = " ".join(typst_strings).strip()
        if typst_string.startswith("$") and typst_string.endswith("$"):
            typst_string = typst_string[1:-1].strip()
        self.typst_string = typst_string or "0"
        super().__init__(
            self.typst_string,
            font_size=font_size,
            text_font=text_font,
            math_font=math_font,
            **kwargs,
        )
        self.set_color_by_typst_to_color_map(self.typst_to_color_map)
        # Scale factor is calculated for 48pt reference, so normalize by dividing by 48
        self.scale(get_typst_math_mob_scale_factor() * font_size)

    def get_svg_string_by_content(self, content: str) -> str:
        # Always render at 48pt (the reference size)
        # Font size scaling happens via self.scale() in __init__
        prefix = _typst_document_prefix(
            48,  # Reference size, not self.font_size
            text_font=self.text_font,
            math_font=self.math_font,
        )
        document = prefix + f"$ {content} $"
        return typst_to_svg(document)
    
    def get_content_prefix_and_suffix(self, is_labelled):
        prefix, suffix = super().get_content_prefix_and_suffix(is_labelled)
        return prefix + "$ ", " $" + suffix

    def get_configured_items(self) -> list[tuple[Span, dict[str, str]]]:
        # In math mode, we cannot inject Typst code for coloring
        # Return empty list; coloring will be applied via set_parts_color on the SVG paths
        return []

    def set_color_by_typst(self, selector, color):
        return self.set_parts_color(selector, color)

    def set_color_by_typst_to_color_map(self, color_map: dict):
        return self.set_parts_color_by_dict(color_map)

    def get_typst(self) -> str:
        return self.get_string()


class TypstTextMobject(_BaseTypstMobject):
    """Render plain text with Typst as an SVG mobject."""

    def __init__(
        self,
        text: str,
        font_size: int = 48,
        text_font: str | list[str] = ["New Computer Modern", "Twitter Color Emoji"],
        math_font: str | list[str] = ["New Computer Modern Math", "Twitter Color Emoji"],
        **kwargs,
    ):
        self.text = text
        super().__init__(
            text,
            font_size=font_size,
            text_font=text_font,
            math_font=math_font,
            **kwargs,
        )
        self.set_color_by_typst_to_color_map(self.typst_to_color_map)
        # Scale factor is calculated for 48pt reference, so normalize by dividing by 48
        self.scale(get_typst_text_mob_scale_factor() * font_size)

    def get_svg_string_by_content(self, content: str) -> str:
        # Always render at 48pt (the reference size)
        # Font size scaling happens via self.scale() in __init__
        prefix = _typst_document_prefix(48, text_font=self.text_font)
        document = prefix + content
        return typst_to_svg(document)

    def set_color_by_typst(self, selector, color):
        return self.set_parts_color(selector, color)

    def set_color_by_typst_to_color_map(self, color_map: dict):
        return self.set_parts_color_by_dict(color_map)

    def get_typst(self) -> str:
        return self.get_string()


class MarkdownMobject(TypstTextMobject):
    """Render Markdown text using Typst as an SVG mobject."""
    def __init__(
        self,
        markdown: str,
        font_size: int = 48,
        text_font: str | list[str] = ["New Computer Modern", "Twitter Color Emoji"],
        math_font: str | list[str] = ["New Computer Modern Math", "Twitter Color Emoji"],
        **kwargs,
    ):
        super().__init__(
            self._build_markdown_as_typst(markdown),
            font_size=font_size,
            text_font=text_font,
            math_font=math_font,
            **kwargs,
        )
    
    @staticmethod
    def _build_markdown_as_typst(markdown: str) -> str:
        markdown = emoji.emojize(markdown, language="alias")
        markdown_with_quotes = json.dumps(markdown, ensure_ascii=False)
        template = f"""
#import "@preview/cmarker:0.1.8"
#import "@preview/mitex:0.2.6": mitex
#cmarker.render({markdown_with_quotes}, math: mitex)
"""
        return template.strip()
