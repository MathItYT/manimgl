from __future__ import annotations

import re
from functools import lru_cache

from manimlib.utils.tex_to_symbol_count import TEX_TO_SYMBOL_COUNT


def parse_tex_submob_map(tex: str) -> tuple[int, list[tuple[tuple[int, int], list[int], str]], list[list[int]]]:
    """
    Parses a TeX string into Typst-rendered submobject mappings.
    Accounts for Typst's fraction rendering order: [numerator, denominator, bar].
    Returns (total_submob_count, token_spans, char_to_submobs).
    """
    char_to_submobs: list[list[int]] = [[] for _ in range(len(tex))]
    token_spans: list[tuple[tuple[int, int], list[int], str]] = []

    def get_arg_span(s: str, start_idx: int) -> tuple[int, int]:
        while start_idx < len(s) and s[start_idx].isspace():
            start_idx += 1
        if start_idx >= len(s):
            return start_idx, start_idx
        if s[start_idx] == "{":
            depth = 1
            idx = start_idx + 1
            while idx < len(s) and depth > 0:
                if s[idx] == "{" and (idx == 0 or s[idx - 1] != "\\"):
                    depth += 1
                elif s[idx] == "}" and (idx == 0 or s[idx - 1] != "\\"):
                    depth -= 1
                idx += 1
            return start_idx, idx
        else:
            if s[start_idx] == "\\":
                m = re.match(r"\\[a-zA-Z]+|\\.", s[start_idx:])
                return start_idx, start_idx + (m.end() if m else 1)
            return start_idx, start_idx + 1

    pending_bars_at_pos: dict[int, list[tuple[int, int]]] = {}
    for m in re.finditer(r"\\(frac|dfrac|tfrac|cfrac)\b", tex):
        frac_span = m.span()
        n_start, n_end = get_arg_span(tex, frac_span[1])
        d_start, d_end = get_arg_span(tex, n_end)
        if d_end not in pending_bars_at_pos:
            pending_bars_at_pos[d_end] = []
        pending_bars_at_pos[d_end].append(frac_span)

    commands_pattern = re.compile(r"""
        (?P<env>\\(begin|end)(\{\w+\})?(\{\w+\})?(\[\w+\])?)|
        (?P<phantom>\\phantom\{[^}]*\})|
        (?P<sqrt>\\sqrt\[[0-9]+\])|
        (?P<escaped_brace>\\[{}])|
        (?P<frac>\\(frac|dfrac|tfrac|cfrac)\b)|
        (?P<cmd>\\[a-zA-Z!,-/:;<>]+)|
        (?P<char>[^\s])
    """, re.VERBOSE)

    current_submob_idx = 0

    def check_pending_bars(pos: int):
        nonlocal current_submob_idx
        if pos in pending_bars_at_pos:
            bars = sorted(pending_bars_at_pos[pos], key=lambda sp: sp[0], reverse=True)
            for f_span in bars:
                token_spans.append((f_span, [current_submob_idx], "frac_bar"))
                for c in range(f_span[0], f_span[1]):
                    char_to_submobs[c].append(current_submob_idx)
                current_submob_idx += 1
            del pending_bars_at_pos[pos]

    pos = 0
    while pos < len(tex):
        check_pending_bars(pos)
        m = commands_pattern.match(tex, pos)
        if not m:
            pos += 1
            continue

        span = m.span()
        if m.group("env") or m.group("phantom"):
            pos = span[1]
            check_pending_bars(pos)
            continue
        elif m.group("frac"):
            pos = span[1]
            check_pending_bars(pos)
            continue
        elif m.group("sqrt"):
            count = len(m.group()) - 5
            indices = list(range(current_submob_idx, current_submob_idx + count))
            token_spans.append((span, indices, m.group()))
            for c in range(span[0], span[1]):
                char_to_submobs[c].extend(indices)
            current_submob_idx += count
        elif m.group("escaped_brace"):
            indices = [current_submob_idx]
            token_spans.append((span, indices, m.group()))
            for c in range(span[0], span[1]):
                char_to_submobs[c].extend(indices)
            current_submob_idx += 1
        elif m.group("cmd"):
            cmd = m.group("cmd")
            count = TEX_TO_SYMBOL_COUNT.get(cmd, 1)
            indices = list(range(current_submob_idx, current_submob_idx + count))
            token_spans.append((span, indices, cmd))
            for c in range(span[0], span[1]):
                char_to_submobs[c].extend(indices)
            current_submob_idx += count
        elif m.group("char"):
            ch = m.group("char")
            count = 0 if ch in "^{}_$\\&~" else 1
            if count > 0:
                indices = list(range(current_submob_idx, current_submob_idx + count))
                token_spans.append((span, indices, ch))
                for c in range(span[0], span[1]):
                    char_to_submobs[c].extend(indices)
                current_submob_idx += count

        pos = span[1]
        check_pending_bars(pos)

    for p in sorted(pending_bars_at_pos.keys()):
        check_pending_bars(p)

    return current_submob_idx, token_spans, char_to_submobs


@lru_cache
def num_tex_symbols(tex: str) -> int:
    total, _, _ = parse_tex_submob_map(tex)
    return total


def remove_tex_environments(tex: str) -> str:
    # Handle \phantom{...} with any content
    tex = re.sub(r"\\phantom\{[^}]*\}", "", tex)
    # Handle other environment commands
    tex = re.sub(r"\\(begin|end)(\{\w+\})?(\{\w+\})?(\[\w+\])?", "", tex)
    return tex
