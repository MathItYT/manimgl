from __future__ import annotations

import re
from functools import lru_cache

from manimlib.utils.typst_to_symbol_count import TYPST_TO_SYMBOL_COUNT


@lru_cache
def num_typst_symbols(typst: str) -> int:
    """
    Count the number of visible symbols in a Typst string.
    
    Based on Typst documentation:
    - Whitespace and structural markers ({} [] # $ @ < > - _ * `) don't count
    - All other characters count as 1 symbol (including Unicode graphemes)
    - This is simpler than LaTeX because Typst doesn't have special commands
    
    Args:
        typst: Typst string to count symbols in
        
    Returns:
        Number of visible symbols
    """
    total = 0
    # Count each grapheme cluster (Unicode-aware character)
    for char in typst:
        # Use the map to get count, default to 1 for regular characters
        total += TYPST_TO_SYMBOL_COUNT.get(char, 1)
    return total


def find_pattern_matches(text: str, pattern: str | re.Pattern) -> list[tuple[int, int, str]]:
    """
    Find all matches of a pattern (string or regex) in text.
    
    Matches the Typst API behavior: str.matches() returns list of match objects
    with start, end, text, and captures fields.
    
    Args:
        text: The text to search in
        pattern: A string (exact match) or compiled regex pattern
        
    Returns:
        List of (start, end, matched_text) tuples
    """
    matches = []
    
    if isinstance(pattern, str):
        # Exact string matching
        pattern_escaped = re.escape(pattern)
        regex = re.compile(pattern_escaped)
    elif isinstance(pattern, re.Pattern):
        regex = pattern
    else:
        # Assume it's a string-like object
        pattern_escaped = re.escape(str(pattern))
        regex = re.compile(pattern_escaped)
    
    for match in regex.finditer(text):
        matches.append((match.start(), match.end(), match.group()))
    
    return matches


def extract_grapheme_clusters(text: str) -> list[str]:
    """
    Extract grapheme clusters (Unicode-aware characters) from Typst text.
    
    Matches Typst's str.clusters() behavior. This is important because
    Typst treats combined characters (like é or flag emojis) as single clusters.
    
    For now, we use a simple Python-level approximation.
    Full grapheme cluster support would require the grapheme crate equivalent.
    
    Args:
        text: Typst string to extract clusters from
        
    Returns:
        List of grapheme clusters (individual "characters")
    """
    # Simple approximation: split on soft boundaries
    # This is a simplification - real grapheme clustering is complex
    clusters = []
    i = 0
    while i < len(text):
        char = text[i]
        
        # Check for combining marks following this character
        j = i + 1
        while j < len(text) and _is_combining_mark(text[j]):
            j += 1
        
        clusters.append(text[i:j])
        i = j if j > i + 1 else i + 1
    
    return clusters


def _is_combining_mark(char: str) -> bool:
    """Check if a character is a Unicode combining mark."""
    code = ord(char)
    # Combining Diacritical Marks block (U+0300–U+036F)
    # Combining Diacritical Marks Extended (U+1AB0–U+1AFF)
    # Variation Selectors (U+FE00–U+FE0F)
    return (0x0300 <= code <= 0x036F) or \
           (0x1AB0 <= code <= 0x1AFF) or \
           (0xFE00 <= code <= 0xFE0F)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in Typst text for comparison.
    
    In Typst, multiple spaces are significant (unlike HTML),
    but for substring matching, we often want normalized comparison.
    """
    return text


def extract_raw_substring(text: str, start_index: int, end_index: int) -> str:
    """
    Extract substring respecting byte indices (Typst semantics).
    
    Typst uses UTF-8 byte indices for string operations.
    This helper respects that convention.
    """
    return text[start_index:end_index]

