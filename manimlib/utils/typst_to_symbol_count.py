# Typst symbol count dictionary
# Based on Typst documentation: https://typst.app/docs/reference/
#
# Typst markup special characters (from syntax documentation):
# - Strong emphasis: * 
# - Emphasis: _
# - Raw text: `
# - Code prefix: #
# - Brackets: [ ]
# - Braces: { }
# - Math mode: $
# - Labels: < >
# - References: @
# - Headings: =
# - Lists: -, +, /
# - Symbol shorthand: ~, ---, etc.
#
# These don't count as visible symbols:
TYPST_TO_SYMBOL_COUNT = {
    # Whitespace (0 symbols)
    " ": 0,
    "\n": 0,
    "\t": 0,
    "\r": 0,
    
    # Markup structural markers (0 symbols - they're metadata)
    "{": 0,
    "}": 0,
    "[": 0,
    "]": 0,
    "#": 0,  # Code mode prefix
    "$": 0,  # Math mode marker
    "@": 0,  # Reference
    "<": 0,  # Label start
    ">": 0,  # Label end
    
    # Emphasis markers in markup (0 symbols - they're formatting)
    "*": 0,  # Strong emphasis
    "_": 0,  # Emphasis
    "`": 0,  # Raw/code
    
    # Math-specific (0 symbols)
    "^": 0,  # Superscript
    "&": 0,  # Alignment point
    
    # Smart quotes and special punctuation (still count as symbols)
    "'": 1,
    '"': 1,
    "—": 1,  # Em dash (symbol shorthand)
    "–": 1,  # En dash
}

