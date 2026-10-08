from __future__ import annotations

import ast
import re

SYNC_CREATE_MOBJECTS = {'Text', 'MarkupText', 'Code'}
ASYNC_CREATE_MOBJECTS = {'Typst', 'TypstText', 'DecimalNumber', 'Integer', 'VideoMobject', 'Sprite'}
TEX_NAMES = {'Tex', 'TexText'}


def latex_to_typst(source: str) -> str:
    protected_text = []

    def protect_text(match: re.Match) -> str:
        protected_text.append(match.group(0))
        return f'¤{len(protected_text) - 1}¤'

    replacements = {
        r'\\cdot': 'dot', r'\\times': 'times', r'\\pm': 'plus.minus',
        r'\\mp': 'minus.plus', r'\\leq': '<=', r'\\le': '<=',
        r'\\geq': '>=', r'\\ge': '>=', r'\\neq': '!=', r'\\ne': '!=',
        r'\\to': '->', r'\\rightarrow': '->',
        r'\\infty': 'infinity', r'\\pi': 'pi', r'\\theta': 'theta',
        r'\\alpha': 'alpha', r'\\beta': 'beta', r'\\gamma': 'gamma',
        r'\\delta': 'delta', r'\\epsilon': 'epsilon',
        r'\\varepsilon': 'varepsilon', r'\\phi': 'phi',
        r'\\varphi': 'varphi', r'\\psi': 'psi', r'\\lambda': 'lambda',
        r'\\mu': 'mu', r'\\nu': 'nu', r'\\xi': 'xi', r'\\rho': 'rho',
        r'\\sigma': 'sigma', r'\\tau': 'tau', r'\\upsilon': 'upsilon',
        r'\\chi': 'chi', r'\\omega': 'omega',
        r'\\Delta': 'Delta', r'\\Gamma': 'Gamma', r'\\Lambda': 'Lambda',
        r'\\Xi': 'Xi', r'\\Pi': 'Pi', r'\\Sigma': 'Sigma',
        r'\\Theta': 'Theta', r'\\Upsilon': 'Upsilon', r'\\Phi': 'Phi',
        r'\\Psi': 'Psi', r'\\Omega': 'Omega',
        r'\\sum': 'sum', r'\\prod': 'product', r'\\int': 'integral',
        r'\\sin': 'sin', r'\\cos': 'cos', r'\\tan': 'tan',
        r'\\cot': 'cot', r'\\sec': 'sec', r'\\csc': 'csc',
        r'\\arcsin': 'arcsin', r'\\arccos': 'arccos', r'\\arctan': 'arctan',
        r'\\sinh': 'sinh', r'\\cosh': 'cosh', r'\\tanh': 'tanh',
        r'\\log': 'log', r'\\ln': 'ln', r'\\exp': 'exp', r'\\lim': 'lim',
        r'\\max': 'max', r'\\min': 'min', r'\\mod': 'mod',
    }

    # La entrada puede llegar con los backslashes escapados dos veces
    # (por ejemplo, ``\\\\frac`` en vez de ``\\frac``), especialmente cuando
    # el código pasó por otra capa de serialización antes de llegar al AST.
    # Normalizamos únicamente las secuencias de backslashes que introducen
    # comandos LaTeX; ``\\\\`` usado como salto de línea de TeX se conserva.
    result = re.sub(r'\\\\+(?=[A-Za-z])', r'\\', source)

    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement, result)

    # TeX groups may be nested, so do not use a single non-nested regex for
    # commands such as frac/sqrt/mathbf.
    def replace_braced_command(result: str, command: str, replacement) -> str:
        pattern = re.compile(r'\\' + re.escape(command) + r'\s*\{')
        while True:
            match = pattern.search(result)
            if match is None:
                return result

            start = match.start()
            brace_start = match.end() - 1
            depth = 0
            end = None
            for index in range(brace_start, len(result)):
                char = result[index]
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end = index
                        break
            if end is None:
                return result

            argument = result[brace_start + 1:end]
            result = result[:start] + replacement(argument) + result[end + 1:]

    def read_group(value: str, start: int):
        if start >= len(value) or value[start] != '{':
            return None, None
        depth = 0
        for index in range(start, len(value)):
            if value[index] == '{':
                depth += 1
            elif value[index] == '}':
                depth -= 1
                if depth == 0:
                    return value[start + 1:index], index + 1
        return None, None

    # Handle nested \frac arguments correctly.
    while True:
        match = re.search(r'\\frac\s*\{', result)
        if match is None:
            break

        start = match.start()
        numerator_start = match.end() - 1
        numerator, after_numerator = read_group(result, numerator_start)
        if after_numerator is None:
            break

        denominator_match = re.match(r'\s*\{', result[after_numerator:])
        if denominator_match is None:
            break

        denominator_start = after_numerator + denominator_match.end() - 1
        denominator, after_denominator = read_group(result, denominator_start)
        if after_denominator is None:
            break

        result = (
            result[:start]
            + f'frac({numerator}, {denominator})'
            + result[after_denominator:]
        )

    result = replace_braced_command(
        result, 'sqrt', lambda argument: f'sqrt({argument})'
    )
    result = replace_braced_command(
        result, 'mathrm', lambda argument: argument
    )
    result = replace_braced_command(
        result, 'mathbf', lambda argument: f'bold({argument})'
    )
    result = replace_braced_command(
        result, 'mathbb', lambda argument: f'bb({argument})'
    )

    result = result.replace(r'\left', '').replace(r'\right', '')
    result = result.replace(r'\,', ' ').replace(r'\;', ' ').replace(r'\!', '')

    # TeX's infix \over creates a fraction from the surrounding expression.
    # This handles the common form a \over b, including occurrences inside
    # braces after the outer group has been extracted.
    def replace_over(value: str) -> str:
        depth = 0
        for index, char in enumerate(value):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
            elif depth == 0 and value.startswith(r'\over', index):
                left = value[:index].rstrip()
                right = value[index + len(r'\over'):].lstrip()
                if left and right:
                    return f'frac({left}, {right})'
        return value

    result = replace_over(result)

    result = re.sub(r'\^\{([^{}]*)\}', r'^\1', result)
    result = re.sub(r'_\{([^{}]*)\}', r'_\1', result)

    # At this point no supported TeX command should retain its leading slash.
    # Preserve escaped literal characters as the corresponding character.
    result = re.sub(r'\\([A-Za-z]+)', r'\1', result)

    typst_words = {
        'dot', 'times', 'plus', 'minus', 'infinity',
        'pi', 'theta', 'alpha', 'beta', 'gamma', 'delta',
        'epsilon', 'varepsilon', 'phi', 'varphi', 'psi', 'lambda',
        'mu', 'nu', 'xi', 'rho', 'sigma', 'tau', 'upsilon', 'chi', 'omega',
        'Delta', 'Gamma', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Theta',
        'Upsilon', 'Phi', 'Psi', 'Omega',
        'sum', 'product', 'integral', 'frac', 'sqrt', 'bold', 'bb',
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
        'log', 'ln', 'exp', 'lim', 'max', 'min', 'mod', 'gcd',
    }

    def split_math_identifiers(match: re.Match) -> str:
        word = match.group(0)
        if word in typst_words:
            return word
        return ' '.join(word)

    result = re.sub(r'#text\[[^]]*\]', protect_text, result)
    result = re.sub(r'[A-Za-z]+', split_math_identifiers, result)

    for index, text_block in enumerate(protected_text):
        result = result.replace(f'¤{index}¤', text_block)
    return result


def _is_true_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


class _BrowserTransformer(ast.NodeTransformer):
    def __init__(self):
        self.changed = False
        self.function_nodes = {}
        self.function_calls = {}
        self.async_functions = set()
        self.in_await = False

    def visit_Module(self, node):
        self._collect_functions(node)
        node = self.generic_visit(node)
        self._propagate_async_functions()
        return node

    def _collect_functions(self, node):
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.function_nodes[child.name] = child
                self.function_calls[child.name] = {
                    call.func.id for call in ast.walk(child)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                }

    def _propagate_async_functions(self):
        changed = True
        while changed:
            changed = False
            for name, function in self.function_nodes.items():
                if name not in self.async_functions and any(isinstance(n, ast.Await) for n in ast.walk(function)):
                    self.async_functions.add(name)
                    changed = True
            for name, calls in self.function_calls.items():
                if name not in self.async_functions and calls & self.async_functions:
                    self.async_functions.add(name)
                    changed = True
        for name in self.async_functions:
            function = self.function_nodes[name]
            if isinstance(function, ast.FunctionDef):
                function.__class__ = ast.AsyncFunctionDef
        async_names = self.async_functions
        class AwaitCalls(ast.NodeTransformer):
            def visit_Await(self, node):
                node.value = self.generic_visit(node.value)
                return node

            def visit_Call(self, call):
                call = self.generic_visit(call)
                if isinstance(call.func, ast.Name) and call.func.id in async_names and call.func.id != 'construct':
                    return ast.copy_location(ast.Await(call), call)
                return call
        for function in self.function_nodes.values():
            # NodeTransformer.visit() accepts an AST node, not a statement list.
            # Visit each statement and preserve the FunctionDef.body list.
            transformer = AwaitCalls()
            new_body = []
            for statement in function.body:
                transformed = transformer.visit(statement)
                if isinstance(transformed, list):
                    new_body.extend(transformed)
                else:
                    new_body.append(transformed)
            function.body = new_body

    def visit_Await(self, node):
        previous = self.in_await
        self.in_await = True
        node.value = self.visit(node.value)
        self.in_await = previous
        return node

    def visit_ClassDef(self, node):
        node = self.generic_visit(node)

        # Browser scenes must always have an async construct().  The generated
        # code replaces synchronous Scene.play()/wait() calls with their async
        # counterparts, so construct itself has to be awaitable as well.
        #
        # Do not restrict this to a class whose base is literally named
        # Scene.  Manim has many Scene subclasses (ThreeDScene,
        # MovingCameraScene, InteractiveScene, and user-defined Scene
        # subclasses), and their construct() methods need the same treatment.
        is_scene_class = any(
            (
                isinstance(base, ast.Name)
                and (base.id == 'Scene' or base.id.endswith('Scene'))
            )
            or (
                isinstance(base, ast.Attribute)
                and (base.attr == 'Scene' or base.attr.endswith('Scene'))
            )
            for base in node.bases
        )

        if is_scene_class:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == 'construct':
                    child.__class__ = ast.AsyncFunctionDef

        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if (
            not self.in_await
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'create'
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ASYNC_CREATE_MOBJECTS
        ):
            self.changed = True
            return ast.copy_location(ast.Await(node), node)

        if not self.in_await and name in SYNC_CREATE_MOBJECTS | ASYNC_CREATE_MOBJECTS:
            node.func = ast.Attribute(ast.Name(id=name, ctx=ast.Load()), 'create', ast.Load())
            self.changed = True
            return ast.copy_location(ast.Await(node), node) if name in ASYNC_CREATE_MOBJECTS else node
        if not self.in_await and name in TEX_NAMES:
            node.func = ast.Attribute(
                value=ast.Name(id='Typst', ctx=ast.Load()),
                attr='create',
                ctx=ast.Load(),
            )
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                node.args[0].value = latex_to_typst(node.args[0].value)
            self.changed = True
            return ast.copy_location(ast.Await(node), node)
        if not self.in_await and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                if node.func.attr == 'play':
                    node.func.attr = 'play_async'
                    self.changed = True
                    return ast.copy_location(ast.Await(node), node)
                if node.func.attr == 'wait':
                    node.func.attr = 'wait_async'
                    self.changed = True
                    return ast.copy_location(ast.Await(node), node)
            if node.func.attr == 'add_numbers':
                node.func.attr = 'add_numbers_async'
                self.changed = True
                return ast.copy_location(ast.Await(node), node)
            if node.func.attr == 'add_coordinates':
                node.func.attr = 'add_coordinates_async'
                self.changed = True
                return ast.copy_location(ast.Await(node), node)
            if node.func.attr == 'add_coordinate_labels':
                node.func.attr = 'add_coordinate_labels_async'
                self.changed = True
                return ast.copy_location(ast.Await(node), node)
            if node.func.attr == 'add_axis_labels':
                node.func.attr = 'add_axis_labels_async'
                self.changed = True
                return ast.copy_location(ast.Await(node), node)
            if node.func.attr == 'get_axis_labels':
                node.func.attr = 'get_axis_labels_async'
                self.changed = True
                return ast.copy_location(ast.Await(node), node)
        if not self.in_await and name == 'NumberLine':
            for keyword in node.keywords:
                if keyword.arg == 'include_numbers' and _is_true_constant(keyword.value):
                    keyword.value = ast.Constant(value=False)
                    helper = ast.Call(ast.Name(id='_browser_add_number_line_numbers', ctx=ast.Load()), [node], [])
                    self.changed = True
                    return ast.copy_location(ast.Await(helper), node)
        return node


def transform_browser_source(source: str) -> str:
    tree = ast.parse(source)
    tree = _BrowserTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

__all__ = ['SYNC_CREATE_MOBJECTS', 'ASYNC_CREATE_MOBJECTS', 'TEX_NAMES', 'latex_to_typst', 'transform_browser_source']
