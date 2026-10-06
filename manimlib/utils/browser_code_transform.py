from __future__ import annotations

import ast
import re

SYNC_CREATE_MOBJECTS = {'Text', 'MarkupText', 'Code'}
ASYNC_CREATE_MOBJECTS = {'Typst', 'TypstText', 'DecimalNumber', 'Integer'}
TEX_NAMES = {'Tex', 'TexText'}


def latex_to_typst(source: str) -> str:
    replacements = {
        r'\\cdot': 'dot', r'\\times': 'times', r'\\pm': 'plus.minus',
        r'\\mp': 'minus.plus', r'\\leq': '<=', r'\\geq': '>=',
        r'\\neq': '!=', r'\\to': '->', r'\\infty': 'infinity',
        r'\\pi': 'pi', r'\\theta': 'theta', r'\\alpha': 'alpha',
        r'\\beta': 'beta', r'\\gamma': 'gamma', r'\\delta': 'delta',
        r'\\Delta': 'Delta', r'\\Sigma': 'Sigma', r'\\Omega': 'Omega',
        r'\\sum': 'sum', r'\\prod': 'product', r'\\int': 'integral',
    }
    result = source
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement, result)
    result = re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'frac(\1, \2)', result)
    result = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'sqrt(\1)', result)
    result = re.sub(r'\\text\s*\{([^{}]*)\}', r'#text[\1]', result)
    result = re.sub(r'\\mathrm\s*\{([^{}]*)\}', r'\1', result)
    result = re.sub(r'\\mathbf\s*\{([^{}]*)\}', r'bold(\1)', result)
    result = re.sub(r'\\mathbb\s*\{([^{}]*)\}', r'bb(\1)', result)
    result = result.replace(r'\left', '').replace(r'\right', '')
    result = result.replace(r'\,', ' ').replace(r'\;', ' ').replace(r'\!', '')
    result = re.sub(r'\^\{([^{}]*)\}', r'^\1', result)
    result = re.sub(r'_\{([^{}]*)\}', r'_\1', result)
    return result


def _is_true_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


class _BrowserTransformer(ast.NodeTransformer):
    def __init__(self):
        self.changed = False
        self.function_nodes = {}
        self.function_calls = {}
        self.async_functions = set()

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
            def visit_Call(self, call):
                call = self.generic_visit(call)
                if isinstance(call.func, ast.Name) and call.func.id in async_names and call.func.id != 'construct':
                    return ast.copy_location(ast.Await(call), call)
                return call
        for function in self.function_nodes.values():
            function.body = AwaitCalls().visit(function.body)

    def visit_ClassDef(self, node):
        node = self.generic_visit(node)
        if any((isinstance(base, ast.Name) and base.id == 'Scene') or (isinstance(base, ast.Attribute) and base.attr == 'Scene') for base in node.bases):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == 'construct':
                    child.__class__ = ast.AsyncFunctionDef
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name in SYNC_CREATE_MOBJECTS | ASYNC_CREATE_MOBJECTS:
            node.func = ast.Attribute(ast.Name(id=name, ctx=ast.Load()), 'create', ast.Load())
            self.changed = True
            return ast.copy_location(ast.Await(node), node) if name in ASYNC_CREATE_MOBJECTS else node
        if name in TEX_NAMES:
            node.func = ast.Attribute(
                value=ast.Name(id='Typst', ctx=ast.Load()),
                attr='create',
                ctx=ast.Load(),
            )
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                node.args[0].value = latex_to_typst(node.args[0].value)
            self.changed = True
            return ast.copy_location(ast.Await(node), node)
        if isinstance(node.func, ast.Attribute):
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
            if node.func.attr == 'add_coordinate_labels':
                node.func.attr = 'add_coordinate_labels_async'
                self.changed = True
                return ast.copy_location(ast.Await(node), node)
            if node.func.attr == 'add_axis_labels':
                node.func.attr = 'add_axis_labels_async'
                self.changed = True
                return ast.copy_location(ast.Await(node), node)
        if name == 'NumberLine':
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
