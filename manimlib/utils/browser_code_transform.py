from __future__ import annotations

import ast

SYNC_CREATE_MOBJECTS = {'Text', 'MarkupText', 'Code'}
ASYNC_CREATE_MOBJECTS = {'Typst', 'TypstText', 'Tex', 'TexText', 'DecimalNumber', 'Integer', 'VideoMobject', 'Sprite'}


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
            and node.func.value.id in (ASYNC_CREATE_MOBJECTS | SYNC_CREATE_MOBJECTS)
        ):
            self.changed = True
            return ast.copy_location(ast.Await(node), node)

        if not self.in_await and name in SYNC_CREATE_MOBJECTS | ASYNC_CREATE_MOBJECTS:
            node.func = ast.Attribute(ast.Name(id=name, ctx=ast.Load()), 'create', ast.Load())
            self.changed = True
            return ast.copy_location(ast.Await(node), node) if name in (ASYNC_CREATE_MOBJECTS | SYNC_CREATE_MOBJECTS) else node
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


__all__ = ['SYNC_CREATE_MOBJECTS', 'ASYNC_CREATE_MOBJECTS', 'transform_browser_source']
