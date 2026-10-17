from __future__ import annotations

import ast
import copy

SYNC_CREATE_MOBJECTS = {'Text', 'MarkupText', 'Code'}
ASYNC_CREATE_MOBJECTS = {'Typst', 'TypstText', 'Tex', 'TexText', 'DecimalNumber', 'Integer', 'VideoMobject', 'Sprite'}

SYNCHRONOUS_MOBJECT_METHODS = {
    'move_to', 'next_to', 'shift', 'scale', 'rotate', 'align_to', 'to_edge', 'to_corner',
    'set_color', 'set_stroke', 'set_fill', 'add', 'remove', 'become', 'match_x', 'match_y',
    'match_z', 'match_width', 'match_height', 'match_depth', 'set_width', 'set_height',
    'fix_in_frame', 'center', 'arrange', 'add_updater', 'remove_updater', 'update',
    'get_center', 'get_width', 'get_height', 'get_start', 'get_end', 'point_from_proportion'
}


def _is_true_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _browser_schedule_task(coro_or_val):
    import asyncio
    import inspect
    if inspect.isawaitable(coro_or_val):
        return asyncio.create_task(coro_or_val)
    return coro_or_val


class _BrowserTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.changed = False
        self.in_await = False
        self.in_lambda = False
        self.in_init = False
        self.used_schedule_helper = False
        self.used_inspect = False
        self.async_create_mobjects = set(ASYNC_CREATE_MOBJECTS)

    def transform(self, tree: ast.AST) -> ast.AST:
        # Phase 0: Transform custom mobject subclasses (e.g. LiveNumber(DecimalNumber))
        # into browser-safe classes with def __init__(self, **kwargs): pass and async def create(cls, ...)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                is_async_subclass = any(
                    (isinstance(b, ast.Name) and (b.id in self.async_create_mobjects or b.id in ('DecimalNumber', 'Integer'))) or
                    (isinstance(b, ast.Attribute) and (b.attr in self.async_create_mobjects or b.attr in ('DecimalNumber', 'Integer')))
                    for b in node.bases
                )
                if is_async_subclass:
                    self.async_create_mobjects.add(node.name)
                    init_node = next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == '__init__'), None)
                    if init_node:
                        create_args = copy.deepcopy(init_node.args)
                        if create_args.args:
                            create_args.args[0].arg = 'cls'

                        create_body = []
                        if create_args.kwarg:
                            create_body.append(ast.parse(f'self = cls(**{create_args.kwarg.arg})').body[0])
                        else:
                            create_body.append(ast.parse('self = cls()').body[0])

                        for stmt in init_node.body:
                            is_super_init = False
                            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                                call = stmt.value
                                if isinstance(call.func, ast.Attribute) and call.func.attr == '__init__':
                                    if (
                                        (isinstance(call.func.value, ast.Call) and isinstance(call.func.value.func, ast.Name) and call.func.value.func.id == 'super') or
                                        (isinstance(call.func.value, ast.Name) and call.func.value.id in ('super', 'DecimalNumber', 'Integer'))
                                    ):
                                        is_super_init = True
                                        call.func.value = ast.Call(
                                            func=ast.Name(id='super', ctx=ast.Load()),
                                            args=[ast.Name(id='cls', ctx=ast.Load()), ast.Name(id='self', ctx=ast.Load())],
                                            keywords=[]
                                        )
                                        call.keywords = [kw for kw in call.keywords if kw.arg != '_skip_text_render']
                                        call.keywords.append(ast.keyword(arg='_skip_text_render', value=ast.Constant(value=True)))
                                        create_body.append(stmt)

                                        first_arg = call.args[0] if call.args else ast.Constant(value=0)
                                        cache_logic = ast.parse('''
try:
    font_key = (self.get_font_size(), tuple(sorted(((k, str(v)) for k, v in self.text_config.items()))))
except Exception:
    font_key = self.get_font_size()
if not hasattr(cls, '_glyph_cache_by_font'):
    cls._glyph_cache_by_font = {}
if font_key not in cls._glyph_cache_by_font:
    from manimlib.mobject.svg.typst_mobject import Typst
    from manimlib.mobject.types.vectorized_mobject import VGroup, VMobject
    chars = '0123456789.-+–, i'
    t = await Typst.create(chars, isolate=tuple(chars), font_size=self.get_font_size(), **self.text_config)
    cache = {}
    for idx, ch in enumerate(chars):
        indices = t.find_submobject_indices_by_span((idx, idx + 1))
        if indices:
            mob = VGroup(*(t.submobjects[i].copy() for i in indices))
            mob.font_size = self.get_font_size()
            cache[ch] = mob
    blank = VMobject()
    blank.font_size = self.get_font_size()
    cache[' '] = blank
    cls._glyph_cache_by_font[font_key] = cache
self._glyph_cache = cls._glyph_cache_by_font[font_key]
''').body
                                        create_body.extend(cache_logic)
                                        render = ast.Expr(value=ast.Call(
                                            func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='set_submobjects_from_number', ctx=ast.Load()),
                                            args=[first_arg],
                                            keywords=[]
                                        ))
                                        draw_fills = ast.Expr(value=ast.Call(
                                            func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='draw_fills_together_if_disjoint', ctx=ast.Load()),
                                            args=[],
                                            keywords=[]
                                        ))
                                        create_body.append(render)
                                        create_body.append(draw_fills)
                            if not is_super_init:
                                create_body.append(stmt)

                        create_body.append(ast.Return(value=ast.Name(id='self', ctx=ast.Load())))

                        create_def = ast.AsyncFunctionDef(
                            name='create',
                            args=create_args,
                            body=create_body,
                            decorator_list=[ast.Name(id='classmethod', ctx=ast.Load())],
                            returns=None
                        )

                        init_node.args = ast.arguments(
                            posonlyargs=[],
                            args=[ast.arg(arg='self')],
                            vararg=None,
                            kwonlyargs=[],
                            kw_defaults=[],
                            kwarg=ast.arg(arg='kwargs'),
                            defaults=[]
                        )
                        init_node.body = [ast.Pass()]
                        node.body.append(create_def)

                        char_to_mob_node = ast.parse('''
def char_to_mob(self, char: str):
    cache = getattr(self, '_glyph_cache', None)
    if cache and char in cache:
        return cache[char].copy()
    return super().char_to_mob(char)
''').body[0]
                        if not any(isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == 'char_to_mob' for m in node.body):
                            node.body.append(char_to_mob_node)

        # Phase 0.5: Transform functions registered via add_key_press_listner (and listener methods) into async
        key_listener_methods = {
            'add_key_press_listner', 'add_key_press_listener',
            'add_key_release_listner', 'add_key_release_listener',
        }

        for cls_node in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for method in cls_node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for sub in ast.walk(method):
                        if isinstance(sub, ast.Call):
                            attr = None
                            if isinstance(sub.func, ast.Attribute):
                                attr = sub.func.attr
                            elif isinstance(sub.func, ast.Name):
                                attr = sub.func.id
                            if attr in key_listener_methods:
                                callback_arg = sub.args[0] if sub.args else next((k.value for k in sub.keywords if k.arg in ('callback', 'fn', 'func')), None)
                                if callback_arg is not None:
                                    if isinstance(callback_arg, ast.Attribute) and isinstance(callback_arg.value, ast.Name) and callback_arg.value.id == 'self':
                                        target = next((m for m in cls_node.body if isinstance(m, ast.FunctionDef) and m.name == callback_arg.attr), None)
                                        if target:
                                            target.__class__ = ast.AsyncFunctionDef
                                    elif isinstance(callback_arg, ast.Name):
                                        target = next((m for m in cls_node.body if isinstance(m, ast.FunctionDef) and m.name == callback_arg.id), None)
                                        if target:
                                            target.__class__ = ast.AsyncFunctionDef

        for sub in ast.walk(tree):
            if isinstance(sub, ast.Call):
                attr = None
                if isinstance(sub.func, ast.Attribute):
                    attr = sub.func.attr
                elif isinstance(sub.func, ast.Name):
                    attr = sub.func.id
                if attr in key_listener_methods:
                    callback_arg = sub.args[0] if sub.args else next((k.value for k in sub.keywords if k.arg in ('callback', 'fn', 'func')), None)
                    if callback_arg is not None:
                        target_name = None
                        if isinstance(callback_arg, ast.Attribute):
                            target_name = callback_arg.attr
                        elif isinstance(callback_arg, ast.Name):
                            target_name = callback_arg.id
                        if target_name:
                            for fn_node in ast.walk(tree):
                                if isinstance(fn_node, ast.FunctionDef) and fn_node.name == target_name:
                                    fn_node.__class__ = ast.AsyncFunctionDef

        # Phase 1: Transform built-in async primitives (Text.create, play_async, wait_async, etc.)
        tree = self.visit(tree)

        # Phase 2: Iteratively find all functions that contain `await` and propagate to their callers
        # until a fixed point is reached.
        while True:
            all_funcs = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    all_funcs.append(node)

            async_func_names = set()
            newly_marked_async = False

            for func in all_funcs:
                if func.name == '__init__':
                    continue

                has_direct_await = False
                for stmt in func.body:
                    for child in ast.walk(stmt):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                            continue
                        if isinstance(child, ast.Await):
                            has_direct_await = True
                            break
                    if has_direct_await:
                        break

                if has_direct_await or isinstance(func, ast.AsyncFunctionDef):
                    if isinstance(func, ast.FunctionDef):
                        func.__class__ = ast.AsyncFunctionDef
                        newly_marked_async = True
                    async_func_names.add(func.name)

            # Propagate to variable aliases (e.g. _goldenlib_mathtex = mathtex)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    if isinstance(node.value, ast.Name) and node.value.id in async_func_names:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                async_func_names.add(target.id)

            await_added = False
            transformer = self

            class AwaitCallTransformer(ast.NodeTransformer):
                def __init__(self, async_names):
                    super().__init__()
                    self.async_names = async_names
                    self.await_added = False
                    self.in_await = False
                    self.in_lambda = False
                    self.in_init = False

                def visit_FunctionDef(self, node):
                    prev_init = self.in_init
                    if node.name == '__init__':
                        self.in_init = True
                    node = self.generic_visit(node)
                    self.in_init = prev_init
                    return node

                def visit_AsyncFunctionDef(self, node):
                    prev_init = self.in_init
                    if node.name == '__init__':
                        self.in_init = True
                    node = self.generic_visit(node)
                    self.in_init = prev_init
                    return node

                def visit_Lambda(self, node):
                    prev_lambda = self.in_lambda
                    self.in_lambda = True
                    node = self.generic_visit(node)
                    self.in_lambda = prev_lambda
                    return node

                def visit_Await(self, node):
                    prev = self.in_await
                    self.in_await = True
                    node.value = self.visit(node.value)
                    self.in_await = prev
                    return node

                def visit_Call(self, node):
                    prev = self.in_await
                    self.in_await = False
                    node = self.generic_visit(node)
                    self.in_await = prev

                    if self.in_await:
                        return node

                    func_name = None
                    is_attr = False
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                        is_attr = True

                    if is_attr and func_name in SYNCHRONOUS_MOBJECT_METHODS:
                        return node

                    if func_name == '__init__':
                        return node

                    if func_name in self.async_names and func_name != 'construct':
                        if self.in_init:
                            return node
                        if self.in_lambda:
                            transformer.used_schedule_helper = True
                            return ast.copy_location(
                                ast.Call(
                                    func=ast.Name(id='_browser_schedule_task', ctx=ast.Load()),
                                    args=[node],
                                    keywords=[]
                                ),
                                node
                            )
                        else:
                            self.await_added = True
                            return ast.copy_location(ast.Await(value=node), node)

                    return node

                def visit_Starred(self, node):
                    node = self.generic_visit(node)
                    if isinstance(node.value, ast.GeneratorExp):
                        if any(isinstance(c, ast.Await) for c in ast.walk(node.value)):
                            node.value = ast.ListComp(
                                elt=node.value.elt,
                                generators=node.value.generators
                            )
                    return node

            caller_transformer = AwaitCallTransformer(async_func_names)
            for func in all_funcs:
                prev_init = caller_transformer.in_init
                if func.name == '__init__':
                    caller_transformer.in_init = True
                new_body = []
                for stmt in func.body:
                    res = caller_transformer.visit(stmt)
                    if isinstance(res, list):
                        new_body.extend(res)
                    else:
                        new_body.append(res)
                func.body = new_body
                caller_transformer.in_init = prev_init

            if caller_transformer.await_added:
                await_added = True

            if not newly_marked_async and not await_added:
                break

        # Ensure any Scene construct method is AsyncFunctionDef
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                is_scene = any(
                    (isinstance(b, ast.Name) and (b.id == 'Scene' or b.id.endswith('Scene'))) or
                    (isinstance(b, ast.Attribute) and (b.attr == 'Scene' or b.attr.endswith('Scene')))
                    for b in node.bases
                )
                if is_scene:
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef) and child.name == 'construct':
                            child.__class__ = ast.AsyncFunctionDef

                existing_methods = {
                    m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                for base_name in ('play', 'wait', 'hold_loop'):
                    async_name = f'{base_name}_async'
                    if base_name in existing_methods and async_name not in existing_methods:
                        assign_node = ast.Assign(
                            targets=[ast.Name(id=async_name, ctx=ast.Store())],
                            value=ast.Name(id=base_name, ctx=ast.Load())
                        )
                        ast.copy_location(assign_node, node)
                        node.body.append(assign_node)

        # Phase 2.5: Transform calls to variable functions in async functions
        import builtins
        try:
            import manimlib
            _manim_symbols = set(dir(manimlib))
        except Exception:
            _manim_symbols = set()

        _builtin_names = set(dir(builtins))
        top_level_names = {
            n.name for n in getattr(tree, 'body', [])
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        imported_names = set()
        for n in getattr(tree, 'body', []):
            if isinstance(n, ast.Import):
                for alias in n.names:
                    imported_names.add(alias.asname or alias.name)
            elif isinstance(n, ast.ImportFrom):
                if n.module == 'manimlib' and any(alias.name == '*' for alias in n.names):
                    imported_names.update(_manim_symbols)
                for alias in n.names:
                    imported_names.add(alias.asname or alias.name)

        known_globals = (
            _builtin_names
            | _manim_symbols
            | top_level_names
            | imported_names
            | {'self', 'super', 'cls'}
            | self.async_create_mobjects
            | SYNC_CREATE_MOBJECTS
        )

        def is_variable_call(call_node: ast.Call) -> bool:
            if not isinstance(call_node.func, ast.Name):
                return False
            if call_node.func.id in known_globals:
                return False
            return True

        def transform_statement_list(stmts: list[ast.stmt]) -> list[ast.stmt]:
            new_stmts = []
            for stmt in stmts:
                if isinstance(stmt, ast.If):
                    stmt.body = transform_statement_list(stmt.body)
                    stmt.orelse = transform_statement_list(stmt.orelse)
                    new_stmts.append(stmt)
                elif isinstance(stmt, (ast.For, ast.While)):
                    stmt.body = transform_statement_list(stmt.body)
                    stmt.orelse = transform_statement_list(stmt.orelse)
                    new_stmts.append(stmt)
                elif isinstance(stmt, ast.Try):
                    stmt.body = transform_statement_list(stmt.body)
                    for h in stmt.handlers:
                        h.body = transform_statement_list(h.body)
                    stmt.orelse = transform_statement_list(stmt.orelse)
                    stmt.finalbody = transform_statement_list(stmt.finalbody)
                    new_stmts.append(stmt)
                elif isinstance(stmt, ast.With):
                    stmt.body = transform_statement_list(stmt.body)
                    new_stmts.append(stmt)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    new_stmts.append(stmt)
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and is_variable_call(stmt.value):
                    self.used_inspect = True
                    temp_var = '_res'
                    assign_node = ast.Assign(
                        targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                        value=stmt.value
                    )
                    ast.copy_location(assign_node, stmt)
                    check_node = ast.If(
                        test=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id='inspect', ctx=ast.Load()),
                                attr='iscoroutine',
                                ctx=ast.Load()
                            ),
                            args=[ast.Name(id=temp_var, ctx=ast.Load())],
                            keywords=[]
                        ),
                        body=[
                            ast.Assign(
                                targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                                value=ast.Await(value=ast.Name(id=temp_var, ctx=ast.Load()))
                            )
                        ],
                        orelse=[]
                    )
                    ast.copy_location(check_node, stmt)
                    ast.copy_location(check_node.body[0], stmt)
                    new_stmts.extend([assign_node, check_node])
                elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and is_variable_call(stmt.value):
                    self.used_inspect = True
                    if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                        var_name = stmt.targets[0].id
                        new_stmts.append(stmt)
                        check_node = ast.If(
                            test=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='inspect', ctx=ast.Load()),
                                    attr='iscoroutine',
                                    ctx=ast.Load()
                                ),
                                args=[ast.Name(id=var_name, ctx=ast.Load())],
                                keywords=[]
                            ),
                            body=[
                                ast.Assign(
                                    targets=[ast.Name(id=var_name, ctx=ast.Store())],
                                    value=ast.Await(value=ast.Name(id=var_name, ctx=ast.Load()))
                                )
                            ],
                            orelse=[]
                        )
                        ast.copy_location(check_node, stmt)
                        ast.copy_location(check_node.body[0], stmt)
                        new_stmts.append(check_node)
                    else:
                        temp_var = '_res'
                        assign_node = ast.Assign(
                            targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                            value=stmt.value
                        )
                        ast.copy_location(assign_node, stmt)
                        check_node = ast.If(
                            test=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='inspect', ctx=ast.Load()),
                                    attr='iscoroutine',
                                    ctx=ast.Load()
                                ),
                                args=[ast.Name(id=temp_var, ctx=ast.Load())],
                                keywords=[]
                            ),
                            body=[
                                ast.Assign(
                                    targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                                    value=ast.Await(value=ast.Name(id=temp_var, ctx=ast.Load()))
                                )
                            ],
                            orelse=[]
                        )
                        ast.copy_location(check_node, stmt)
                        ast.copy_location(check_node.body[0], stmt)
                        final_assign = ast.Assign(
                            targets=stmt.targets,
                            value=ast.Name(id=temp_var, ctx=ast.Load())
                        )
                        ast.copy_location(final_assign, stmt)
                        new_stmts.extend([assign_node, check_node, final_assign])
                elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call) and is_variable_call(stmt.value):
                    self.used_inspect = True
                    temp_var = '_res'
                    assign_node = ast.Assign(
                        targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                        value=stmt.value
                    )
                    ast.copy_location(assign_node, stmt)
                    check_node = ast.If(
                        test=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id='inspect', ctx=ast.Load()),
                                attr='iscoroutine',
                                ctx=ast.Load()
                            ),
                            args=[ast.Name(id=temp_var, ctx=ast.Load())],
                            keywords=[]
                        ),
                        body=[
                            ast.Assign(
                                targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                                value=ast.Await(value=ast.Name(id=temp_var, ctx=ast.Load()))
                            )
                        ],
                        orelse=[]
                    )
                    ast.copy_location(check_node, stmt)
                    ast.copy_location(check_node.body[0], stmt)
                    ret_node = ast.Return(value=ast.Name(id=temp_var, ctx=ast.Load()))
                    ast.copy_location(ret_node, stmt)
                    new_stmts.extend([assign_node, check_node, ret_node])
                else:
                    new_stmts.append(stmt)
            return new_stmts

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                node.body = transform_statement_list(node.body)

        if self.used_inspect:
            has_inspect = any(
                isinstance(n, ast.Import) and any(alias.name == 'inspect' for alias in n.names)
                for n in getattr(tree, 'body', [])
            )
            if not has_inspect:
                idx = 0
                while idx < len(tree.body):
                    item = tree.body[idx]
                    if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                        idx += 1
                    elif isinstance(item, ast.ImportFrom) and item.module == '__future__':
                        idx += 1
                    else:
                        break
                tree.body.insert(idx, ast.Import(names=[ast.alias(name='inspect', asname=None)]))

        if self.used_schedule_helper:
            has_helper = any(
                isinstance(n, ast.FunctionDef) and n.name == '_browser_schedule_task'
                for n in getattr(tree, 'body', [])
            )
            if not has_helper:
                helper_code = (
                    "def _browser_schedule_task(coro_or_val):\n"
                    "    import asyncio\n"
                    "    import inspect\n"
                    "    if inspect.isawaitable(coro_or_val):\n"
                    "        return asyncio.create_task(coro_or_val)\n"
                    "    return coro_or_val\n"
                )
                tree.body = ast.parse(helper_code).body + tree.body

        return tree

    def visit_FunctionDef(self, node):
        prev_init = self.in_init
        if node.name == '__init__':
            self.in_init = True
        node = self.generic_visit(node)
        self.in_init = prev_init
        return node

    def visit_AsyncFunctionDef(self, node):
        prev_init = self.in_init
        if node.name == '__init__':
            self.in_init = True
        node = self.generic_visit(node)
        self.in_init = prev_init
        return node

    def visit_Lambda(self, node):
        prev = self.in_lambda
        self.in_lambda = True
        node = self.generic_visit(node)
        self.in_lambda = prev
        return node

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

        if self.in_init:
            return node

        if (
            not self.in_await
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'create'
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in (self.async_create_mobjects | SYNC_CREATE_MOBJECTS)
        ):
            self.changed = True
            if self.in_lambda:
                self.used_schedule_helper = True
                return ast.copy_location(
                    ast.Call(func=ast.Name(id='_browser_schedule_task', ctx=ast.Load()), args=[node], keywords=[]),
                    node
                )
            return ast.copy_location(ast.Await(node), node)

        if not self.in_await and name in (SYNC_CREATE_MOBJECTS | self.async_create_mobjects):
            node.func = ast.Attribute(ast.Name(id=name, ctx=ast.Load()), 'create', ast.Load())
            self.changed = True
            if self.in_lambda:
                self.used_schedule_helper = True
                return ast.copy_location(
                    ast.Call(func=ast.Name(id='_browser_schedule_task', ctx=ast.Load()), args=[node], keywords=[]),
                    node
                )
            return ast.copy_location(ast.Await(node), node)

        if not self.in_await and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in ('play', 'wait'):
                node.func.attr = f'{attr}_async'
                self.changed = True
                if self.in_lambda:
                    self.used_schedule_helper = True
                    return ast.copy_location(
                        ast.Call(func=ast.Name(id='_browser_schedule_task', ctx=ast.Load()), args=[node], keywords=[]),
                        node
                    )
                return ast.copy_location(ast.Await(node), node)

            if attr == 'hold_loop':
                node.func.attr = 'hold_loop_async'
                self.changed = True
                if self.in_lambda:
                    self.used_schedule_helper = True
                    return ast.copy_location(
                        ast.Call(func=ast.Name(id='_browser_schedule_task', ctx=ast.Load()), args=[node], keywords=[]),
                        node
                    )
                return ast.copy_location(ast.Await(node), node)

            if attr in ('add_numbers', 'add_coordinates', 'add_coordinate_labels', 'add_axis_labels', 'get_axis_labels'):
                node.func.attr = f'{attr}_async'
                self.changed = True
                if self.in_lambda:
                    self.used_schedule_helper = True
                    return ast.copy_location(
                        ast.Call(func=ast.Name(id='_browser_schedule_task', ctx=ast.Load()), args=[node], keywords=[]),
                        node
                    )
                return ast.copy_location(ast.Await(node), node)

        if not self.in_await and name == 'NumberLine':
            for keyword in node.keywords:
                if keyword.arg == 'include_numbers' and _is_true_constant(keyword.value):
                    keyword.value = ast.Constant(value=False)
                    helper = ast.Call(ast.Name(id='_browser_add_number_line_numbers', ctx=ast.Load()), [node], [])
                    self.changed = True
                    if self.in_lambda:
                        self.used_schedule_helper = True
                        return ast.copy_location(
                            ast.Call(func=ast.Name(id='_browser_schedule_task', ctx=ast.Load()), args=[helper], keywords=[]),
                            node
                        )
                    return ast.copy_location(ast.Await(helper), node)

        return node

    def visit_Starred(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.GeneratorExp):
            if any(isinstance(c, ast.Await) for c in ast.walk(node.value)):
                node.value = ast.ListComp(
                    elt=node.value.elt,
                    generators=node.value.generators
                )
        return node


def transform_browser_source(source: str) -> str:
    tree = ast.parse(source)
    tree = _BrowserTransformer().transform(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


__all__ = ['SYNC_CREATE_MOBJECTS', 'ASYNC_CREATE_MOBJECTS', '_browser_schedule_task', 'transform_browser_source']
