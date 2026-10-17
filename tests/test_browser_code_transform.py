import ast
import asyncio
import inspect
import unittest

from manimlib.utils.browser_code_transform import transform_browser_source


class BrowserCodeTransformTests(unittest.TestCase):
    def test_key_listener_callback_transformed_to_async(self):
        source = """
class Handler:
    def __init__(self, catcher):
        catcher.add_key_press_listner(self._key)

    def _key(self, mob, data):
        return False
"""
        transformed = transform_browser_source(source)
        self.assertIn("async def _key(self, mob, data):", transformed)

    def test_variable_call_in_async_func_bare_expr(self):
        source = """
class Handler:
    def __init__(self, catcher):
        catcher.add_key_press_listner(self._key)

    def _key(self, mob, data):
        fn = self.keys.get('r')
        if fn is None:
            return None
        fn()
        return False
"""
        transformed = transform_browser_source(source)
        self.assertIn("async def _key(self, mob, data):", transformed)
        self.assertIn("_res = fn()", transformed)
        self.assertIn("if inspect.iscoroutine(_res):", transformed)
        self.assertIn("_res = await _res", transformed)
        self.assertIn("import inspect", transformed)

    def test_variable_call_in_async_func_assignment(self):
        source = """
async def run_callback(callback):
    result = callback()
    return result
"""
        transformed = transform_browser_source(source)
        self.assertIn("result = callback()", transformed)
        self.assertIn("if inspect.iscoroutine(result):", transformed)
        self.assertIn("result = await result", transformed)

    def test_variable_call_in_async_func_return(self):
        source = """
async def run_callback(callback):
    return callback()
"""
        transformed = transform_browser_source(source)
        self.assertIn("_res = callback()", transformed)
        self.assertIn("if inspect.iscoroutine(_res):", transformed)
        self.assertIn("_res = await _res", transformed)
        self.assertIn("return _res", transformed)

    def test_builtins_and_methods_not_treated_as_variable_calls(self):
        source = """
async def handler(data):
    ch = chr(data['symbol'])
    length = len(data)
    print(ch, length)
    items = list(range(10))
    return items
"""
        transformed = transform_browser_source(source)
        # Builtins should not be wrapped with inspect checks
        self.assertNotIn("iscoroutine", transformed)
        self.assertIn("chr(data['symbol'])", transformed)
        self.assertIn("len(data)", transformed)
        self.assertIn("print(ch, length)", transformed)

    def test_runtime_execution_with_sync_and_async_callbacks(self):
        source = """
class Handler:
    def __init__(self, catcher):
        self.keys = {}
        catcher.add_key_press_listner(self._key)

    def _key(self, mob, data):
        fn = self.keys.get(data['char'])
        if fn is None:
            return None
        fn()
        return False
"""
        transformed = transform_browser_source(source)
        scope = {}
        exec(transformed, scope)
        HandlerClass = scope["Handler"]

        class FakeCatcher:
            def add_key_press_listner(self, cb):
                self.cb = cb

        catcher = FakeCatcher()
        handler = HandlerClass(catcher)

        # 1. Test synchronous callback
        sync_called = False

        def sync_fn():
            nonlocal sync_called
            sync_called = True

        handler.keys["s"] = sync_fn
        res_sync = asyncio.run(catcher.cb(None, {"char": "s"}))
        self.assertTrue(sync_called)
        self.assertEqual(res_sync, False)

        # 2. Test asynchronous callback (coroutine function)
        async_called = False

        async def async_fn():
            nonlocal async_called
            await asyncio.sleep(0.01)
            async_called = True

        handler.keys["a"] = async_fn
        res_async = asyncio.run(catcher.cb(None, {"char": "a"}))
        self.assertTrue(async_called)
        self.assertEqual(res_async, False)

    def test_red_neuronal_artifact_transformation(self):
        with open("mia/artifacts/red_neuronal.py") as f:
            src = f.read()

        transformed = transform_browser_source(src)
        compile(transformed, "<string>", "exec")

        self.assertIn("async def _key(self, mob, data):", transformed)
        self.assertIn("_res = fn()", transformed)
        self.assertIn("if inspect.iscoroutine(_res):", transformed)
        self.assertIn("_res = await _res", transformed)
        self.assertIn("import inspect", transformed)
        self.assertIn("async def setup(self):", transformed)
        self.assertIn("await super().setup()", transformed)
        self.assertIn("v.reset(notify=False)", transformed)
        self.assertNotIn("await v.reset", transformed)

    def test_scene_setup_transformed_to_async(self):
        source = """
class MyScene(Scene):
    def setup(self):
        super().setup()
        self.val = 42

    def construct(self):
        self.wait(1)
"""
        transformed = transform_browser_source(source)
        self.assertIn("async def setup(self):", transformed)
        self.assertIn("await super().setup()", transformed)
        self.assertIn("async def construct(self):", transformed)

    def test_non_scene_sync_methods_not_awaited_when_scene_has_same_name(self):
        source = """
class LiveValue:
    def reset(self, notify=True):
        return self

class MyScene(Scene):
    def reset(self):
        for v in self.lives:
            v.reset(notify=False)
        self.wait(1)
"""
        transformed = transform_browser_source(source)
        self.assertIn("def reset(self, notify=True):", transformed)
        self.assertIn("async def reset(self):", transformed)
        self.assertIn("v.reset(notify=False)", transformed)
        self.assertNotIn("await v.reset", transformed)

    def test_runtime_execution_livevalue_reset_no_cant_be_awaited_error(self):
        source = """
class LiveValue:
    def __init__(self, val):
        self.initial = val
        self.val = val

    def reset(self, notify=True):
        self.val = self.initial
        return self

class DummyScene(Scene):
    def __init__(self):
        self.lives = [LiveValue(10), LiveValue(20)]
        self.waited = False

    async def wait_async(self, t):
        self.waited = True

    def reset(self):
        for v in self.lives:
            v.reset(notify=False)
        self.wait(1)
"""
        transformed = transform_browser_source(source)
        scope = {"Scene": object}
        exec(transformed, scope)
        DummySceneClass = scope["DummyScene"]

        scene = DummySceneClass()
        scene.lives[0].val = 99
        scene.lives[1].val = 88

        # Calling reset should succeed and NOT raise "LiveValue object can't be awaited"
        asyncio.run(scene.reset())

        self.assertEqual(scene.lives[0].val, 10)
        self.assertEqual(scene.lives[1].val, 20)
        self.assertTrue(scene.waited)


if __name__ == "__main__":
    unittest.main()

