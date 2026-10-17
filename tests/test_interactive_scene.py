import asyncio
import inspect
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from manimlib.constants import ORIGIN, RIGHT, UP
from manimlib.mobject.geometry import Circle, Square
from manimlib.scene.interactive_scene import InteractiveScene
from manimlib.scene.scene import Scene


class DummyWindow:
    def __init__(self):
        self.pressed_keys = set()
        self.gpu = MagicMock()
        self.gpu.samples = 1
        self.gpu.device = MagicMock()

    def get_size(self):
        return (1920, 1080)

    def is_key_pressed(self, key):
        return key in self.pressed_keys

    def init_for_scene(self, scene):
        pass


class InteractiveSceneAsyncTests(unittest.TestCase):
    def setUp(self):
        self.scene = InteractiveScene(window=DummyWindow(), skip_animations=True)

    def test_setup_native_vs_emscripten(self):
        # Native platform: setup() runs synchronously
        with patch("sys.platform", "linux"):
            res = self.scene.setup()
            self.assertFalse(inspect.isawaitable(res))
            self.assertIsNotNone(self.scene.crosshair)
            self.assertIsNotNone(self.scene.selection)

        # Emscripten platform: setup() returns an awaitable coroutine
        with patch("sys.platform", "emscripten"):
            coro = self.scene.setup()
            self.assertTrue(inspect.isawaitable(coro))
            asyncio.run(coro)
            self.assertIsNotNone(self.scene.crosshair)
            self.assertIsNotNone(self.scene.selection)

    def test_events_return_awaitables_on_emscripten(self):
        with patch("sys.platform", "emscripten"):
            # Setup
            asyncio.run(self.scene.setup_async())

            # Mouse motion
            motion_res = self.scene.on_mouse_motion(np.array([1.0, 2.0, 0.0]), np.array([0.1, 0.1, 0.0]))
            self.assertTrue(inspect.isawaitable(motion_res))
            asyncio.run(motion_res)
            np.testing.assert_allclose(
                self.scene.crosshair.get_center(),
                self.scene.frame.to_fixed_frame_point(np.array([1.0, 2.0, 0.0]))
            )

            # Mouse drag
            drag_res = self.scene.on_mouse_drag(
                np.array([2.0, 3.0, 0.0]), np.array([0.1, 0.1, 0.0]), buttons=1, modifiers=0
            )
            self.assertTrue(inspect.isawaitable(drag_res))
            asyncio.run(drag_res)
            np.testing.assert_allclose(
                self.scene.crosshair.get_center(),
                self.scene.frame.to_fixed_frame_point(np.array([2.0, 3.0, 0.0]))
            )

            # Mouse release
            rel_res = self.scene.on_mouse_release(np.array([2.0, 3.0, 0.0]), button=1, mods=0)
            self.assertTrue(inspect.isawaitable(rel_res))
            asyncio.run(rel_res)

            # Key press (e.g. ord('s') for select)
            kp_res = self.scene.on_key_press(ord("s"), modifiers=0)
            self.assertTrue(inspect.isawaitable(kp_res))
            asyncio.run(kp_res)

            # Key release
            kr_res = self.scene.on_key_release(ord("s"), modifiers=0)
            self.assertTrue(inspect.isawaitable(kr_res))
            asyncio.run(kr_res)

    def test_begin_build_async_awaits_setup(self):
        with patch("sys.platform", "emscripten"):
            scene = InteractiveScene(window=DummyWindow(), skip_animations=True)
            self.assertFalse(hasattr(scene, "selection") and scene.selection is not None)
            asyncio.run(scene._begin_build_async())
            self.assertTrue(hasattr(scene, "selection") and scene.selection is not None)
            self.assertTrue(hasattr(scene, "crosshair") and scene.crosshair is not None)


if __name__ == "__main__":
    unittest.main()
