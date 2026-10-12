"""CPU simulator smoke checks; no model weights, training, or GPU required."""

import importlib.util
import os
from pathlib import Path
import unittest


class EnvironmentSmokeTests(unittest.TestCase):
    def smoke(self, task, action, extra=()):
        required = ("numpy", "PIL", "pygame", "cloudpickle", *extra)
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            message = "Missing CPU smoke dependencies: " + ", ".join(missing)
            if os.environ.get("AGIMA_REQUIRE_ENV_SMOKE") == "1":
                self.fail(message)
            self.skipTest(message)
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import gymnasium as gym
        import numpy as np
        # The vendored Gymnasium initialization sets an audio driver on Linux.
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        root = Path(__file__).resolve().parents[1]
        self.assertTrue(Path(gym.__file__).resolve().is_relative_to(root))
        env = gym.make(task)
        self.addCleanup(env.close)
        env.reset(seed=42)
        simulator = env.unwrapped
        frame = np.asarray(simulator.render()).copy()
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[-1], 3)
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(frame.size, 0)
        state = simulator.get_init_state()
        transition = simulator.step(action)
        after = np.asarray(simulator.render()).copy()
        self.assertEqual(len(transition), 5)
        simulator.reset(init_state=state)
        np.testing.assert_array_equal(simulator.render(), frame)
        replay = simulator.step(action)
        np.testing.assert_array_equal(simulator.render(), after)
        self.assertEqual(transition[1:4], replay[1:4])

    def test_sokoban_reset_render_action_and_state_replay(self):
        self.smoke("sokoban/easy", "('move', 1)")

    def test_maze_reset_render_action_and_state_replay(self):
        self.smoke("maze_2d/easy", "('move', 1)")

    def test_pacman_reset_render_action_and_state_replay(self):
        self.smoke("pacman_2d/easy", "('move', 1)")

    def test_pusht_reset_render_action_and_state_replay(self):
        self.smoke("pusht/easy", "('move', [256, 256])", ("torch", "cv2", "pymunk", "shapely"))


if __name__ == "__main__":
    unittest.main()
