"""Exercise the JPEG worker protocol with real subprocesses and CPU image data."""

import base64
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class JpegEncoderTests(unittest.TestCase):
    def setUp(self):
        missing = [name for name in ("numpy", "PIL") if importlib.util.find_spec(name) is None]
        if missing:
            message = "Missing CPU JPEG test dependencies: " + ", ".join(missing)
            if os.environ.get("AGIMA_REQUIRE_ENV_SMOKE") == "1":
                self.fail(message)
            self.skipTest(message)
        import numpy as np
        from PIL import Image
        self.np, self.Image = np, Image
        spec = importlib.util.spec_from_file_location(
            "jpeg_worker_checks", ROOT / "gymnasium/vlm/nonturbo_jpeg.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        # Transport tests use the installed Pillow codec. Its reported capability
        # is substituted only here, so no separate IJG installation is needed.
        self.worker = self.module._WORKER_SCRIPT.replace(
            "features.check_feature('libjpeg_turbo')", "False")

    def frame(self, seed):
        return self.np.random.default_rng(seed).integers(0, 256, (256, 256, 3), dtype=self.np.uint8)

    def expected(self, frame):
        output = io.BytesIO()
        self.Image.fromarray(frame).save(output, format="JPEG", quality=90)
        return base64.b64encode(output.getvalue()).decode("ascii")

    def test_large_frames_are_complete_and_next_frame_is_not_corrupted(self):
        with patch.object(self.module, "_WORKER_SCRIPT", self.worker):
            encoder = self.module.NonTurboJpegEncoder(sys.executable)
            self.addCleanup(encoder.close)
            for seed in (0, 1):
                frame = self.frame(seed)
                expected = self.expected(frame)
                self.assertGreater(len(expected), 65536)
                self.assertEqual(encoder.encode(frame), expected)

    def test_turbo_interpreter_is_rejected_and_failed_worker_is_stopped(self):
        worker = self.module._WORKER_SCRIPT.replace("features.check_feature('libjpeg_turbo')", "True")
        processes = []
        popen = self.module.subprocess.Popen

        def record_process(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process

        with patch.object(self.module, "_WORKER_SCRIPT", worker), \
                patch.object(self.module.subprocess, "Popen", side_effect=record_process):
            with self.assertRaisesRegex(RuntimeError, "requires Pillow built with non-turbo libjpeg"):
                self.module.NonTurboJpegEncoder(sys.executable)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())

    def test_incomplete_frame_fails_and_next_call_starts_a_clean_worker(self):
        broken = self.worker.replace("stdout.write(out)", "stdout.write(out[:7]); stdout.flush(); sys.exit(1)")
        with patch.object(self.module, "_WORKER_SCRIPT", broken):
            encoder = self.module.NonTurboJpegEncoder(sys.executable)
            self.addCleanup(encoder.close)
            with self.assertRaisesRegex(RuntimeError, "before completing a frame"):
                encoder.encode(self.frame(0))
        self.assertIsNone(encoder._proc)
        with patch.object(self.module, "_WORKER_SCRIPT", self.worker):
            frame = self.frame(1)
            self.assertEqual(encoder.encode(frame), self.expected(frame))


if __name__ == "__main__":
    unittest.main()
