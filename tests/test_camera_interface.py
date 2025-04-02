import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.camera_interface import CameraInterface

class TestCameraInterface(unittest.TestCase):
    def setUp(self):
        self.camera = CameraInterface(camera_index=0)

    def test_initialize(self):
        self.camera.initialize()
        self.assertIsNotNone(self.camera.capture)

    def test_get_frame(self):
        self.camera.initialize()
        frame = self.camera.get_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[0], 720)  # Kiểm tra chiều cao
        self.assertEqual(frame.shape[1], 1280)  # Kiểm tra chiều rộng

    def tearDown(self):
        self.camera.release()

if __name__ == "__main__":
    unittest.main()