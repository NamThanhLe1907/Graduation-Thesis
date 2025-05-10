"""
Test cho mock camera với threading
"""
import time
import unittest

from tests.mock_frame_camera import MockFrameCamera
from tests.test_utils import create_mock_camera


class TestMockFrameCamera(unittest.TestCase):
    def test_capture_frames(self):
        """Test lấy frame từ MockFrameCamera"""
        # Tạo camera với threading thay vì multiprocessing
        cam = MockFrameCamera(create_mock_camera)
        
        # Khởi động
        cam.start()
        
        frames = []
        deadline = time.time() + 5  # đợi tối đa 5s
        print("Bắt đầu lấy frame...")
        
        while time.time() < deadline and len(frames) < 10:
            f = cam.get_frame(timeout=0.5)
            if f is not None:
                print(f"Lấy được frame: {f}")
                frames.append(f)
            time.sleep(0.1)

        # In thông tin debug
        print(f"Frames captured: {len(frames)}")
        print(f"Frame counter: {cam.frame_counter}")
        print(f"Frames trong camera: {cam.frames_captured}")
        
        # Dừng thread
        cam.stop()

        # Phải lấy được ít nhất 1 frame
        self.assertGreater(len(frames), 0, "Không lấy được frame nào")
        for f in frames:
            self.assertTrue(f.startswith("frame_"))

        # Bộ đếm phải >= số frame thu được
        self.assertGreaterEqual(cam.frame_counter, len(frames))


if __name__ == "__main__":
    unittest.main(verbosity=2) 