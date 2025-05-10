import time
import unittest

from utility.video_processor import FrameCamera
from tests.test_utils import create_mock_camera


class MockCamera:
    """Camera giả lập – trả chuỗi 'frame_1', 'frame_2', …"""
    def __init__(self):
        self._n = 0

    def get_frame(self):
        self._n += 1
        return f"frame_{self._n}"


# Tạo hàm factory riêng biệt thay vì dùng lambda để tránh vấn đề pickle
def create_mock_camera():
    return MockCamera()


class TestFrameCamera(unittest.TestCase):
    def test_capture_frames(self):
        """Test lấy frame từ FrameCamera"""
        # Quan trọng: sử dụng function từ top-level module, không dùng lambda
        cam = FrameCamera(create_mock_camera)
        
        # Khởi động và đợi process con - sử dụng cơ chế đồng bộ hóa mới
        started = cam.start(timeout=10.0)  # đợi tối đa 10s
        self.assertTrue(started, "Process con không khởi động được")

        # Đợi một chút để process con có thể tạo frame
        time.sleep(0.5)
        
        frames = []
        deadline = time.time() + 8  # tăng thời gian chờ lên 8s
        print("Bắt đầu lấy frame...")
        
        while time.time() < deadline and len(frames) < 10:
            f = cam.get_frame(timeout=0.5)  # tăng timeout lên 0.5s
            if f is not None:
                print(f"Lấy được frame: {f}")
                frames.append(f)
            time.sleep(0.2)  # tăng thời gian sleep giữa các lần get

        # In thông tin debug
        print(f"Frames captured: {len(frames)}")
        print(f"Frame counter: {cam.frame_counter.value}")
        
        # Dừng process
        cam.stop()

        # Phải lấy được ít nhất 1 frame
        self.assertGreater(len(frames), 0, "Không lấy được frame nào")
        for f in frames:
            self.assertTrue(f.startswith("frame_"))

        # Bộ đếm trong process con phải >= số frame thu được
        self.assertGreaterEqual(cam.frame_counter.value, len(frames))


if __name__ == "__main__":
    unittest.main(verbosity=2)

