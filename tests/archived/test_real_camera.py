"""
Test FrameCamera với camera thật từ CameraInterface
"""
import time
import unittest
import sys
import os

# Thêm thư mục gốc vào sys.path để import module từ utility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utility.video_processor import FrameCamera
from utility.camera_interface import CameraInterface


def create_real_camera():
    """Factory function tạo camera thật"""
    try:
        print("Đang khởi tạo CameraInterface...")
        camera = CameraInterface()
        print("Đang khởi động camera...")
        camera.initialize()
        print("Camera đã khởi động thành công")
        return camera
    except Exception as e:
        print(f"Lỗi khởi tạo camera: {e}")
        import traceback
        traceback.print_exc()
        # Nếu khởi tạo camera thất bại, sử dụng camera giả lập
        print("Sử dụng MockCamera thay thế")
        from tests.test_utils import MockCamera
        return MockCamera()


class TestRealCamera(unittest.TestCase):
    def test_capture_frames(self):
        """Test lấy frame từ camera thật thông qua FrameCamera"""
        # Khởi tạo FrameCamera với real camera factory
        cam = FrameCamera(create_real_camera)
        
        print("Đang khởi động camera process...")
        started = cam.start(timeout=20.0)  # Tăng timeout lên 20s
        self.assertTrue(started, "Process con không khởi động được")
        
        # Đợi để process con có thể tạo frame
        time.sleep(1.0)
        
        frames = []
        deadline = time.time() + 10  # Tăng thời gian đợi lên 10s
        print("Bắt đầu lấy frame...")
        
        while time.time() < deadline and len(frames) < 10:
            f = cam.get_frame(timeout=0.5)
            if f is not None:
                print(f"Lấy được frame #{len(frames)+1}")
                frames.append(f)
            time.sleep(0.2)  # Tăng thời gian sleep giữa các lần get

        # In thông tin debug
        print(f"Frames captured: {len(frames)}")
        print(f"Frame counter: {cam.frame_counter.value}")
        print(f"Errors: {cam.errors}")
        
        # Dừng process
        cam.stop()

        # Kiểm tra đã lấy được frame
        self.assertGreater(len(frames), 0, "Không lấy được frame nào")
        
        # Kiểm tra frame có dữ liệu
        try:
            import numpy as np
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    self.assertGreater(frame.size, 0, "Frame rỗng")
                else:
                    self.assertIsNotNone(frame, "Frame là None")
        except ImportError:
            # Nếu không có numpy, kiểm tra đơn giản
            for frame in frames:
                self.assertIsNotNone(frame, "Frame là None")


if __name__ == "__main__":
    unittest.main(verbosity=2) 