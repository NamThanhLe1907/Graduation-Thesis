import unittest
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utility.performance_monitor import PerformanceMonitor

class TestPerformanceMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = PerformanceMonitor()

    def test_fps(self):
        for _ in range(5):  # Giả lập xử lý 5 khung hình
            self.monitor.start_frame()
            time.sleep(0.1)  # Giả lập thời gian inference
            self.monitor.end_frame()
        fps = self.monitor.get_fps()
        self.assertGreater(fps, 0)  # Kiểm tra FPS lớn hơn 0

    def test_average_inference_time(self):
        for _ in range(5):  # Giả lập xử lý 5 khung hình
            self.monitor.start_frame()
            time.sleep(0.1)  # Giả lập thời gian inference
            self.monitor.end_frame()
        avg_inference_time = self.monitor.get_average_inference_time()
        self.assertGreater(avg_inference_time, 0)  # Kiểm tra thời gian inference trung bình lớn hơn 0

if __name__ == "__main__":
    unittest.main()