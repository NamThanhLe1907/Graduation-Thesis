import unittest
import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.visualization import AppGUI

class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = AppGUI()
        self.test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Khung hình đen
        self.boxes = [[100, 100, 200, 200]]  # Example bounding box
        self.confidences = [0.95]  # Example confidence
        self.class_ids = [0]  # Example class ID
        self.class_names = ["pallet"]  # Example class names

    def test_draw_boxes(self):
        frame_with_boxes = self.visualizer.draw_boxes(self.test_frame, self.boxes, self.confidences, self.class_ids, self.class_names)
        self.assertIsNotNone(frame_with_boxes)

    def test_show_frame(self):
        # Không thể kiểm tra hiển thị, nhưng có thể đảm bảo không có lỗi khi gọi
        self.visualizer.show_frame(self.test_frame)

if __name__ == "__main__":
    unittest.main()