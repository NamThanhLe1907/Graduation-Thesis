import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.postprocessing import PostProcessor

class TestPostProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PostProcessor(alpha=0.2)
        self.current_boxes = np.array([[100, 100, 200, 200], [150, 150, 250, 250]])  # Example bounding boxes

    def test_smooth_boxes(self):
        smoothed_boxes = self.processor.smooth_boxes(self.current_boxes)
        self.assertIsNotNone(smoothed_boxes)
        self.assertEqual(len(smoothed_boxes), len(self.current_boxes))

    def test_detect_collisions(self):
        collisions = self.processor.detect_collisions(self.current_boxes)
        self.assertGreater(len(collisions), 0)  # Kiểm tra có phát hiện va chạm

if __name__ == "__main__":
    unittest.main()