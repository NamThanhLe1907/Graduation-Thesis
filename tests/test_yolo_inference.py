import unittest
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.yolo_inference import YOLOInference

class TestYOLOInference(unittest.TestCase):
    def setUp(self):
        self.yolo_inference = YOLOInference(model_path='best.pt')
        self.test_frame = cv2.imread('sample_image.jpg')  # Thay thế bằng một hình ảnh mẫu

    def test_infer(self):
        results = self.yolo_inference.infer(self.test_frame)
        self.assertIsNotNone(results)
        self.annotated_frame= results[0].plot()# Kiểm tra có phát hiện đối tượng

if __name__ == "__main__":
    unittest.main()