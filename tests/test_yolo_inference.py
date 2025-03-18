import unittest
import cv2
from utils.yolo_inference import YOLOInference

class TestYOLOInference(unittest.TestCase):
    def setUp(self):
        self.yolo_inference = YOLOInference(model_path='best.pt')
        self.test_frame = cv2.imread('sample_image.jpg')  # Thay thế bằng một hình ảnh mẫu

    def test_infer(self):
        results = self.yolo_inference.infer(self.test_frame)
        self.assertIsNotNone(results)
        self.assertGreater(len(results.xyxy[0]), 0)  # Kiểm tra có phát hiện đối tượng

if __name__ == "__main__":
    unittest.main()