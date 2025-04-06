import unittest
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utility.yolo_inference import YOLOInference


class TestYOLOInference(unittest.TestCase):
    def setUp(self):
        model_path = Path(__file__).parent / 'best.pt'  # Nếu file best.pt nằm cùng thư mục tests
        self.yolo_inference = YOLOInference(model_path=str(model_path))
        self.test_frame = cv2.imread('image_3.jpg')  # Thay thế bằng hình ảnh mẫu của bạn

    def test_infer(self):
        results = self.yolo_inference.infer(self.test_frame)
        self.assertIsNotNone(results)
        annotated_frame = results[0].plot()  # Tạo hình ảnh đã annotate

        # Hiển thị hình ảnh đã annotate
        cv2.imshow("Annotated Result", annotated_frame)
        cv2.waitKey(0)  # Chờ người dùng nhấn phím
        cv2.destroyAllWindows()  # Đóng tất cả cửa sổ hiển thị hình ảnh

if __name__ == "__main__":
    unittest.main()