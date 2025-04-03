import unittest
import numpy as np
import os
import cv2
import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path để import module
sys.path.append(str(Path(__file__).parent.parent))
from src.postprocessing import PostProcessor

def draw_boxes(image, boxes, color, thickness=2):
    """
    Hàm hỗ trợ vẽ các bounding box lên hình.
    """
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

class TestPostProcessor(unittest.TestCase):
    def setUp(self):
        # Khởi tạo PostProcessor với alpha = 0.2
        self.processor = PostProcessor(alpha=0.2)
        # Ví dụ bounding boxes: [x1, y1, x2, y2]
        self.current_boxes = np.array([[100, 100, 200, 200], [150, 150, 250, 250]])
        # Tạo hình nền trắng dùng cho việc hiển thị (400x400)
        self.image = np.ones((400, 400, 3), dtype=np.uint8) * 255

    def test_visualize_smooth_boxes(self):
        """
        Hiển thị kết quả làm mịn:
          - Vẽ bounding box gốc bằng màu xanh (blue).
          - Vẽ bounding box đã được làm mịn bằng màu xanh lá (green).
        """
        # Lấy kết quả làm mịn
        smoothed_boxes = self.processor.smooth_boxes(self.current_boxes)
        
        vis_image = self.image.copy()
        # Vẽ các box gốc (màu xanh)
        draw_boxes(vis_image, self.current_boxes, (255, 0, 0), 2)
        # Vẽ các box sau khi làm mịn (màu xanh lá)
        draw_boxes(vis_image, smoothed_boxes, (0, 255, 0), 2)
        
        cv2.imshow("Smooth Boxes Visualization", vis_image)
        cv2.waitKey(3000)  # Hiển thị trong 3 giây
        cv2.destroyAllWindows()

    def test_visualize_collisions(self):
        """
        Hiển thị kết quả phát hiện va chạm:
          - Vẽ các bounding box gốc (màu xanh).
          - Đánh dấu điểm trung tâm của các box có va chạm và vẽ đường nối giữa chúng (màu đỏ).
        """
        collisions = self.processor.detect_collisions(self.current_boxes)
        vis_image = self.image.copy()
        # Vẽ bounding box gốc bằng màu xanh
        draw_boxes(vis_image, self.current_boxes, (255, 0, 0), 2)
        
        # Với mỗi cặp box có va chạm, vẽ vòng tròn và đường nối
        for (i, j) in collisions:
            box1 = self.current_boxes[i]
            box2 = self.current_boxes[j]
            center1 = (int((box1[0] + box1[2]) / 2), int((box1[1] + box1[3]) / 2))
            center2 = (int((box2[0] + box2[2]) / 2), int((box2[1] + box2[3]) / 2))
            cv2.circle(vis_image, center1, 5, (0, 0, 255), -1)
            cv2.circle(vis_image, center2, 5, (0, 0, 255), -1)
            cv2.line(vis_image, center1, center2, (0, 0, 255), 2)
        
        cv2.imshow("Collisions Visualization", vis_image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    def test_display_annotated_image(self):
        """
        Hiển thị hình annotated từ file (ví dụ: debug_output.jpg).
        Lưu ý: Test này sẽ mở cửa sổ hiển thị trong 3 giây.
        """
        image_path = "debug_output.jpg"  # Đường dẫn tới hình annotated (đã có từ ứng dụng chính)
        self.assertTrue(os.path.exists(image_path), "Annotated image file does not exist.")
        
        annotated_image = cv2.imread(image_path)
        self.assertIsNotNone(annotated_image, "Failed to load annotated image.")
        
        cv2.imshow("Annotated Image", annotated_image)
        cv2.waitKey(3000)  # Hiển thị trong 3 giây
        cv2.destroyAllWindows()

if __name__ == "__main__":
    unittest.main()
