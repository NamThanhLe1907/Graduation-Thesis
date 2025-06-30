"""
Cung cấp các công cụ xử lý khung hình từ camera.
"""
import cv2
import numpy as np
from typing import Tuple, Optional


class FrameProcessor:
    """Lớp xử lý khung hình với các phương thức hữu ích."""

    def __init__(self, 
                 resize_dims: Optional[Tuple[int, int]] = None,
                 blur_kernel: int = 0,
                 clahe_clip: float = 2.0,
                 denoise_strength: int = 0):
        """
        Khởi tạo bộ xử lý khung hình.
        
        Args:
            resize_dims: Kích thước mới cho khung hình (width, height), None để giữ nguyên
            blur_kernel: Kích thước kernel làm mờ (0 để tắt)
            clahe_clip: Giới hạn clip cho CLAHE (Contrast Limited Adaptive Histogram Equalization)
            denoise_strength: Cường độ khử nhiễu (0 để tắt)
        """
        self.resize_dims = resize_dims
        self.blur_kernel = blur_kernel
        self.clahe_clip = clahe_clip
        self.denoise_strength = denoise_strength
        
        # Khởi tạo CLAHE
        if self.clahe_clip > 0:
            self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý khung hình: điều chỉnh độ tương phản và giảm nhiễu.
        
        Args:
            frame: Khung hình cần xử lý
            
        Returns:
            np.ndarray: Khung hình đã xử lý
        """
        if frame is None:
            raise ValueError("Khung hình đầu vào không thể là None")
            
        # Thay đổi kích thước nếu cần
        if self.resize_dims is not None:
            frame = cv2.resize(frame, self.resize_dims)
            
        # Làm mờ khung hình nếu cần
        if self.blur_kernel > 0:
            frame = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
            
        # Cải thiện độ tương phản nếu cần
        if self.clahe_clip > 0:
            # Chuyển đổi sang LAB màu
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Áp dụng CLAHE vào kênh L
            cl = self.clahe.apply(l)
            
            # Ghép kênh và chuyển đổi trở lại BGR
            merged = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            
        # Khử nhiễu nếu cần
        if self.denoise_strength > 0:
            frame = cv2.fastNlMeansDenoisingColored(
                frame, 
                None, 
                self.denoise_strength, 
                self.denoise_strength, 
                7, 
                21
            )
            
        return frame
        
    def apply_text_overlay(self, frame: np.ndarray, text: str, position: Tuple[int, int] = (10, 30),
                          font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
        """
        Thêm văn bản vào khung hình.
        
        Args:
            frame: Khung hình cần thêm văn bản
            text: Nội dung văn bản
            position: Vị trí (x, y) của văn bản
            font_scale: Tỷ lệ phông chữ
            color: Màu sắc BGR (Blue, Green, Red)
            thickness: Độ dày của chữ
            
        Returns:
            np.ndarray: Khung hình đã thêm văn bản
        """
        # Tạo bản sao để không thay đổi khung hình gốc
        result = frame.copy()
        cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness, cv2.LINE_AA)
        return result
    
    def draw_detection_boxes(self, frame: np.ndarray, boxes: list, 
                            color: Tuple[int, int, int] = (0, 255, 0),
                            thickness: int = 2) -> np.ndarray:
        """
        Vẽ các hộp giới hạn lên khung hình.
        
        Args:
            frame: Khung hình cần vẽ
            boxes: Danh sách các hộp giới hạn dạng [x1, y1, x2, y2] hoặc [x, y, w, h]
            color: Màu sắc BGR (Blue, Green, Red)
            thickness: Độ dày của đường viền
            
        Returns:
            np.ndarray: Khung hình đã vẽ các hộp giới hạn
        """
        # Tạo bản sao để không thay đổi khung hình gốc
        result = frame.copy()
        
        for box in boxes:
            if len(box) == 4:
                # Kiểm tra loại bounding box (x1,y1,x2,y2) hoặc (x,y,w,h)
                if box[2] < box[0] or box[3] < box[1]:  # Nếu là (x,y,w,h) thì w,h luôn dương
                    # Chuyển đổi từ (x,y,w,h) sang (x1,y1,x2,y2)
                    x, y, w, h = box
                    cv2.rectangle(result, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)
                else:
                    # Đã là (x1,y1,x2,y2)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
        return result


if __name__ == "__main__":
    processor = FrameProcessor(blur_kernel=3, clahe_clip=2.0)
    # Example usage
    frame = cv2.imread('sample_image.jpg')  # Thay bằng khung hình thực tế
    
    if frame is not None:
        processed_frame = processor.preprocess_frame(frame)
        
        # Thêm văn bản thử nghiệm
        processed_frame = processor.apply_text_overlay(processed_frame, "Test Processing", (50, 50))
        
        # Vẽ bounding box thử nghiệm
        boxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
        processed_frame = processor.draw_detection_boxes(processed_frame, boxes)
        
        cv2.imshow("Processed Frame", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không thể đọc ảnh thử nghiệm") 