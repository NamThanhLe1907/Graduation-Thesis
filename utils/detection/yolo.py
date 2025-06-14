"""
Cung cấp lớp và phương thức để phát hiện đối tượng bằng YOLO.
"""
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import os


class YOLOInference:
    """
    Lớp thực hiện suy luận YOLO cho phát hiện đối tượng hướng chữ nhật (OBB - Oriented Bounding Box).
    """
    
    def __init__(self, 
                 model_path: str = "best.pt", 
                 conf: Optional[float] = None, 
                 iou: float = 0.5,
                 device: Optional[str] = None):
        """
        Khởi tạo mô hình YOLO.
        
        Args:
            model_path: Đường dẫn đến file mô hình
            conf: Ngưỡng tin cậy (confidence threshold), None để sử dụng mặc định của mô hình
            iou: Ngưỡng IoU cho Non-Maximum Suppression
            device: Thiết bị xử lý ('cuda', 'cpu' hoặc None để tự phát hiện)
        """
        # Tự phát hiện thiết bị nếu không chỉ định
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Kiểm tra đường dẫn mô hình
        if not os.path.exists(model_path):
            # Thử tìm trong thư mục hiện tại
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path_alt = os.path.join(current_dir, os.path.basename(model_path))
            if os.path.exists(model_path_alt):
                model_path = model_path_alt
            else:
                raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
                
        print(f"Đang tải mô hình YOLO từ {model_path} trên thiết bị {device}")
        
        self.device = device
        self.use_cuda = device == 'cuda'
        
        # Tải mô hình với cấu hình phù hợp
        self.model = YOLO(model_path, task='obb')
        
        # Nếu sử dụng CUDA, bật half precision
        if self.use_cuda:
            self.model = self.model.half().to(device)
        else:
            self.model = self.model.to(device)
            
        # Thiết lập ngưỡng phát hiện
        self.model.conf = conf  # Ngưỡng tin cậy
        self.model.iou = iou    # Ngưỡng IoU cho NMS
        
        print(f"Đã khởi tạo mô hình YOLO: {type(self.model).__name__}")
        
    def infer(self, frame: np.ndarray) -> Any:
        """
        Thực hiện suy luận trên một khung hình.
        
        Args:
            frame: Khung hình dạng numpy array (BGR)
            
        Returns:
            results: Kết quả từ YOLO model
        """
        # Kiểm tra khung hình đầu vào
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào không hợp lệ")
        
        # Thực hiện suy luận với cấu hình phù hợp
        if self.use_cuda:
            with torch.amp.autocast('cuda'):  # Sử dụng autocast để tối ưu hiệu suất
                results = self.model(
                    frame, 
                    task='obb', 
                    imgsz=self.model.args['imgsz'], 
                    device=self.device
                )
        else:
            results = self.model(
                frame, 
                task='obb', 
                imgsz=self.model.args['imgsz'], 
                device=self.device
            )
            
        return results
        
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Phát hiện đối tượng và trả về các thông tin cần thiết.
        
        Args:
            frame: Khung hình dạng numpy array (BGR)
            
        Returns:
            Dict: Từ điển chứa thông tin phát hiện đối tượng
            {
                'bounding_boxes': List các bounding box dạng [x1, y1, x2, y2],
                'scores': List các giá trị tin cậy,
                'classes': List các lớp đối tượng,
                'obb': Đối tượng OBB gốc từ YOLO,
                'annotated_frame': Khung hình có vẽ kết quả phát hiện
            }
        """
        results = self.infer(frame)
        
        # Trích xuất thông tin cần thiết từ kết quả
        result_obj = results[0]  # Lấy kết quả đầu tiên
        
        # Trích xuất thông tin
        boxes = result_obj.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
        
        # Chuẩn bị kết quả
        detection_result = {
            'bounding_boxes': xyxy.tolist() if isinstance(xyxy, np.ndarray) else xyxy,
            'scores': confs.tolist() if isinstance(confs, np.ndarray) else confs,
            'classes': cls.tolist() if isinstance(cls, np.ndarray) else cls,
            'obb': result_obj.obb,  # Giữ nguyên đối tượng OBB
            'annotated_frame': result_obj.plot()  # Vẽ kết quả lên khung hình
        }
        
        return detection_result


if __name__ == "__main__":
    inference = YOLOInference(model_path="best.pt", conf=0.1)
    frame = cv2.imread("./sample_image.jpg")  # Thay bằng khung hình thực tế
    
    if frame is None:
        print("Lỗi: Không thể đọc 'sample_image.jpg'.")
    else:
        # Phát hiện đối tượng
        detections = inference.detect(frame)
        
        # Hiển thị kết quả
        annotated_frame = detections['annotated_frame']
        cv2.imshow("Kết quả phát hiện", annotated_frame)
        
        # In ra số lượng đối tượng phát hiện được
        print(f"Đã phát hiện {len(detections['bounding_boxes'])} đối tượng.")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows() 