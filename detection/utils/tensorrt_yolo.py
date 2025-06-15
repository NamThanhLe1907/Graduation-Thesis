"""
Cung cấp lớp và phương thức để phát hiện đối tượng bằng YOLO TensorRT.
"""
import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from typing import Dict, List, Optional, Union, Tuple, Any

class YOLOTensorRT:
    """
    Lớp thực hiện suy luận YOLO sử dụng TensorRT engine cho phát hiện đối tượng.
    """
    
    def __init__(self, 
                 engine_path: str = "best.engine", 
                 conf: Optional[float] = 0.25, 
                 iou: float = 0.5):
        """
        Khởi tạo mô hình YOLO TensorRT.
        
        Args:
            engine_path: Đường dẫn đến file TensorRT engine
            conf: Ngưỡng tin cậy (confidence threshold)
            iou: Ngưỡng IoU cho Non-Maximum Suppression
        """
        # Kiểm tra đường dẫn engine
        if not os.path.exists(engine_path):
            # Thử tìm trong thư mục hiện tại
            current_dir = os.path.dirname(os.path.abspath(__file__))
            engine_path_alt = os.path.join(current_dir, os.path.basename(engine_path))
            if os.path.exists(engine_path_alt):
                engine_path = engine_path_alt
            else:
                raise FileNotFoundError(f"Không tìm thấy file engine: {engine_path}")
                
        print(f"Đang tải TensorRT engine từ {engine_path}")
        
        # Sử dụng YOLO trực tiếp thay vì pycuda
        self.model = YOLO(engine_path, task='obb')
        
        # Lưu ngưỡng tin cậy và IoU
        self.conf_threshold = conf
        self.iou_threshold = iou
        self.model.conf = conf
        self.model.iou = iou
        
        # Khởi tạo device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Đang sử dụng device: {self.device}")
        
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Phát hiện đối tượng OBB và trả về các thông tin cần thiết.
        
        Args:
            frame: Khung hình dạng numpy array (BGR)
            
        Returns:
            Dict: Từ điển chứa thông tin phát hiện đối tượng
            {
                'obb_boxes': List các OBB dạng [cx, cy, width, height, angle],
                'corners': List các góc OBB (8 điểm tọa độ),
                'scores': List các giá trị tin cậy,
                'classes': List các lớp đối tượng,
                'annotated_frame': Khung hình có vẽ kết quả phát hiện
            }
        """
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào không hợp lệ")
        
        # Thực hiện suy luận bằng model YOLO OBB
        results = self.model(frame, conf=self.conf_threshold, verbose=False, imgsz=(1280,720), device=0)
        
        # Trích xuất thông tin OBB từ kết quả
        result_obj = results[0]
        
        # Trích xuất thông tin OBB
        detection_result = {}
        
        # Thêm khung hình đã được vẽ kết quả phát hiện
        detection_result['annotated_frame'] = result_obj.plot()
        
        # Trích xuất thông tin OBB nếu có
        if hasattr(result_obj, 'obb') and result_obj.obb is not None:
            obb = result_obj.obb
            
            # Trích xuất thông tin cơ bản
            if hasattr(obb, 'conf'):
                confs = obb.conf.cpu().numpy() if hasattr(obb.conf, 'cpu') else obb.conf
                detection_result['scores'] = confs.tolist() if isinstance(confs, np.ndarray) else confs
            
            if hasattr(obb, 'cls'):
                cls = obb.cls.cpu().numpy() if hasattr(obb.cls, 'cpu') else obb.cls
                detection_result['classes'] = cls.tolist() if isinstance(cls, np.ndarray) else cls
            
            # Trích xuất thông tin OBB cụ thể
            if hasattr(obb, 'xyxy'):
                xyxy = obb.xyxy.cpu().numpy() if hasattr(obb.xyxy, 'cpu') else obb.xyxy
                detection_result['bounding_boxes'] = xyxy.tolist() if isinstance(xyxy, np.ndarray) else xyxy
            
            if hasattr(obb, 'xywhr'):
                xywhr = obb.xywhr.cpu().numpy() if hasattr(obb.xywhr, 'cpu') else obb.xywhr
                detection_result['obb_boxes'] = xywhr.tolist() if isinstance(xywhr, np.ndarray) else xywhr
            
            if hasattr(obb, 'xy'):
                corners = obb.xy
                if hasattr(corners, 'cpu'):
                    corners = corners.cpu().numpy()
                detection_result['corners'] = corners.tolist() if isinstance(corners, np.ndarray) else corners
        
        # Nếu không tìm thấy thông tin OBB, thử lấy thông tin từ boxes thông thường
        elif hasattr(result_obj, 'boxes') and result_obj.boxes is not None:
            boxes = result_obj.boxes
            
            # Trích xuất thông tin cơ bản
            if hasattr(boxes, 'conf'):
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                detection_result['scores'] = confs.tolist() if isinstance(confs, np.ndarray) else confs
            
            if hasattr(boxes, 'cls'):
                cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                detection_result['classes'] = cls.tolist() if isinstance(cls, np.ndarray) else cls
            
            if hasattr(boxes, 'xyxy'):
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                detection_result['bounding_boxes'] = xyxy.tolist() if isinstance(xyxy, np.ndarray) else xyxy
        
        return detection_result


if __name__ == "__main__":
    inference = YOLOTensorRT(engine_path="best.engine", conf=0.25)
    frame = cv2.imread("./sample_image.jpg")  # Thay bằng khung hình thực tế
    
    if frame is None:
        print("Lỗi: Không thể đọc 'sample_image.jpg'.")
    else:
        # Phát hiện đối tượng
        detections = inference.detect(frame)
        
        # Hiển thị kết quả
        annotated_frame = detections['annotated_frame']
        cv2.imshow("Kết quả phát hiện", annotated_frame)
        
        # In ra thông tin về các đối tượng phát hiện được
        if 'obb_boxes' in detections:
            print(f"Đã phát hiện {len(detections['obb_boxes'])} đối tượng OBB.")
        elif 'bounding_boxes' in detections:
            print(f"Đã phát hiện {len(detections['bounding_boxes'])} đối tượng.")
        else:
            print("Không phát hiện đối tượng nào.")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows() 