"""
Cung cấp lớp và phương thức để phát hiện đối tượng bằng YOLO TensorRT.
Tương thích với Ultralytics OBB format mới.
"""
import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import (
    xyxy2xywh, 
    xywh2xyxy, 
    xywhr2xyxyxyxy,
    xyxyxyxy2xywhr,
    empty_like
)
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
        
    @staticmethod
    def convert_xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi format [center_x, center_y, width, height] sang [x1, y1, x2, y2].
        Sử dụng function từ ultralytics.utils.ops cho tính nhất quán.
        
        Args:
            xywh: Mảng numpy với shape (N, 4) chứa [center_x, center_y, width, height]
            
        Returns:
            np.ndarray: Mảng với shape (N, 4) chứa [x1, y1, x2, y2]
        """
        # Chuyển sang tensor để sử dụng ultralytics function
        if isinstance(xywh, np.ndarray):
            xywh_tensor = torch.from_numpy(xywh).float()
            xyxy_tensor = xywh2xyxy(xywh_tensor)
            return xyxy_tensor.numpy()
        else:
            return xywh2xyxy(xywh).numpy()
    
    @staticmethod 
    def convert_xywhr_to_corners(xywhr: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi từ [cx, cy, w, h, rotation] sang 4 corner points.
        Sử dụng function từ ultralytics.utils.ops cho tính nhất quán.
        
        Args:
            xywhr: Mảng numpy với shape (N, 5) chứa [center_x, center_y, width, height, rotation]
            
        Returns:
            np.ndarray: Mảng với shape (N, 4, 2) chứa 4 corner points cho mỗi box
        """
        # Chuyển sang tensor để sử dụng ultralytics function
        if isinstance(xywhr, np.ndarray):
            xywhr_tensor = torch.from_numpy(xywhr).float()
            corners_tensor = xywhr2xyxyxyxy(xywhr_tensor)
            return corners_tensor.numpy()
        else:
            return xywhr2xyxyxyxy(xywhr).numpy()
    
    @staticmethod
    def get_obb_corners(center_x: float, center_y: float, width: float, height: float, angle: float) -> np.ndarray:
        """
        Tính toán 4 góc của oriented bounding box.
        
        Args:
            center_x: Tọa độ x của tâm
            center_y: Tọa độ y của tâm
            width: Chiều rộng của box
            height: Chiều cao của box
            angle: Góc xoay (radian)
            
        Returns:
            np.ndarray: Mảng 4x2 chứa tọa độ 4 góc [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # Sử dụng ultralytics function để đảm bảo tính nhất quán
        xywhr = np.array([[center_x, center_y, width, height, angle]])
        corners = YOLOTensorRT.convert_xywhr_to_corners(xywhr)
        return corners[0]  # Trả về corners của box đầu tiên
    
    def extract_obb_data_new_format(self, obb_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Trích xuất dữ liệu từ OBB tensor format mới của Ultralytics.
        
        Args:
            obb_tensor: Tensor với shape (N, 7) = [x, y, w, h, angle, confidence, class]
            
        Returns:
            Dict: Dữ liệu OBB đã được xử lý
        """
        # Chuyển tensor sang numpy
        if hasattr(obb_tensor, 'cpu'):
            obb_data = obb_tensor.cpu().numpy()
        else:
            obb_data = obb_tensor
            
        result = {
            'bounding_boxes': [],
            'obb_boxes': [],
            'corners': [],
            'scores': [],
            'classes': []
        }
        
        if len(obb_data.shape) == 2 and obb_data.shape[1] >= 7:
            # Extract theo format mới: [x, y, w, h, angle, confidence, class]
            centers_and_sizes = obb_data[:, :4]  # [center_x, center_y, width, height]
            angles = obb_data[:, 4]              # angle (radian)
            confidences = obb_data[:, 5]         # confidence
            classes = obb_data[:, 6]             # class
            
            # Chuyển đổi sang xyxy format bằng ultralytics function
            xyxy = self.convert_xywh_to_xyxy(centers_and_sizes)
            
            # Tạo xywhr format (center_x, center_y, width, height, rotation)
            xywhr = np.column_stack([centers_and_sizes, angles])
            
            # Tính toán corners cho tất cả OBB bằng ultralytics function
            corners_array = self.convert_xywhr_to_corners(xywhr)  # Shape: (N, 4, 2)
            corners_list = [corners_array[i].tolist() for i in range(len(corners_array))]
            
            result.update({
                'bounding_boxes': xyxy.tolist(),
                'obb_boxes': xywhr.tolist(),
                'corners': corners_list,
                'scores': confidences.tolist(),
                'classes': classes.tolist()
            })
            
        return result
    
    def extract_obb_data_legacy_format(self, obb) -> Dict[str, Any]:
        """
        Trích xuất dữ liệu từ OBB object format cũ (có attributes riêng biệt).
        
        Args:
            obb: OBB object với các attributes như xyxy, xywhr, conf, cls
            
        Returns:
            Dict: Dữ liệu OBB đã được xử lý
        """
        result = {}
        
        # Trích xuất thông tin cơ bản
        if hasattr(obb, 'conf'):
            confs = obb.conf.cpu().numpy() if hasattr(obb.conf, 'cpu') else obb.conf
            result['scores'] = confs.tolist() if isinstance(confs, np.ndarray) else confs
        
        if hasattr(obb, 'cls'):
            cls = obb.cls.cpu().numpy() if hasattr(obb.cls, 'cpu') else obb.cls
            result['classes'] = cls.tolist() if isinstance(cls, np.ndarray) else cls
        
        # Trích xuất thông tin OBB cụ thể
        if hasattr(obb, 'xyxy'):
            xyxy = obb.xyxy.cpu().numpy() if hasattr(obb.xyxy, 'cpu') else obb.xyxy
            result['bounding_boxes'] = xyxy.tolist() if isinstance(xyxy, np.ndarray) else xyxy
        
        if hasattr(obb, 'xywhr'):
            xywhr = obb.xywhr.cpu().numpy() if hasattr(obb.xywhr, 'cpu') else obb.xywhr
            result['obb_boxes'] = xywhr.tolist() if isinstance(xywhr, np.ndarray) else xywhr
            
            # Tính toán corners từ xywhr bằng ultralytics function
            if len(xywhr.shape) == 2:
                corners_array = self.convert_xywhr_to_corners(xywhr)
                result['corners'] = [corners_array[i].tolist() for i in range(len(corners_array))]
        
        # Fallback: sử dụng xy hoặc xyxyxyxy nếu có
        if 'corners' not in result:
            if hasattr(obb, 'xy'):
                corners = obb.xy
                if hasattr(corners, 'cpu'):
                    corners = corners.cpu().numpy()
                result['corners'] = corners.tolist() if isinstance(corners, np.ndarray) else corners
            elif hasattr(obb, 'xyxyxyxy'):
                # Nếu có xyxyxyxy, chuyển đổi thành corners format
                xyxyxyxy = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, 'cpu') else obb.xyxyxyxy
                corners_list = []
                for poly in xyxyxyxy:
                    # Chuyển flat array (8 giá trị) thành 4 điểm (4x2)
                    corners = np.array(poly).reshape(4, 2)
                    corners_list.append(corners.tolist())
                result['corners'] = corners_list
            
        return result
        
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Phát hiện đối tượng OBB và trả về các thông tin cần thiết.
        Tương thích với cả format mới và cũ của Ultralytics.
        
        Args:
            frame: Khung hình dạng numpy array (BGR)
            
        Returns:
            Dict: Từ điển chứa thông tin phát hiện đối tượng
            {
                'bounding_boxes': List các bounding box dạng [x1, y1, x2, y2],
                'obb_boxes': List các OBB dạng [cx, cy, width, height, angle],
                'corners': List các góc OBB (4 điểm tọa độ cho mỗi box),
                'scores': List các giá trị tin cậy,
                'classes': List các lớp đối tượng,
                'annotated_frame': Khung hình có vẽ kết quả phát hiện,
                'format_type': 'new', 'legacy' hoặc 'fallback' để biết format nào được sử dụng
            }
        """
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào không hợp lệ")
        
        # Thực hiện suy luận bằng model YOLO OBB
        results = self.model(frame, conf=self.conf_threshold, verbose=False, imgsz=(1280,1024), device=0)
        
        # Trích xuất thông tin OBB từ kết quả
        result_obj = results[0]
        
        # Khởi tạo kết quả
        detection_result = {
            'annotated_frame': result_obj.plot(),
            'bounding_boxes': [],
            'obb_boxes': [],
            'corners': [],
            'scores': [],
            'classes': [],
            'format_type': 'unknown'
        }
        
        # Kiểm tra và trích xuất thông tin OBB
        if hasattr(result_obj, 'obb') and result_obj.obb is not None:
            obb = result_obj.obb
            
            # Kiểm tra xem đây là tensor (format mới) hay object (format cũ)
            if isinstance(obb, torch.Tensor):
                # Format mới: OBB là tensor
                # print("[DEBUG] Sử dụng format mới của Ultralytics OBB (tensor)")
                obb_data = self.extract_obb_data_new_format(obb)
                detection_result.update(obb_data)
                detection_result['format_type'] = 'new'
                
            else:
                # Format cũ: OBB là object với các attributes
                # print("[DEBUG] Sử dụng format cũ của Ultralytics OBB (object)")
                obb_data = self.extract_obb_data_legacy_format(obb)
                detection_result.update(obb_data)
                detection_result['format_type'] = 'legacy'
        
        # Fallback: Nếu không có OBB data, thử lấy từ boxes thông thường
        if not detection_result['bounding_boxes'] and hasattr(result_obj, 'boxes') and result_obj.boxes is not None:
            # print("[DEBUG] Fallback: Sử dụng detection boxes thông thường")
            boxes = result_obj.boxes
            
            if hasattr(boxes, 'xyxy'):
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                detection_result['bounding_boxes'] = xyxy.tolist() if isinstance(xyxy, np.ndarray) else xyxy
            
            if hasattr(boxes, 'conf'):
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                detection_result['scores'] = confs.tolist() if isinstance(confs, np.ndarray) else confs
            
            if hasattr(boxes, 'cls'):
                cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                detection_result['classes'] = cls.tolist() if isinstance(cls, np.ndarray) else cls
                
            detection_result['format_type'] = 'fallback'
        
        # Thông tin debug
        # print(f"[DEBUG] Detected {len(detection_result['bounding_boxes'])} objects")
        # print(f"[DEBUG] Format type: {detection_result['format_type']}")
        # if detection_result['obb_boxes']:
            # print(f"[DEBUG] OBB boxes available: {len(detection_result['obb_boxes'])}")
        # if detection_result['corners']:
            # print(f"[DEBUG] Corners available: {len(detection_result['corners'])}")
        
        return detection_result

    @staticmethod
    def draw_rotated_box(image: np.ndarray, corners: List[List[float]], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Vẽ rotated bounding box lên ảnh bằng cách sử dụng 4 corners.
        
        Args:
            image: Ảnh để vẽ lên
            corners: List các corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            color: Màu của box (B, G, R)
            thickness: Độ dày của đường viền
            
        Returns:
            np.ndarray: Ảnh đã được vẽ box
        """
        # Chuyển corners thành numpy array và đảm bảo kiểu int
        pts = np.array(corners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Vẽ polygon (rotated box)
        cv2.polylines(image, [pts], True, color, thickness)
        
        return image

    @staticmethod
    def draw_rotated_boxes(image: np.ndarray, corners_list: List[List[List[float]]], 
                          colors: Optional[List[Tuple[int, int, int]]] = None, 
                          thickness: int = 2) -> np.ndarray:
        """
        Vẽ nhiều rotated bounding boxes lên ảnh.
        
        Args:
            image: Ảnh để vẽ lên
            corners_list: List các corners cho mỗi box
            colors: List màu sắc cho mỗi box, nếu None thì dùng màu mặc định
            thickness: Độ dày của đường viền
            
        Returns:
            np.ndarray: Ảnh đã được vẽ boxes
        """
        if colors is None:
            # Tạo list màu mặc định
            default_colors = [
                (0, 255, 0),    # Xanh lá
                (255, 0, 0),    # Đỏ
                (0, 0, 255),    # Xanh dương
                (255, 255, 0),  # Vàng
                (255, 0, 255),  # Tím
                (0, 255, 255),  # Cyan
            ]
            colors = [default_colors[i % len(default_colors)] for i in range(len(corners_list))]
        
        result_image = image.copy()
        for i, corners in enumerate(corners_list):
            color = colors[i] if i < len(colors) else (0, 255, 0)
            result_image = YOLOTensorRT.draw_rotated_box(result_image, corners, color, thickness)
        
        return result_image


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
        
        # Vẽ rotated bounding boxes nếu có corners
        if detections['corners']:
            rotated_frame = frame.copy()
            rotated_frame = YOLOTensorRT.draw_rotated_boxes(rotated_frame, detections['corners'])
            cv2.imshow("Rotated Bounding Boxes", rotated_frame)
        
        # In ra thông tin về các đối tượng phát hiện được
        print(f"\n=== THÔNG TIN PHÁT HIỆN ===")
        print(f"Format type: {detections['format_type']}")
        print(f"Số đối tượng phát hiện: {len(detections['bounding_boxes'])}")
        
        if detections['obb_boxes']:
            print(f"OBB boxes: {len(detections['obb_boxes'])}")
            for i, obb in enumerate(detections['obb_boxes'][:3]):  # In 3 box đầu tiên
                print(f"  Box {i+1}: center=({obb[0]:.1f}, {obb[1]:.1f}), size=({obb[2]:.1f}x{obb[3]:.1f}), angle={obb[4]:.2f}rad")
        
        if detections['corners']:
            print(f"Corners available: {len(detections['corners'])}")
            for i, corners in enumerate(detections['corners'][:3]):  # In 3 corners đầu tiên
                print(f"  Box {i+1} corners: {corners}")
            
        if detections['scores']:
            print(f"Confidence scores: {[f'{score:.2f}' for score in detections['scores'][:5]]}")  # In 5 score đầu tiên
        
        cv2.waitKey(0)
        cv2.destroyAllWindows() 