"""
Cung cấp các công cụ xử lý sau khi phát hiện đối tượng.
"""
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
import logging


class PostProcessor:
    """
    Lớp xử lý sau khi phát hiện đối tượng, bao gồm lọc, làm mịn và phát hiện va chạm.
    """
    
    def __init__(self, alpha: float = 0.2, conf_threshold: float = 0.85,
                min_area: float = 0.02, max_area: float = 0.8,
                wh_ratio_range: Tuple[float, float] = (0.5, 2.5),
                enable_geometry_filter: bool = True,
                debug_mode: bool = False):
        """
        Khởi tạo bộ xử lý sau phát hiện.
        
        Args:
            alpha: Hệ số cho thuật toán làm mịn EMA (Exponential Moving Average)
            conf_threshold: Ngưỡng tin cậy để phát hiện va chạm
            min_area: Diện tích tối thiểu (tỷ lệ so với ảnh) cho bounding box hợp lệ
            max_area: Diện tích tối đa (tỷ lệ so với ảnh) cho bounding box hợp lệ
            wh_ratio_range: Khoảng tỷ lệ width/height hợp lệ (min, max)
            enable_geometry_filter: Bật/tắt bộ lọc hình học
            debug_mode: Chế độ debug để in thông tin chi tiết
        """
        self.alpha = alpha
        self.conf_threshold = conf_threshold
        self.previous_boxes = None
        
        # Các ràng buộc hình học tối ưu cho phát hiện pallet
        self.min_area = min_area      # % diện tích ảnh
        self.max_area = max_area      # % diện tích ảnh
        self.wh_ratio_range = wh_ratio_range  # Tỷ lệ W/H phù hợp
        self.enable_geometry_filter = enable_geometry_filter
        self.debug_mode = debug_mode
        
        # Cài đặt logger
        self.logger = logging.getLogger("PostProcessor")
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    def filter_by_geometry(self, boxes: np.ndarray, frame_resolution: Tuple[int, int]) -> Tuple[np.ndarray, List[int]]:
        """
        Lọc boxes dựa trên các đặc trưng hình học.
        
        Args:
            boxes: Mảng boxes với shape (N, 5) ở định dạng [xc, yc, w, h, theta]
            frame_resolution: (width, height) của khung hình (pixel)
        
        Returns:
            Tuple: (filtered_boxes, valid_indices)
                - filtered_boxes: Các boxes hợp lệ sau khi lọc
                - valid_indices: Chỉ số của các boxes hợp lệ
        """
        valid_indices = []
        img_area = frame_resolution[0] * frame_resolution[1]
        
        if self.debug_mode:
            self.logger.debug(f"Image area: {img_area} pixels")
            
        for i, box in enumerate(boxes):
            xc, yc, w, h, theta = box
            
            # Tính kích thước thực sau khi xoay
            cos_t = np.abs(np.cos(theta))
            sin_t = np.abs(np.sin(theta))
            effective_w = w * cos_t + h * sin_t
            effective_h = w * sin_t + h * cos_t

            normalized_area = (effective_w * effective_h) / img_area
            wh_ratio = effective_w / effective_h if effective_h > 0 else 0

            # Thiết lập ngưỡng động dựa trên góc xoay
            rotation_factor = np.abs(theta) / np.pi  # từ 0 đến 1
            dynamic_ratio_min = max(0.1, self.wh_ratio_range[0] * (1 - rotation_factor))
            dynamic_ratio_max = min(10.0, self.wh_ratio_range[1] * (1 + rotation_factor))
            
            # Quyết định giữ lại box hay không
            keep = True
            
            if self.enable_geometry_filter:
                if normalized_area < self.min_area:
                    if self.debug_mode:
                        self.logger.debug(f"Box {i} rejected: NormArea {normalized_area:.4f} < min {self.min_area}")
                    keep = False
                elif normalized_area > self.max_area:
                    if self.debug_mode:
                        self.logger.debug(f"Box {i} rejected: NormArea {normalized_area:.4f} > max {self.max_area}")
                    keep = False
                elif wh_ratio < dynamic_ratio_min:
                    if self.debug_mode:
                        self.logger.debug(f"Box {i} rejected: Ratio {wh_ratio:.2f} < min {dynamic_ratio_min:.2f}")
                    keep = False
                elif wh_ratio > dynamic_ratio_max:
                    if self.debug_mode:
                        self.logger.debug(f"Box {i} rejected: Ratio {wh_ratio:.2f} > max {dynamic_ratio_max:.2f}")
                    keep = False
            
            if keep:
                valid_indices.append(i)
                if self.debug_mode:
                    self.logger.debug(f"Box {i} kept: NormArea {normalized_area:.4f}, Ratio {wh_ratio:.2f}, Theta {np.rad2deg(theta):.1f}°")
        
        # Lấy các boxes còn lại
        if len(valid_indices) > 0:
            filtered_boxes = boxes[valid_indices]
        else:
            filtered_boxes = np.array([])
            
        return filtered_boxes, valid_indices

    def smooth_boxes(self, current_boxes: np.ndarray) -> np.ndarray:
        """
        Làm mịn (smoothing) các boxes dùng EMA (Exponential Moving Average).
        
        Args:
            current_boxes: Boxes hiện tại với shape (N,5) hoặc bất kỳ
        
        Returns:
            np.ndarray: Boxes đã được làm mịn
        """
        # Kiểm tra nếu không có boxes trước đó hoặc kích thước không khớp
        if self.previous_boxes is None or current_boxes.shape != self.previous_boxes.shape:
            self.previous_boxes = current_boxes.copy()
            return current_boxes
        
        # Làm mịn với EMA
        try:
            # Giới hạn giá trị trong khoảng phù hợp để tránh vấn đề số học
            current_boxes_limited = np.clip(current_boxes, 0, 1)
            previous_boxes_limited = np.clip(self.previous_boxes, 0, 1)
            
            # Áp dụng công thức EMA
            smoothed = self.alpha * current_boxes_limited + (1 - self.alpha) * previous_boxes_limited
            
            # Lưu lại kết quả
            self.previous_boxes = smoothed.copy()
            return smoothed
        except Exception as e:
            self.logger.error(f"Lỗi khi làm mịn boxes: {str(e)}")
            return current_boxes

    def detect_collisions(self, obb: Any, frame_resolution: Tuple[int, int], 
                          confs: np.ndarray, valid_indices: Optional[List[int]] = None) -> List[Tuple[int, int]]:
        """
        Phát hiện va chạm giữa các box dựa trên thuật toán SAT.
        
        Args:
            obb: Đối tượng OBB từ kết quả YOLO, phải có thuộc tính xyxyxyxy
            frame_resolution: (width, height) của khung hình (pixel)
            confs: Mảng confidence scores tương ứng với các boxes
            valid_indices: Danh sách chỉ số các box hợp lệ (từ filter_by_geometry)
        
        Returns:
            List[Tuple[int, int]]: Danh sách các cặp chỉ số các box có va chạm
        """
        collisions = []
        polygons = []
        
        try:
            # Lấy mảng tọa độ 8 giá trị từ obb và chuyển thành mảng đa giác 4x2
            poly_array = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, "cpu") else obb.xyxyxyxy
            
            for poly in poly_array:
                # Chuyển flat array (8 giá trị) thành 4 điểm (4x2)
                polygons.append(np.array(poly).reshape(4, 2))
            
            # Nếu có valid_indices, chỉ lấy các box hợp lệ
            if valid_indices is not None:
                polygons = [polygons[i] for i in valid_indices]
                confs = confs[valid_indices] if len(valid_indices) > 0 else np.array([])
                
            # Phát hiện va chạm giữa các cặp đa giác
            for i in range(len(polygons)):
                for j in range(i + 1, len(polygons)):
                    # Kiểm tra giao nhau và đủ điều kiện confidence
                    if (self._sat_intersect(polygons[i], polygons[j]) and 
                        confs[i] >= self.conf_threshold and 
                        confs[j] >= self.conf_threshold):
                        collisions.append((i, j))
                        if self.debug_mode:
                            self.logger.debug(f"Va chạm phát hiện giữa box {i} và {j}")
        except Exception as e:
            self.logger.error(f"Lỗi khi phát hiện va chạm: {str(e)}")
            
        return collisions

    def _sat_intersect(self, poly1: np.ndarray, poly2: np.ndarray) -> bool:
        """
        Triển khai thuật toán Separating Axis Theorem (SAT) cho 2 đa giác lồi.
        
        Args:
            poly1, poly2: Mảng với shape (4,2) chứa tọa độ các đỉnh
        
        Returns:
            bool: True nếu có giao nhau, False nếu không
        """
        # Lặp qua từng cạnh của từng đa giác
        for polygon in [poly1, poly2]:
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]
                
                # Tính vector pháp tuyến
                normal = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
                
                # Chiếu các đỉnh của 2 đa giác lên đường pháp tuyến
                proj1 = poly1 @ normal
                proj2 = poly2 @ normal
                
                # Nếu có trục phân tách (không giao nhau trên trục này)
                if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                    return False
        
        # Không tìm thấy trục phân tách nào => có giao nhau
        return True

    def convert_to_xyxy(self, obb: Any) -> np.ndarray:
        """
        Chuyển đổi OBB sang định dạng axis-aligned XYXY chuẩn.
        
        Args:
            obb: Đối tượng OBB chứa thuộc tính xyxy
        
        Returns:
            np.ndarray: Mảng các box theo định dạng XYXY [x1, y1, x2, y2]
        """
        try:
            return obb.xyxy.cpu().numpy() if hasattr(obb.xyxy, "cpu") else obb.xyxy
        except Exception as e:
            self.logger.error(f"Lỗi khi chuyển đổi sang XYXY: {str(e)}")
            return np.array([])
            
    def process_detections(self, 
                           detection_result: Dict[str, Any], 
                           frame_resolution: Tuple[int, int]) -> Dict[str, Any]:
        """
        Xử lý kết quả phát hiện đầy đủ, bao gồm lọc, làm mịn và phát hiện va chạm.
        
        Args:
            detection_result: Kết quả phát hiện từ YOLOInference
            frame_resolution: (width, height) của khung hình
            
        Returns:
            Dict: Kết quả phát hiện đã được xử lý, bao gồm:
                - 'bounding_boxes': Danh sách các bounding box sau xử lý
                - 'filtered_indices': Chỉ số các box hợp lệ
                - 'collisions': Danh sách các cặp box va chạm
                - ... (các thông tin khác từ detection_result)
        """
        result = detection_result.copy()
        
        try:
            # Lấy thông tin cần thiết
            obb = result.get('obb')
            scores = np.array(result.get('scores', []))
            
            if obb is not None and hasattr(obb, 'xywhr'):
                # Lấy mảng xywhr
                boxes = obb.xywhr.cpu().numpy() if hasattr(obb.xywhr, 'cpu') else obb.xywhr
                
                # Lọc theo hình học
                filtered_boxes, valid_indices = self.filter_by_geometry(boxes, frame_resolution)
                
                # Làm mịn boxes nếu có
                if filtered_boxes.size > 0:
                    smoothed_boxes = self.smooth_boxes(filtered_boxes)
                else:
                    smoothed_boxes = filtered_boxes
                
                # Phát hiện va chạm
                collisions = self.detect_collisions(obb, frame_resolution, scores, valid_indices)
                
                # Bổ sung thông tin vào kết quả
                result['filtered_boxes'] = smoothed_boxes.tolist() if isinstance(smoothed_boxes, np.ndarray) else smoothed_boxes
                result['filtered_indices'] = valid_indices
                result['collisions'] = collisions
        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý kết quả phát hiện: {str(e)}")
        
        return result


if __name__ == "__main__":
    # Class giả lập OBB để kiểm thử
    class MockOBB:
        def __init__(self, xywhr, xyxy, xyxyxyxy, conf):
            self.xywhr = xywhr
            self.xyxy = xyxy
            self.xyxyxyxy = xyxyxyxy
            self.conf = conf
    
    # Tạo processor cho việc kiểm thử
    processor = PostProcessor(
        alpha=0.4, 
        conf_threshold=0.65,
        debug_mode=True
    )
    
    # Tạo test data cho 3 boxes ở định dạng [xc, yc, w, h, theta]
    test_xywhr = np.array([
        [0.5, 0.5, 0.2, 0.3, np.deg2rad(45)],
        [0.5, 0.5, 0.2, 0.3, np.deg2rad(-45)],
        [0.1, 0.1, 0.05, 0.05, 0.0]
    ])
    
    # Tạo dữ liệu giả lập xyxy
    test_xyxy = np.array([
        [0.4, 0.4, 0.6, 0.6],
        [0.4, 0.4, 0.6, 0.6],
        [0.05, 0.05, 0.15, 0.15]
    ])
    
    # Tạo dữ liệu giả lập xyxyxyxy (8 điểm cho 4 góc của mỗi box)
    test_xyxyxyxy = np.array([
        [0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.6],  # box 1
        [0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.6],  # box 2
        [0.05, 0.05, 0.15, 0.05, 0.15, 0.15, 0.05, 0.15]  # box 3
    ])
    
    # Các confidence scores
    test_conf = np.array([0.9, 0.8, 0.7])
    
    # Tạo đối tượng OBB mô phỏng
    test_obb = MockOBB(test_xywhr, test_xyxy, test_xyxyxyxy, test_conf)
    
    print("=== KIỂM THỬ BỘ XỬ LÝ SAU PHÁT HIỆN ===")
    
    # Test filter_by_geometry
    filtered, valid_indices = processor.filter_by_geometry(test_xywhr, frame_resolution=(640, 480))
    print(f"\nĐã giữ lại {len(filtered)}/{len(test_xywhr)} boxes sau khi lọc hình học")
    
    # Test smooth_boxes
    smoothed = processor.smooth_boxes(filtered)
    print(f"\nBoxes sau khi làm mịn:\n{smoothed}")
    
    # Test detect_collisions
    collisions = processor.detect_collisions(test_obb, frame_resolution=(640, 480), 
                                             confs=test_conf, valid_indices=valid_indices)
    print(f"\nVa chạm phát hiện được: {collisions}")
    
    # Test chuyển đổi sang XYXY
    xyxy_boxes = processor.convert_to_xyxy(test_obb)
    print(f"\nChuyển đổi sang XYXY boxes:\n{xyxy_boxes}")
    
    # Test xử lý toàn bộ detections
    mock_detection_result = {
        'bounding_boxes': test_xyxy.tolist(),
        'scores': test_conf.tolist(),
        'classes': [0, 0, 0],
        'obb': test_obb
    }
    
    processed_result = processor.process_detections(mock_detection_result, (640, 480))
    print(f"\nKết quả xử lý đầy đủ:\n{processed_result.keys()}") 