"""
Module chia pallet thành các vùng nhỏ hơn để depth estimation.
Đặt giữa YOLO detection và depth estimation trong pipeline.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import math
import os
import json
from datetime import datetime


class ModuleDivision:
    """
    Lớp chia pallet thành các vùng con đơn giản.
    """
    
    def __init__(self, debug: bool = False):
        """Khởi tạo Module Division."""
        self.debug = debug
        
        # ⭐ ORIENTATION LOCK SYSTEM ⭐
        self.orientation_locked = False
        self.locked_orientation_data = None
        self.orientation_lock_file = "orientation_lock.json"
        
        # Auto-load saved orientation lock if exists
        self.load_orientation_lock()
    
    def divide_module_1(self, pallet_bbox: List[float], layer: int = 1) -> List[Dict[str, Any]]:
        """
        Chia pallet theo Module 1:
        - Layer 1: Chia 3 phần bằng nhau theo chiều rộng (3 cột x 1 hàng)
        - Layer 2: Chia 3 phần bằng nhau theo chiều dài (1 cột x 3 hàng)
        
        Args:
            pallet_bbox: [x1, y1, x2, y2] của pallet
            layer: Lớp cần chia (1 hoặc 2)
            
        Returns:
            List[Dict]: Danh sách các vùng con với thông tin tọa độ và orientation
        """
        if layer not in [1, 2]:
            raise ValueError("Layer phải là 1 hoặc 2")
        
        x1, y1, x2, y2 = pallet_bbox
        width = x2 - x1
        height = y2 - y1
        
        # Đối với regular bbox, orientation luôn là 0° (không xoay)
        pallet_orientation = 0.0
        
        regions = []
        
        if layer == 1:
            # Layer 1: Chia 3 phần theo chiều rộng (3 cột x 1 hàng)
            part_width = width / 3
            
            for i in range(3):
                # Tính tọa độ từng vùng
                region_x1 = x1 + i * part_width
                region_y1 = y1
                region_x2 = x1 + (i + 1) * part_width
                region_y2 = y2
                
                # Tính tâm vùng
                center_x = (region_x1 + region_x2) / 2
                center_y = (region_y1 + region_y2) / 2
                
                regions.append({
                    'region_id': i + 1,  # 1, 2, 3
                    'bbox': [region_x1, region_y1, region_x2, region_y2],
                    'center': [center_x, center_y],
                    'layer': layer,
                    'module': 1,
                    'orientation': pallet_orientation,  # Regular bbox có orientation = 0°
                    'pallet_orientation': pallet_orientation
                })
        
        elif layer == 2:
            # Layer 2: Chia 3 phần theo chiều dài (1 cột x 3 hàng)
            part_height = height / 3
            
            for i in range(3):
                # Tính tọa độ từng vùng
                region_x1 = x1
                region_y1 = y1 + i * part_height
                region_x2 = x2
                region_y2 = y1 + (i + 1) * part_height
                
                # Tính tâm vùng
                center_x = (region_x1 + region_x2) / 2
                center_y = (region_y1 + region_y2) / 2
                
                regions.append({
                    'region_id': i + 1,  # 1, 2, 3
                    'bbox': [region_x1, region_y1, region_x2, region_y2],
                    'center': [center_x, center_y],
                    'layer': layer,
                    'module': 1,
                    'orientation': pallet_orientation,  # Regular bbox có orientation = 0°
                    'pallet_orientation': pallet_orientation
                })
        
        return regions
    
    def divide_obb_module_1(self, pallet_corners: List[List[float]], layer: int = 1) -> List[Dict[str, Any]]:
        """
        Chia pallet OBB theo Module 1 với orientation lock support:
        - Layer 1: Chia 3 phần theo chiều dài thực tế (cạnh 12cm)  
        - Layer 2: Chia 3 phần theo chiều rộng thực tế (cạnh 10cm)
        
        Args:
            pallet_corners: Danh sách 4 góc của pallet OBB
            layer: Lớp cần chia (1 hoặc 2)
            
        Returns:
            List[Dict]: Danh sách các vùng con với corners và thông tin chi tiết
        """
        if self.debug:
            print(f"  [divide_obb] layer={layer}, corners count={len(pallet_corners)}")
        
        if layer not in [1, 2]:
            raise ValueError("Layer phải là 1 hoặc 2")
        
        # Convert to numpy array để tính toán
        corners_array = np.array(pallet_corners)
        
        # ⭐ ORIENTATION LOCK CHECK ⭐
        if self.orientation_locked and self.locked_orientation_data:
            # Sử dụng locked orientation thay vì detect
            locked_layer = self.locked_orientation_data.get('layer')
            if locked_layer == layer:
                # Find matching pallet trong locked data (giả sử pallet đầu tiên)
                locked_pallets = self.locked_orientation_data.get('pallets', [])
                if locked_pallets:
                    locked_pallet = locked_pallets[0]  # Use first pallet as reference
                    long_edge_is_edge1 = locked_pallet['long_edge_is_edge1']
                    pallet_orientation = locked_pallet['orientation']
                    
                    if self.debug:
                        print(f"    🔒 Using LOCKED orientation: {pallet_orientation:.1f}°, strategy: {locked_pallet['division_strategy']}")
                else:
                    # Fallback to detection
                    long_edge_is_edge1, pallet_orientation = self._detect_pallet_orientation(corners_array)
                    if self.debug:
                        print(f"    🔄 Locked data empty, fallback to detection")
            else:
                # Layer mismatch, fallback to detection
                long_edge_is_edge1, pallet_orientation = self._detect_pallet_orientation(corners_array)
                if self.debug:
                    print(f"    🔄 Layer mismatch (locked:{locked_layer} vs current:{layer}), fallback to detection")
        else:
            # Normal detection
            long_edge_is_edge1, pallet_orientation = self._detect_pallet_orientation(corners_array)
            if self.debug:
                print(f"    🔍 Auto-detection mode")
        
        # ⭐ TIẾP TỤC XỬ LÝ VỚI ORIENTATION ĐÃ XÁC ĐỊNH ⭐
        if self.debug:
            print(f"    orientation={pallet_orientation:.1f}°, edge_ratio={long_edge_is_edge1}")
        
        # Tìm corners theo thứ tự: top-left, top-right, bottom-right, bottom-left
        sorted_by_y = corners_array[np.argsort(corners_array[:, 1])]
        top_points = sorted_by_y[:2]  # 2 điểm trên cùng
        bottom_points = sorted_by_y[2:]  # 2 điểm dưới cùng
        
        # Sắp xếp các điểm trên và dưới theo x
        top_left, top_right = top_points[np.argsort(top_points[:, 0])]
        bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
        
        regions = []
        
        if layer == 1:
            # Layer 1: Chia theo CHIỀU DÀI thực tế (cạnh 12cm)
            if long_edge_is_edge1:
                # Cạnh dài là Edge1 (horizontal direction) -> chia theo X
                if self.debug:
                    print(f"    Layer 1: Chia theo cạnh dài (Edge1 - horizontal) → chia theo X")
                    
                for i in range(3):
                    ratio_start = i / 3.0
                    ratio_end = (i + 1) / 3.0
                    
                    region_corners = self._interpolate_obb_region(
                        top_left, top_right, bottom_left, bottom_right,
                        x_start=ratio_start, x_end=ratio_end,
                        y_start=0.0, y_end=1.0
                    )
                    
                    region_bbox = self._corners_to_bbox(region_corners)
                    region_center = self._corners_to_center(region_corners)
                    
                    regions.append({
                        'region_id': i + 1,
                        'bbox': region_bbox,
                        'center': region_center,
                        'corners': region_corners,
                        'layer': layer,
                        'module': 1,
                        'division_direction': 'along_long_edge_X',
                        'rule': 'layer1_along_12cm_edge',
                        'orientation': pallet_orientation,  # Regions kế thừa orientation từ pallet gốc
                        'pallet_orientation': pallet_orientation
                    })
            else:
                # Cạnh dài là Edge2 (vertical direction) -> chia theo Y
                if self.debug:
                    print(f"    Layer 1: Chia theo cạnh dài (Edge2 - vertical) → chia theo Y")
                    
                for i in range(3):
                    ratio_start = i / 3.0
                    ratio_end = (i + 1) / 3.0
                    
                    region_corners = self._interpolate_obb_region(
                        top_left, top_right, bottom_left, bottom_right,
                        x_start=0.0, x_end=1.0,
                        y_start=ratio_start, y_end=ratio_end
                    )
                    
                    region_bbox = self._corners_to_bbox(region_corners)
                    region_center = self._corners_to_center(region_corners)
                    
                    regions.append({
                        'region_id': i + 1,
                        'bbox': region_bbox,
                        'center': region_center,
                        'corners': region_corners,
                        'layer': layer,
                        'module': 1,
                        'division_direction': 'along_long_edge_Y',
                        'rule': 'layer1_along_12cm_edge',
                        'orientation': pallet_orientation,  # Regions kế thừa orientation từ pallet gốc
                        'pallet_orientation': pallet_orientation
                    })
        
        elif layer == 2:
            # Layer 2: Chia theo CHIỀU NGANG thực tế (cạnh 10cm)
            if not long_edge_is_edge1:
                # Cạnh ngắn là Edge1 (horizontal direction) -> chia theo X
                if self.debug:
                    print(f"    Layer 2: Chia theo cạnh ngắn (Edge1 - horizontal) → chia theo X")
                    
                for i in range(3):
                    ratio_start = i / 3.0
                    ratio_end = (i + 1) / 3.0
                    
                    region_corners = self._interpolate_obb_region(
                        top_left, top_right, bottom_left, bottom_right,
                        x_start=ratio_start, x_end=ratio_end,
                        y_start=0.0, y_end=1.0
                    )
                    
                    region_bbox = self._corners_to_bbox(region_corners)
                    region_center = self._corners_to_center(region_corners)
                    
                    regions.append({
                        'region_id': i + 1,
                        'bbox': region_bbox,
                        'center': region_center,
                        'corners': region_corners,
                        'layer': layer,
                        'module': 1,
                        'division_direction': 'along_short_edge_X',
                        'rule': 'layer2_along_10cm_edge',
                        'orientation': pallet_orientation,  # Regions kế thừa orientation từ pallet gốc
                        'pallet_orientation': pallet_orientation
                    })
            else:
                # Cạnh ngắn là Edge2 (vertical direction) -> chia theo Y
                if self.debug:
                    print(f"    Layer 2: Chia theo cạnh ngắn (Edge2 - vertical) → chia theo Y")
                    
                for i in range(3):
                    ratio_start = i / 3.0
                    ratio_end = (i + 1) / 3.0
                    
                    region_corners = self._interpolate_obb_region(
                        top_left, top_right, bottom_left, bottom_right,
                        x_start=0.0, x_end=1.0,
                        y_start=ratio_start, y_end=ratio_end
                    )
                    
                    region_bbox = self._corners_to_bbox(region_corners)
                    region_center = self._corners_to_center(region_corners)
                    
                    regions.append({
                        'region_id': i + 1,
                        'bbox': region_bbox,
                        'center': region_center,
                        'corners': region_corners,
                        'layer': layer,
                        'module': 1,
                        'division_direction': 'along_short_edge_Y',
                        'rule': 'layer2_along_10cm_edge',
                        'orientation': pallet_orientation,  # Regions kế thừa orientation từ pallet gốc
                        'pallet_orientation': pallet_orientation
                    })
        
        if self.debug:
            print(f"    Created {len(regions)} regions: " + ", ".join([f"R{r['region_id']}" for r in regions]))
        
        return regions
    
    def _detect_pallet_orientation(self, corners_array: np.ndarray) -> Tuple[bool, float]:
        """
        Detect pallet orientation và edge ratios
        
        Args:
            corners_array: Numpy array của corners
            
        Returns:
            Tuple[bool, float]: (long_edge_is_edge1, pallet_orientation)
        """
        # Tính edge lengths và orientation
        edge1 = corners_array[1] - corners_array[0]  # Cạnh từ corner 0 -> 1
        edge2 = corners_array[3] - corners_array[0]  # Cạnh từ corner 0 -> 3
        
        edge1_length = np.linalg.norm(edge1)
        edge2_length = np.linalg.norm(edge2)
        
        # Tính orientation (góc của cạnh đầu tiên so với trục X)
        pallet_orientation = math.degrees(math.atan2(edge1[1], edge1[0]))
        
        # Chuẩn hóa góc về range [-180, 180]
        while pallet_orientation > 180:
            pallet_orientation -= 360
        while pallet_orientation <= -180:
            pallet_orientation += 360
        
        # Xác định cạnh nào dài hơn (giả định: cạnh dài = 12cm, cạnh ngắn = 10cm)
        long_edge_is_edge1 = edge1_length > edge2_length
        
        return long_edge_is_edge1, pallet_orientation
    
    def _calculate_pallet_orientation(self, pallet_corners: List[List[float]]) -> float:
        """
        Tính orientation (góc xoay) của pallet dựa trên corners.
        
        Args:
            pallet_corners: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] corners của pallet OBB
            
        Returns:
            float: Góc orientation của pallet (degrees) theo hệ tọa độ custom
        """
        corners_array = np.array(pallet_corners)
        
        # Tính 2 cạnh kề nhau để xác định cạnh dài (trục chính)
        edge1 = corners_array[1] - corners_array[0]  # Cạnh từ corner 0 -> 1
        edge2 = corners_array[2] - corners_array[1]  # Cạnh từ corner 1 -> 2
        
        edge1_length = np.linalg.norm(edge1)
        edge2_length = np.linalg.norm(edge2)
        
        # Chọn cạnh dài làm trục chính (major axis)
        if edge1_length > edge2_length:
            major_axis_vector = edge1
        else:
            major_axis_vector = edge2
        
        # Tính góc của trục chính theo hệ tọa độ custom
        # Custom coordinate: X+ = E→W (left), Y+ = N→S (down)
        angle_rad = np.arctan2(-major_axis_vector[1], -major_axis_vector[0])  # Đảo dấu để phù hợp với hệ tọa độ custom
        angle_deg = np.degrees(angle_rad)
        
        # Chuẩn hóa góc về [-180, 180]
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg <= -180:
            angle_deg += 360
            
        if self.debug:
            print(f"[ModuleDivision] Pallet orientation calculation:")
            print(f"  Major axis vector: ({major_axis_vector[0]:.1f}, {major_axis_vector[1]:.1f})")
            print(f"  Calculated angle: {angle_deg:.1f}°")
        
        return angle_deg
    
    def _interpolate_obb_region(self, top_left: np.ndarray, top_right: np.ndarray, 
                               bottom_left: np.ndarray, bottom_right: np.ndarray,
                               x_start: float, x_end: float, y_start: float, y_end: float) -> List[List[float]]:
        """
        Tính corners của một region trong OBB bằng bilinear interpolation.
        
        Args:
            top_left, top_right, bottom_left, bottom_right: Corners của OBB gốc
            x_start, x_end: Tỷ lệ bắt đầu và kết thúc theo trục x (0.0 - 1.0)
            y_start, y_end: Tỷ lệ bắt đầu và kết thúc theo trục y (0.0 - 1.0)
            
        Returns:
            List[List[float]]: Corners của region [[x,y], [x,y], [x,y], [x,y]]
        """
        # Tính các điểm interpolation
        # Top edge points
        top_start = top_left + x_start * (top_right - top_left)
        top_end = top_left + x_end * (top_right - top_left)
        
        # Bottom edge points
        bottom_start = bottom_left + x_start * (bottom_right - bottom_left)
        bottom_end = bottom_left + x_end * (bottom_right - bottom_left)
        
        # Region corners bằng interpolation theo y
        region_top_left = top_start + y_start * (bottom_start - top_start)
        region_top_right = top_end + y_start * (bottom_end - top_end)
        region_bottom_right = top_end + y_end * (bottom_end - top_end)
        region_bottom_left = top_start + y_end * (bottom_start - top_start)
        
        # Trả về theo format [[x,y], [x,y], [x,y], [x,y]]
        return [
            [float(region_top_left[0]), float(region_top_left[1])],      # top-left
            [float(region_top_right[0]), float(region_top_right[1])],    # top-right
            [float(region_bottom_right[0]), float(region_bottom_right[1])], # bottom-right
            [float(region_bottom_left[0]), float(region_bottom_left[1])]   # bottom-left
        ]
    
    def _corners_to_bbox(self, corners: List[List[float]]) -> List[float]:
        """
        Chuyển corners thành bounding box.
        
        Args:
            corners: [[x,y], [x,y], [x,y], [x,y]]
            
        Returns:
            List[float]: [x1, y1, x2, y2]
        """
        corners_array = np.array(corners)
        min_x = np.min(corners_array[:, 0])
        min_y = np.min(corners_array[:, 1])
        max_x = np.max(corners_array[:, 0])
        max_y = np.max(corners_array[:, 1])
        return [float(min_x), float(min_y), float(max_x), float(max_y)]
    
    def _corners_to_center(self, corners: List[List[float]]) -> List[float]:
        """
        Tính center từ corners.
        
        Args:
            corners: [[x,y], [x,y], [x,y], [x,y]]
            
        Returns:
            List[float]: [center_x, center_y]
        """
        corners_array = np.array(corners)
        center_x = np.mean(corners_array[:, 0])
        center_y = np.mean(corners_array[:, 1])
        return [float(center_x), float(center_y)]
    
    def process_pallet_detections(self, detection_result: Dict[str, Any], 
                                 layer: int = 1, target_classes: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Xử lý kết quả detection từ YOLO và chia thành các vùng nhỏ.
        
        Args:
            detection_result: Kết quả từ YOLO detection
            layer: Lớp cần chia
            
        Returns:
            Dict: Kết quả đã được chia với thông tin các vùng con
            {
                'original_detection': Dict,     # Kết quả YOLO gốc
                'divided_regions': List[Dict],  # Các vùng đã chia
                'total_regions': int,           # Tổng số vùng
                'processing_info': Dict         # Thông tin xử lý
            }
        """
        # Xử lý default value cho target_classes
        if target_classes is None:
            target_classes = [0.0, 1.0, 2.0]  # Tạm thời chấp nhận tất cả class để debug
        
        # ⭐ ENHANCED DEBUG: Show input and processing ⭐
        if self.debug:
            print(f"\n[ModuleDivision DEBUG] process_pallet_detections: layer={layer}, target_classes={target_classes}")
        
        result = {
            'original_detection': detection_result,
            'divided_regions': [],
            'total_regions': 0,
            'processing_info': {
                'layer': layer,
                'module': 1,
                'success': False,
                'error': None,
                'target_classes': target_classes,
                'filtered_detection': 0
            }
        }
        
        try:
            # Ưu tiên sử dụng corners (rotated boxes) nếu có, fallback về bounding_boxes
            classes = detection_result.get('classes', [])
            corners_list = detection_result.get('corners', [])
            bounding_boxes = detection_result.get('bounding_boxes', [])
            if self.debug:
                print(f"  Input: classes={len(classes)}, corners={len(corners_list)}, bboxes={len(bounding_boxes)}")            
            #Filter chỉ lấy detection thuộc target_classes
            filtered_corners = []
            filtered_bboxes = []

            if self.debug:
                print(f"  Filtering for target classes: {target_classes}")
            if classes:
                for i, cls in enumerate(classes):
                    if cls in target_classes:
                        if corners_list and i < len(corners_list):
                            filtered_corners.append(corners_list[i])
                        if bounding_boxes and i < len(bounding_boxes):
                            filtered_bboxes.append(bounding_boxes[i])
                        if self.debug:
                            print(f"    Accepted class {cls} at index {i}")
            else:
                #Fallback: Nếu không có thông tin class, lấy tất cả
                filtered_corners = corners_list
                filtered_bboxes = bounding_boxes
                if self.debug:
                    print(f"  No class info, using all detections")
            
            result['processing_info']['filtered_detection'] = len(filtered_corners) + len(filtered_bboxes)
            
            # Kiểm tra xem có corners không
            if filtered_corners and len(filtered_corners) > 0:
                if self.debug:
                    print(f"  Using {len(filtered_corners)} rotated bounding boxes (corners)")
                all_regions = self._process_with_obb_corners(filtered_corners, layer)
            elif filtered_bboxes and len(filtered_bboxes) > 0:
                if self.debug:
                    print(f"  Fallback: Using {len(filtered_bboxes)} regular bounding boxes")
                all_regions = self._process_with_bboxes(filtered_bboxes, layer)
            else:
                if self.debug:
                    print(f"  No corners or bounding_boxes found after filtering")
                all_regions = []
            
            result['divided_regions'] = all_regions
            result['total_regions'] = len(all_regions)
            result['processing_info']['success'] = True
            
            if self.debug:
                print(f"  Final result: success={result['processing_info']['success']}, total_regions={result['total_regions']}")
                region_summary = []
                for region in all_regions:
                    pallet_id = region.get('pallet_id', '?')
                    region_id = region.get('region_id', '?')
                    region_summary.append(f"P{pallet_id}R{region_id}")
                print(f"  Regions: {', '.join(region_summary)}")
            
        except Exception as e:
            result['processing_info']['error'] = str(e)
            print(f"[ModuleDivision ERROR] Lỗi khi chia pallet: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def _process_with_obb_corners(self, corners_list: List[List], layer: int) -> List[Dict[str, Any]]:
        """
        Xử lý detection với rotated bounding boxes (corners) - phiên bản mới chính xác.
        
        Args:
            corners_list: Danh sách các corners từ YOLO
            layer: Lớp cần chia
            
        Returns:
            List[Dict]: Danh sách các vùng đã chia với corners chính xác
        """
        if self.debug:
            print(f"\n[ModuleDivision DEBUG] _process_with_obb_corners: {len(corners_list)} pallets, layer={layer}")
        
        all_regions = []
        
        for pallet_idx, corners in enumerate(corners_list):
            # ⭐ XÁC ĐỊNH PALLET_ID DỰA TRÊN VỊ TRÍ ⭐
            center_x, center_y = self._calculate_pallet_center(corners)
            pallet_id = self._determine_pallet_region_id(center_x)
            
            if self.debug:
                print(f"  Processing pallet {pallet_idx + 1}: center=({center_x:.1f}, {center_y:.1f}) → P{pallet_id}")
            
            # Chia pallet OBB trực tiếp dựa trên corners
            regions = self.divide_obb_module_1(corners, layer=layer)
            
            if self.debug:
                print(f"  Created {len(regions)} regions")
            
            # ⭐ SỬA LOGIC GÁN PALLET_ID ⭐
            for region_idx, region in enumerate(regions):
                region['pallet_id'] = pallet_id  # Sử dụng pallet_id dựa trên vị trí
                region['global_region_id'] = len(all_regions) + 1
                region['original_corners'] = corners  # Lưu corners gốc của pallet
                if self.debug:
                    print(f"    P{region['pallet_id']}R{region.get('region_id', 'MISSING')}")
            
            all_regions.extend(regions)
        
        if self.debug:
            print(f"[ModuleDivision DEBUG] Final result: {len(all_regions)} total regions")
        
        return all_regions
    
    def _process_with_corners(self, corners_list: List[List], layer: int) -> List[Dict[str, Any]]:
        """
        Xử lý detection với rotated bounding boxes (corners) - phiên bản cũ (deprecated).
        Được giữ lại để tương thích ngược.
        
        Args:
            corners_list: Danh sách các corners từ YOLO
            layer: Lớp cần chia
            
        Returns:
            List[Dict]: Danh sách các vùng đã chia
        """
        print("[ModuleDivision] Sử dụng phương thức cũ _process_with_corners (deprecated)")
        return self._process_with_obb_corners(corners_list, layer)
    
    def _process_with_bboxes(self, bounding_boxes: List[List], layer: int) -> List[Dict[str, Any]]:
        """
        Xử lý detection với regular bounding boxes (fallback).
        
        Args:
            bounding_boxes: Danh sách các bounding boxes từ YOLO
            layer: Lớp cần chia
            
        Returns:
            List[Dict]: Danh sách các vùng đã chia
        """
        all_regions = []
        
        for pallet_idx, pallet_bbox in enumerate(bounding_boxes):
            # ⭐ XÁC ĐỊNH PALLET_ID DỰA TRÊN VỊ TRÍ ⭐
            center_x, center_y = self._calculate_bbox_center(pallet_bbox)
            pallet_id = self._determine_pallet_region_id(center_x)
            
            if self.debug:
                print(f"  Processing pallet {pallet_idx + 1}: center=({center_x:.1f}, {center_y:.1f}) → P{pallet_id}")
            
            # Chia pallet thành các vùng nhỏ
            regions = self.divide_module_1(pallet_bbox, layer=layer)
            
            # ⭐ SỬA LOGIC GÁN PALLET_ID ⭐
            for region in regions:
                region['pallet_id'] = pallet_id  # Sử dụng pallet_id dựa trên vị trí
                region['global_region_id'] = len(all_regions) + 1
                # Không có corners cho regular bbox
            
            all_regions.extend(regions)
        
        return all_regions
    
    def prepare_for_depth_estimation(self, divided_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chuẩn bị dữ liệu cho depth estimation.
        
        Args:
            divided_result: Kết quả từ process_pallet_detections
            
        Returns:
            List[Dict]: Danh sách các vùng cần depth estimation
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'center': [x, y],
                    'region_info': Dict,  # Thông tin chi tiết vùng
                    'corners': List (optional) # Corners của region nếu có
                }
            ]
        """
        # print(f"\n[ModuleDivision DEBUG] prepare_for_depth_estimation START:")
        # print(f"  divided_result keys: {list(divided_result.keys())}")
        # print(f"  divided_result total_regions: {divided_result.get('total_regions', 0)}")
        # print(f"  divided_result success: {divided_result.get('processing_info', {}).get('success', False)}")
        
        depth_regions = []
        
        try:
            regions = divided_result.get('divided_regions', [])
            # print(f"  divided_regions count: {len(regions)}")
            
            for i, region in enumerate(regions):
                # print(f"\n  Processing region {i}:")
                # print(f"    region keys: {list(region.keys())}")
                # print(f"    pallet_id: {region.get('pallet_id', 'MISSING')}")
                # print(f"    region_id: {region.get('region_id', 'MISSING')}")
                # print(f"    global_region_id: {region.get('global_region_id', 'MISSING')}")
                # print(f"    center: {region.get('center', 'MISSING')}")
                # print(f"    bbox: {region.get('bbox', 'MISSING')}")
                
                depth_region = {
                    'bbox': region['bbox'],
                    'center': region['center'],
                    'region_info': {
                        'region_id': region['region_id'],
                        'pallet_id': region.get('pallet_id', 1),
                        'global_region_id': region.get('global_region_id', 1),
                        'layer': region['layer'],
                        'module': region['module']
                    },
                    'orientation': region.get('orientation', 0.0),  # Thêm orientation
                    'pallet_orientation': region.get('pallet_orientation', 0.0)  # Thêm pallet orientation
                }
                
                # print(f"    Created depth_region.region_info: {depth_region['region_info']}")
                
                # Thêm corners nếu có (cho rotated boxes)
                if 'corners' in region:
                    depth_region['corners'] = region['corners']
                    # print(f"    Added corners: {len(region['corners'])} points")
                
                # Thêm corners gốc của pallet nếu có
                if 'original_corners' in region:
                    depth_region['original_corners'] = region['original_corners']
                    # print(f"    Added original_corners")
                
                depth_regions.append(depth_region)
                # print(f"    Final depth_region P{depth_region['region_info']['pallet_id']}R{depth_region['region_info']['region_id']}")
        
        except Exception as e:
            print(f"[ModuleDivision ERROR] Lỗi khi chuẩn bị dữ liệu cho depth: {e}")
            import traceback
            traceback.print_exc()
        
        # print(f"\n[ModuleDivision DEBUG] prepare_for_depth_estimation RESULT:")
        # print(f"  depth_regions count: {len(depth_regions)}")
        for i, depth_region in enumerate(depth_regions):
            region_info = depth_region.get('region_info', {})
            pallet_id = region_info.get('pallet_id', 'MISSING')
            region_id = region_info.get('region_id', 'MISSING')
            center = depth_region.get('center', [0, 0])
            # print(f"    depth_region {i}: P{pallet_id}R{region_id} center=({center[0]:.1f}, {center[1]:.1f})")
        
        return depth_regions

    # ⭐ SEQUENCE METHODS - ADDED FOR PLAN IMPLEMENTATION ⭐
    
    def get_regions_by_sequence(self, regions_data: List[Dict], pallet_id: int, 
                               sequence: List[int] = [1, 3, 2]) -> List[Dict]:
        """
        Trích xuất regions theo thứ tự cụ thể cho một pallet.
        
        Args:
            regions_data: List regions từ prepare_for_depth_estimation
            pallet_id: ID của pallet cần lấy regions
            sequence: Thứ tự regions cần lấy (default: [1, 3, 2])
            
        Returns:
            List[Dict]: Danh sách regions theo thứ tự sequence
        """
        pallet_regions = [r for r in regions_data 
                         if r.get('region_info', {}).get('pallet_id') == pallet_id]
        
        ordered_regions = []
        for seq_id in sequence:
            matching_region = self.get_specific_region(regions_data, pallet_id, seq_id)
            if matching_region:
                ordered_regions.append(matching_region)
        
        if self.debug:
            print(f"[ModuleDivision] Pallet {pallet_id} sequence {sequence}: Found {len(ordered_regions)} regions")
        
        return ordered_regions
    
    def get_next_available_region(self, regions_data: List[Dict], pallet_id: int, 
                                 sequence: List[int] = [1, 3, 2]) -> Optional[Dict]:
        """
        Lấy region tiếp theo có sẵn theo sequence cho một pallet.
        
        Args:
            regions_data: List regions từ prepare_for_depth_estimation
            pallet_id: ID của pallet
            sequence: Thứ tự regions (default: [1, 3, 2])
            
        Returns:
            Dict: Region tiếp theo hoặc None nếu không có
        """
        ordered_regions = self.get_regions_by_sequence(regions_data, pallet_id, sequence)
        
        # Trả về region đầu tiên có sẵn
        if ordered_regions:
            return ordered_regions[0]
        
        return None
    
    def get_specific_region(self, regions_data: List[Dict], pallet_id: int, region_id: int) -> Optional[Dict]:
        """
        Lấy region cụ thể bằng pallet_id và region_id.
        
        Args:
            regions_data: List regions từ prepare_for_depth_estimation
            pallet_id: ID của pallet
            region_id: ID của region (1, 2, 3)
            
        Returns:
            Dict: Region data hoặc None nếu không tìm thấy
        """
        for region in regions_data:
            region_info = region.get('region_info', {})
            if (region_info.get('pallet_id') == pallet_id and 
                region_info.get('region_id') == region_id):
                return region
        
        if self.debug:
            print(f"[ModuleDivision] Không tìm thấy Pallet {pallet_id} Region {region_id}")
        
        return None
    
    def get_all_pallets_sequence(self, regions_data: List[Dict], 
                                sequence: List[int] = [1, 3, 2]) -> Dict[int, List[Dict]]:
        """
        Lấy sequence cho tất cả pallets.
        
        Args:
            regions_data: List regions từ prepare_for_depth_estimation
            sequence: Thứ tự regions (default: [1, 3, 2])
            
        Returns:
            Dict[pallet_id, List[regions]]: Mapping pallet_id -> ordered regions
        """
        # Tìm tất cả pallet IDs
        pallet_ids = set()
        for region in regions_data:
            pallet_id = region.get('region_info', {}).get('pallet_id')
            if pallet_id:
                pallet_ids.add(pallet_id)
        
        # Lấy sequence cho từng pallet
        result = {}
        for pallet_id in sorted(pallet_ids):
            result[pallet_id] = self.get_regions_by_sequence(regions_data, pallet_id, sequence)
        
        if self.debug:
            print(f"[ModuleDivision] All pallets sequence: {len(result)} pallets found")
            for pid, regions in result.items():
                print(f"  Pallet {pid}: {len(regions)} regions in sequence {sequence}")
        
        return result

    def lock_current_orientation(self, pallet_corners_list: List[List], layer: int, save_to_file: bool = True):
        """
        🔒 Lock orientation hiện tại để sử dụng cho tất cả frames tiếp theo
        
        Args:
            pallet_corners_list: List corners của các pallets hiện tại
            layer: Layer hiện tại (1 hoặc 2)
            save_to_file: Có lưu vào file không
        """
        if not pallet_corners_list:
            print(f"[ModuleDivision] ❌ Không có pallet corners để lock!")
            return False
        
        # Analyze orientation for each pallet
        orientation_data = {
            'layer': layer,
            'locked_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'pallets': []
        }
        
        for i, corners in enumerate(pallet_corners_list):
            pallet_orientation = self._calculate_pallet_orientation(corners)
            
            # Determine division strategy based on current analysis
            corners_array = np.array(corners)
            edge1 = corners_array[1] - corners_array[0]
            edge2 = corners_array[3] - corners_array[0]
            edge1_length = np.linalg.norm(edge1)
            edge2_length = np.linalg.norm(edge2)
            long_edge_is_edge1 = edge1_length > edge2_length
            
            pallet_data = {
                'pallet_id': i + 1,
                'orientation': float(pallet_orientation),  # Convert numpy float to Python float
                'edge1_length': float(edge1_length),
                'edge2_length': float(edge2_length),
                'long_edge_is_edge1': bool(long_edge_is_edge1),  # Convert numpy bool to Python bool
                'division_strategy': self._determine_division_strategy(layer, long_edge_is_edge1)
            }
            
            orientation_data['pallets'].append(pallet_data)
        
        # Save orientation lock
        self.locked_orientation_data = orientation_data
        self.orientation_locked = True
        
        if save_to_file:
            self.save_orientation_lock()
        
        print(f"[ModuleDivision] 🔒 LOCKED orientation for layer {layer}")
        print(f"  📋 Locked {len(pallet_corners_list)} pallets")
        for pallet in orientation_data['pallets']:
            print(f"    P{pallet['pallet_id']}: {pallet['orientation']:.1f}° → {pallet['division_strategy']}")
        
        return True
    
    def unlock_orientation(self, delete_file: bool = False):
        """
        🔓 Unlock orientation để quay về auto-detection
        
        Args:
            delete_file: Có xóa file lock không
        """
        self.orientation_locked = False
        self.locked_orientation_data = None
        
        if delete_file and os.path.exists(self.orientation_lock_file):
            os.remove(self.orientation_lock_file)
            print(f"[ModuleDivision] 🗑️ Đã xóa file {self.orientation_lock_file}")
        
        print(f"[ModuleDivision] 🔓 UNLOCKED orientation - quay về auto-detection")
    
    def save_orientation_lock(self):
        """Lưu orientation lock ra file"""
        if not self.locked_orientation_data:
            return False
        
        try:
            with open(self.orientation_lock_file, 'w', encoding='utf-8') as f:
                json.dump(self.locked_orientation_data, f, indent=2, ensure_ascii=False)
            print(f"[ModuleDivision] 💾 Đã lưu orientation lock ra {self.orientation_lock_file}")
            return True
        except Exception as e:
            print(f"[ModuleDivision] ❌ Lỗi lưu orientation lock: {e}")
            return False
    
    def load_orientation_lock(self):
        """Load orientation lock từ file"""
        try:
            if os.path.exists(self.orientation_lock_file):
                with open(self.orientation_lock_file, 'r', encoding='utf-8') as f:
                    self.locked_orientation_data = json.load(f)
                self.orientation_locked = True
                
                layer = self.locked_orientation_data.get('layer', 1)
                pallet_count = len(self.locked_orientation_data.get('pallets', []))
                locked_at = self.locked_orientation_data.get('locked_at', 'unknown')
                
                print(f"[ModuleDivision] 📁 Loaded orientation lock từ {self.orientation_lock_file}")
                print(f"  🔒 Layer {layer}, {pallet_count} pallets, locked at {locked_at}")
                return True
        except Exception as e:
            print(f"[ModuleDivision] ⚠️ Không thể load orientation lock: {e}")
        
        return False
    
    def get_orientation_lock_status(self):
        """Lấy status của orientation lock"""
        if not self.orientation_locked:
            return {
                'locked': False,
                'status': 'Auto-detection enabled'
            }
        
        return {
            'locked': True,
            'data': self.locked_orientation_data,
            'status': f"Locked for layer {self.locked_orientation_data.get('layer', '?')}"
        }
    
    def _determine_division_strategy(self, layer: int, long_edge_is_edge1: bool) -> str:
        """Xác định strategy chia region"""
        if layer == 1:
            if long_edge_is_edge1:
                return 'layer1_along_long_edge_X'
            else:
                return 'layer1_along_long_edge_Y'
        elif layer == 2:
            if not long_edge_is_edge1:
                return 'layer2_along_short_edge_X'
            else:
                return 'layer2_along_short_edge_Y'
        return 'unknown'

    def _determine_pallet_region_id(self, center_x: float) -> int:
        """
        Xác định pallet_id dựa trên vị trí X của center point.
        
        Args:
            center_x: Tọa độ X của center point
            
        Returns:
            int: 1 cho pallets1 (trái), 2 cho pallets2 (phải)
        """
        # Heuristic dựa trên regions definition:
        # - pallets1: X từ 356 đến 821
        # - pallets2: X từ 821 đến 1272
        # Threshold: X = 821 (boundary giữa pallets1 và pallets2)
        
        if center_x < 821:
            pallet_id = 1  # pallets1 → P1Rx
            region_name = "pallets1"
        else:
            pallet_id = 2  # pallets2 → P2Rx
            region_name = "pallets2"
        
        if self.debug:
            print(f"    🎯 Center X={center_x:.1f} → {region_name} (P{pallet_id})")
        
        return pallet_id
    
    def _calculate_pallet_center(self, corners: List[List[float]]) -> Tuple[float, float]:
        """
        Tính center point của pallet từ corners.
        
        Args:
            corners: [[x,y], [x,y], [x,y], [x,y]] corners của pallet
            
        Returns:
            Tuple[float, float]: (center_x, center_y)
        """
        corners_array = np.array(corners)
        center_x = np.mean(corners_array[:, 0])
        center_y = np.mean(corners_array[:, 1])
        return center_x, center_y
    
    def _calculate_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Tính center point của pallet từ bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box của pallet
            
        Returns:
            Tuple[float, float]: (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y


# Để test module
if __name__ == "__main__":
    # Test Module Division
    divider = ModuleDivision()
    
    # Pallet mẫu
    test_pallet = [100, 100, 400, 300]  # x1, y1, x2, y2
    
    print("=== TEST MODULE DIVISION ===")
    print(f"Pallet gốc: {test_pallet}")
    print(f"Kích thước: {400-100}x{300-100} = 300x200")
    
    # Test Layer 1
    print("\n--- LAYER 1 (3 phần theo chiều rộng) ---")
    regions_layer1 = divider.divide_module_1(test_pallet, layer=1)
    for region in regions_layer1:
        bbox = region['bbox']
        center = region['center']
        print(f"Vùng {region['region_id']}: bbox={[int(x) for x in bbox]}, center=({center[0]:.1f}, {center[1]:.1f})")
    
    # Test Layer 2
    print("\n--- LAYER 2 (3 phần theo chiều dài) ---")
    regions_layer2 = divider.divide_module_1(test_pallet, layer=2)
    for region in regions_layer2:
        bbox = region['bbox']
        center = region['center']
        print(f"Vùng {region['region_id']}: bbox={[int(x) for x in bbox]}, center=({center[0]:.1f}, {center[1]:.1f})")
    
    # Test với detection result mẫu
    print("\n--- TEST VỚI DETECTION RESULT ---")
    mock_detection = {
        'bounding_boxes': [test_pallet],
        'scores': [0.9],
        'classes': [0]
    }
    
    result = divider.process_pallet_detections(mock_detection, layer=1)
    print(f"Tổng số vùng: {result['total_regions']}")
    print(f"Thành công: {result['processing_info']['success']}")
    
    # Test prepare for depth
    depth_data = divider.prepare_for_depth_estimation(result)
    print(f"\nDữ liệu cho depth estimation: {len(depth_data)} vùng")
    for i, data in enumerate(depth_data):
        info = data['region_info']
        print(f"  Vùng {i+1}: Pallet {info['pallet_id']}, Region {info['region_id']}, Layer {info['layer']}")