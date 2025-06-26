"""
Module phân tích góc xoay cho robot IRB-460.
Tính toán theta 4 (end-effector rotation) dựa trên YOLO detection và module division.
"""
import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Optional, Any

class RotationAnalyzer:
    """
    Phân tích góc xoay từ YOLO detection và tính toán theta 4 cho robot IRB-460.
    """
    
    def __init__(self, debug: bool = True):
        """
        Args:
            debug: Bật chế độ debug để hiển thị thông tin chi tiết
        """
        self.debug = debug
        
    def analyze_yolo_angles(self, detections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Phân tích góc xoay từ YOLO OBB detection.
        
        Args:
            detections: Kết quả detection từ YOLO (có chứa obb_boxes)
            
        Returns:
            List[Dict]: Danh sách thông tin góc cho mỗi object
        """
        results = []
        
        # Lấy thông tin OBB từ detections
        obb_boxes = detections.get('obb_boxes', [])
        class_names = detections.get('class_names', [])
        confidences = detections.get('confidences', [])
        
        if self.debug:
            print(f"\n=== PHÂN TÍCH GÓCS XOAY YOLO ===")
            print(f"Số objects phát hiện: {len(obb_boxes)}")
        
        for i, obb in enumerate(obb_boxes):
            if len(obb) >= 5:  # [cx, cy, width, height, angle]
                cx, cy, width, height, angle_rad = obb[:5]
                
                # Chuyển từ radian sang độ
                angle_deg = math.degrees(angle_rad)
                
                # Chuẩn hóa góc về [-180, 180]
                angle_normalized = self._normalize_angle(angle_deg)
                
                # Xác định hướng theo hệ quy chiếu image
                angle_info = self._analyze_angle_direction(angle_normalized)
                
                object_info = {
                    'object_id': i,
                    'class_name': class_names[i] if i < len(class_names) else f"object_{i}",
                    'confidence': confidences[i] if i < len(confidences) else 1.0,
                    'center': (cx, cy),
                    'size': (width, height),
                    'angle_rad': angle_rad,
                    'angle_deg': angle_deg,
                    'angle_normalized': angle_normalized,
                    'angle_info': angle_info
                }
                
                results.append(object_info)
                
                if self.debug:
                    print(f"\nObject {i} ({object_info['class_name']}):")
                    print(f"  Center: ({cx:.1f}, {cy:.1f})")
                    print(f"  Size: {width:.1f} x {height:.1f}")
                    print(f"  Góc gốc: {angle_deg:.1f}°")
                    print(f"  Góc chuẩn hóa: {angle_normalized:.1f}°")
                    print(f"  Hướng: {angle_info['direction']}")
                    print(f"  Mô tả: {angle_info['description']}")
        
        return results
    
    def analyze_module_division_layout(self, division_results: List[Dict[str, Any]], 
                                     layer: int = 1) -> List[Dict[str, Any]]:
        """
        Phân tích layout từ module division để xác định hướng đặt load.
        
        Args:
            division_results: Kết quả từ module division
            layer: Layer đang xử lý (1 hoặc 2)
            
        Returns:
            List[Dict]: Thông tin hướng đặt cho mỗi region
        """
        layout_info = []
        
        if self.debug:
            print(f"\n=== PHÂN TÍCH MODULE DIVISION LAYOUT (Layer {layer}) ===")
            print(f"Số regions: {len(division_results)}")
        
        for region_data in division_results:
            region_info = region_data.get('region_info', {})
            
            # Lấy thông tin region
            pallet_id = region_info.get('pallet_id', 1)
            region_id = region_info.get('region_id', 1)
            current_layer = region_info.get('layer', layer)
            
            # Xác định hướng đặt load dựa trên layout pattern
            placement_angle = self._determine_placement_angle(pallet_id, region_id, current_layer)
            
            # Lấy corners nếu có
            corners = region_data.get('corners', [])
            
            layout_data = {
                'pallet_id': pallet_id,
                'region_id': region_id,
                'layer': current_layer,
                'placement_angle': placement_angle,
                'corners': corners,
                'center': region_data.get('center', (0, 0)),
                'bbox': region_data.get('bbox', [])
            }
            
            layout_info.append(layout_data)
            
            if self.debug:
                print(f"\nPallet {pallet_id}, Region {region_id} (Layer {current_layer}):")
                print(f"  Góc đặt load: {placement_angle:.1f}°")
                print(f"  Center: {layout_data['center']}")
                if corners:
                    print(f"  Corners: {len(corners)} điểm")
        
        return layout_info
    
    def calculate_theta4_rotation(self, load_angle: float, target_placement_angle: float) -> Dict[str, Any]:
        """
        Tính toán góc xoay theta 4 cần thiết để load trùng với hướng đặt trên pallet.
        
        Args:
            load_angle: Góc hiện tại của load trên băng tải (độ)
            target_placement_angle: Góc đặt mục tiêu trên pallet (độ)
            
        Returns:
            Dict: Thông tin chi tiết về góc xoay cần thiết
        """
        # Tính góc xoay cần thiết
        rotation_needed = target_placement_angle - load_angle
        
        # Chuẩn hóa góc xoay về [-180, 180]
        rotation_normalized = self._normalize_angle(rotation_needed)
        
        # Xác định hướng xoay
        if rotation_normalized > 0:
            rotation_direction = "Counter-clockwise (CCW)"
            rotation_sign = "+"
        elif rotation_normalized < 0:
            rotation_direction = "Clockwise (CW)"
            rotation_sign = ""
        else:
            rotation_direction = "No rotation needed"
            rotation_sign = ""
        
        # Tính góc theta 4 cho robot IRB-460
        # Lưu ý: Có thể cần điều chỉnh dựa trên cấu hình thực tế của robot
        theta4_value = rotation_normalized
        
        result = {
            'load_angle': load_angle,
            'target_angle': target_placement_angle,
            'rotation_needed': rotation_needed,
            'rotation_normalized': rotation_normalized,
            'rotation_direction': rotation_direction,
            'rotation_sign': rotation_sign,
            'theta4_value': theta4_value,
            'theta4_command': f"THETA4 = {rotation_sign}{abs(theta4_value):.1f}°"
        }
        
        if self.debug:
            print(f"\n=== TÍNH TOÁN THETA 4 ===")
            print(f"Góc load hiện tại: {load_angle:.1f}°")
            print(f"Góc đặt mục tiêu: {target_placement_angle:.1f}°")
            print(f"Góc xoay cần thiết: {rotation_sign}{abs(rotation_normalized):.1f}°")
            print(f"Hướng xoay: {rotation_direction}")
            print(f"Lệnh THETA4: {result['theta4_command']}")
        
        return result
    
    def process_complete_analysis(self, detections: Dict[str, Any], 
                                division_results: List[Dict[str, Any]], 
                                layer: int = 1) -> Dict[str, Any]:
        """
        Thực hiện phân tích hoàn chỉnh từ detection đến tính toán theta 4.
        
        Args:
            detections: Kết quả YOLO detection
            division_results: Kết quả module division  
            layer: Layer đang xử lý
            
        Returns:
            Dict: Kết quả phân tích hoàn chỉnh
        """
        # Phân tích góc YOLO
        yolo_angles = self.analyze_yolo_angles(detections)
        
        # Phân tích layout module division
        layout_info = self.analyze_module_division_layout(division_results, layer)
        
        # Tìm load objects (không phải pallet)
        load_objects = [obj for obj in yolo_angles if 'load' in obj['class_name'].lower()]
        pallet_objects = [obj for obj in yolo_angles if 'pallet' in obj['class_name'].lower() or 'load' not in obj['class_name'].lower()]
        
        # Tính toán theta 4 cho mỗi load
        theta4_calculations = []
        
        if self.debug:
            print(f"\n=== PHÂN TÍCH HOÀN CHỈNH ===")
            print(f"Load objects: {len(load_objects)}")
            print(f"Pallet objects: {len(pallet_objects)}")
            print(f"Layout regions: {len(layout_info)}")
        
        for i, load_obj in enumerate(load_objects):
            # Lấy góc của load
            load_angle = load_obj['angle_normalized']
            
            # Giả sử đặt vào region đầu tiên (có thể cải thiện bằng cách match theo vị trí)
            if layout_info:
                target_region = layout_info[i % len(layout_info)]
                target_angle = target_region['placement_angle']
                
                # Tính toán theta 4
                theta4_calc = self.calculate_theta4_rotation(load_angle, target_angle)
                theta4_calc['load_object'] = load_obj
                theta4_calc['target_region'] = target_region
                
                theta4_calculations.append(theta4_calc)
        
        result = {
            'yolo_angles': yolo_angles,
            'layout_info': layout_info,
            'theta4_calculations': theta4_calculations,
            'load_objects': load_objects,
            'pallet_objects': pallet_objects,
            'summary': {
                'total_objects': len(yolo_angles),
                'load_count': len(load_objects),
                'pallet_count': len(pallet_objects),
                'regions_count': len(layout_info),
                'theta4_count': len(theta4_calculations)
            }
        }
        
        return result
    
    def _normalize_angle(self, angle_deg: float) -> float:
        """Chuẩn hóa góc về khoảng [-180, 180]"""
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg <= -180:
            angle_deg += 360
        return angle_deg
    
    def _analyze_angle_direction(self, angle_deg: float) -> Dict[str, str]:
        """Phân tích hướng của góc xoay"""
        angle_abs = abs(angle_deg)
        
        if angle_abs < 5:
            direction = "Horizontal (→)"
            description = "Gần như nằm ngang"
        elif 85 < angle_abs < 95:
            direction = "Vertical (↑)" if angle_deg > 0 else "Vertical (↓)"
            description = "Gần như thẳng đứng"
        elif 0 < angle_deg < 90:
            direction = "NE (↗)"
            description = "Hướng đông bắc"
        elif 90 < angle_deg < 180:
            direction = "NW (↖)"
            description = "Hướng tây bắc"
        elif -90 < angle_deg < 0:
            direction = "SE (↘)"
            description = "Hướng đông nam"
        elif -180 < angle_deg < -90:
            direction = "SW (↙)"
            description = "Hướng tây nam"
        else:
            direction = "Unknown"
            description = "Không xác định"
        
        return {
            'direction': direction,
            'description': description,
            'quadrant': self._get_quadrant(angle_deg)
        }
    
    def _get_quadrant(self, angle_deg: float) -> str:
        """Xác định góc phần tư"""
        if 0 <= angle_deg < 90:
            return "Quadrant I"
        elif 90 <= angle_deg < 180:
            return "Quadrant II"
        elif -180 <= angle_deg < -90:
            return "Quadrant III"
        elif -90 <= angle_deg < 0:
            return "Quadrant IV"
        else:
            return "Boundary"
    
    def _determine_placement_angle(self, pallet_id: int, region_id: int, layer: int) -> float:
        """
        Xác định góc đặt load dựa trên pallet layout pattern.
        
        Lưu ý: Đây là ví dụ đơn giản, cần điều chỉnh theo layout thực tế.
        """
        # Pattern mẫu cho layer 1: đặt theo hàng ngang
        if layer == 1:
            if region_id in [1, 2]:  # Hàng trên
                return 0.0  # Nằm ngang
            elif region_id in [3, 4]:  # Hàng dưới
                return 0.0  # Nằm ngang
        
        # Pattern mẫu cho layer 2: đặt xen kẽ
        elif layer == 2:
            if region_id % 2 == 1:  # Region lẻ
                return 0.0  # Nằm ngang
            else:  # Region chẵn
                return 90.0  # Thẳng đứng
        
        # Mặc định
        return 0.0
    
    def create_visualization(self, image: np.ndarray, analysis_result: Dict[str, Any]) -> np.ndarray:
        """
        Tạo ảnh visualization với thông tin góc xoay và theta 4.
        
        Args:
            image: Ảnh gốc
            analysis_result: Kết quả phân tích từ process_complete_analysis
            
        Returns:
            np.ndarray: Ảnh đã được vẽ thông tin
        """
        result_image = image.copy()
        
        # Màu sắc
        load_color = (0, 255, 0)    # Xanh lá cho load
        pallet_color = (255, 0, 0)  # Đỏ cho pallet
        region_color = (0, 255, 255) # Vàng cho region
        text_bg_color = (0, 0, 0)   # Đen cho background text
        
        # Vẽ load objects
        for load_obj in analysis_result['load_objects']:
            cx, cy = load_obj['center']
            angle = load_obj['angle_normalized']
            
            # Vẽ điểm center
            cv2.circle(result_image, (int(cx), int(cy)), 5, load_color, -1)
            
            # Vẽ vector hướng
            length = 50
            end_x = cx + length * math.cos(math.radians(angle))
            end_y = cy + length * math.sin(math.radians(angle))
            cv2.arrowedLine(result_image, (int(cx), int(cy)), (int(end_x), int(end_y)), load_color, 3)
            
            # Vẽ text thông tin
            text = f"Load: {angle:.1f}°"
            self._draw_text_with_background(result_image, text, (int(cx) + 10, int(cy) - 10), load_color)
        
        # Vẽ pallet objects
        for pallet_obj in analysis_result['pallet_objects']:
            cx, cy = pallet_obj['center']
            angle = pallet_obj['angle_normalized']
            
            # Vẽ điểm center
            cv2.circle(result_image, (int(cx), int(cy)), 5, pallet_color, -1)
            
            # Vẽ vector hướng
            length = 40
            end_x = cx + length * math.cos(math.radians(angle))
            end_y = cy + length * math.sin(math.radians(angle))
            cv2.arrowedLine(result_image, (int(cx), int(cy)), (int(end_x), int(end_y)), pallet_color, 2)
            
            # Vẽ text thông tin
            text = f"Pallet: {angle:.1f}°"
            self._draw_text_with_background(result_image, text, (int(cx) + 10, int(cy) + 20), pallet_color)
        
        # Vẽ regions từ module division
        for layout in analysis_result['layout_info']:
            corners = layout.get('corners', [])
            if corners:
                # Vẽ region boundary
                pts = np.array(corners, np.int32).reshape((-1, 1, 2))
                cv2.polylines(result_image, [pts], True, region_color, 2)
                
                # Vẽ center và góc đặt
                cx, cy = layout['center']
                placement_angle = layout['placement_angle']
                
                cv2.circle(result_image, (int(cx), int(cy)), 3, region_color, -1)
                
                # Vẽ vector hướng đặt
                length = 30
                end_x = cx + length * math.cos(math.radians(placement_angle))
                end_y = cy + length * math.sin(math.radians(placement_angle))
                cv2.arrowedLine(result_image, (int(cx), int(cy)), (int(end_x), int(end_y)), region_color, 2)
                
                # Text thông tin region
                region_text = f"R{layout['region_id']}: {placement_angle:.1f}°"
                self._draw_text_with_background(result_image, region_text, (int(cx) - 30, int(cy) - 20), region_color)
        
        # Vẽ thông tin theta 4 ở góc trên
        y_offset = 30
        for i, theta4_calc in enumerate(analysis_result['theta4_calculations']):
            theta4_text = f"THETA4 #{i+1}: {theta4_calc['theta4_command']}"
            self._draw_text_with_background(result_image, theta4_text, (10, y_offset + i * 30), (255, 255, 255))
            
            rotation_text = f"  Xoay {theta4_calc['rotation_direction']}"
            self._draw_text_with_background(result_image, rotation_text, (10, y_offset + i * 30 + 20), (200, 200, 200))
        
        return result_image
    
    def _draw_text_with_background(self, image: np.ndarray, text: str, position: Tuple[int, int], 
                                 color: Tuple[int, int, int], bg_color: Tuple[int, int, int] = (0, 0, 0)):
        """Vẽ text với background để dễ đọc"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Lấy kích thước text
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Vẽ background
        cv2.rectangle(image, 
                     (position[0] - 2, position[1] - text_size[1] - 2),
                     (position[0] + text_size[0] + 2, position[1] + 2),
                     bg_color, -1)
        
        # Vẽ text
        cv2.putText(image, text, position, font, font_scale, color, thickness) 