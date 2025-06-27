"""
Calculator cho góc xoay theta 4 robot IRB-460.
Tích hợp với YOLO detection và module division để tính toán góc xoay cần thiết.

HỆ TỌA ĐỘ CUSTOM:
- Trục X+: Hướng từ ĐÔNG sang TÂY (phải → trái) ←
- Trục Y+: Hướng xuống dưới (bắc → nam) ↓  
- 0°: Từ đông sang tây (E→W) ←
- 90°: Xuống dưới (N→S) ↓
- 180°: Từ tây sang đông (W→E) →
- -90°: Lên trên (S→N) ↑

Vector visualization đã được điều chỉnh để tuân theo hệ tọa độ này.
"""
import cv2
import os
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Any

# Import từ hệ thống hiện có
try:
    from detection import (YOLOTensorRT, ModuleDivision)
except ImportError:
    print("Lỗi import! Hãy đảm bảo đang chạy từ thư mục gốc của project")
    exit(1)

# Đường dẫn model (sử dụng chung với use_tensorrt_example.py)
ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "utils", "detection", "best.engine")

class Theta4Calculator:
    """
    Calculator để tính toán góc xoay theta 4 cho robot IRB-460.
    """
    
    def __init__(self, debug: bool = True):
        """
        Args:
            debug: Bật chế độ debug để hiển thị thông tin chi tiết
        """
        self.debug = debug
        
    def normalize_angle(self, angle_deg: float) -> float:
        """Chuẩn hóa góc về khoảng [-180, 180]"""
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg <= -180:
            angle_deg += 360
        return angle_deg
    
    def analyze_object_dimensions_and_orientation(self, detections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Phân tích kích thước và hướng của objects từ YOLO OBB detection.
        Xác định trục chính (chiều dài) và hướng X theo trục chính.
        
        Args:
            detections: Kết quả detection từ YOLO (có chứa obb_boxes)
            
        Returns:
            List[Dict]: Danh sách thông tin chi tiết cho mỗi object
        """
        results = []
        
        # Lấy thông tin OBB từ detections
        obb_boxes = detections.get('obb_boxes', [])
        class_names = detections.get('class_names', [])
        confidences = detections.get('confidences', [])
        
        if self.debug:
            print(f"\n=== PHÂN TÍCH KÍCH THƯỚC VÀ HƯỚNG OBJECTS ===")
            print(f"Số objects phát hiện: {len(obb_boxes)}")
        
        for i, obb in enumerate(obb_boxes):
            if len(obb) >= 5:  # [cx, cy, width, height, angle]
                cx, cy, width, height, angle_rad = obb[:5]
                
                # Chuyển từ radian sang độ
                angle_deg = math.degrees(angle_rad)
                
                # Chuẩn hóa góc về [-180, 180]
                angle_normalized = self.normalize_angle(angle_deg)
                
                # Xác định trục chính (chiều dài) và trục phụ (chiều rộng)
                if width >= height:
                    length = width
                    width_minor = height
                    major_axis_is_width = True
                    # Nếu width là trục chính, góc đã đúng
                    major_axis_angle = angle_normalized
                else:
                    length = height
                    width_minor = width
                    major_axis_is_width = False
                    # Nếu height là trục chính, cần điều chỉnh góc
                    major_axis_angle = self.normalize_angle(angle_normalized + 90)
                
                # Tính tỷ lệ dài/rộng
                aspect_ratio = length / width_minor
                
                # Xác định loại object dựa trên tỷ lệ
                if aspect_ratio > 2.0:
                    shape_type = "Dài" 
                elif aspect_ratio < 1.5:
                    shape_type = "Vuông"
                else:
                    shape_type = "Chữ nhật"
                
                object_info = {
                    'object_id': i,
                    'class_name': class_names[i] if i < len(class_names) else f"object_{i}",
                    'confidence': confidences[i] if i < len(confidences) else 1.0,
                    'center': (cx, cy),
                    'original_size': (width, height),
                    'length': length,
                    'width': width_minor,
                    'aspect_ratio': aspect_ratio,
                    'shape_type': shape_type,
                    'major_axis_is_width': major_axis_is_width,
                    'original_angle_rad': angle_rad,
                    'original_angle_deg': angle_deg,
                    'original_angle_normalized': angle_normalized,
                    'major_axis_angle': major_axis_angle,  # Góc của trục chính (chiều dài)
                    'x_axis_angle': major_axis_angle  # Trục X object theo chiều dài
                }
                
                results.append(object_info)
                
                if self.debug:
                    print(f"\nObject {i} ({object_info['class_name']}):")
                    print(f"  Center: ({cx:.1f}, {cy:.1f})")
                    print(f"  Kích thước gốc: {width:.1f} x {height:.1f}")
                    print(f"  Chiều dài: {length:.1f} px")
                    print(f"  Chiều rộng: {width_minor:.1f} px")
                    print(f"  Tỷ lệ L/W: {aspect_ratio:.2f} ({shape_type})")
                    print(f"  Trục chính là: {'Width' if major_axis_is_width else 'Height'}")
                    print(f"  Góc gốc YOLO: {angle_normalized:.1f}°")
                    print(f"  Góc trục X (theo chiều dài): {major_axis_angle:.1f}°")
                    
                    # Phân tích hướng trục X object theo hệ tọa độ custom
                    self.analyze_direction(major_axis_angle, "  Hướng trục X object")
        
        return results
    
    def analyze_direction(self, angle: float, prefix: str = "Hướng"):
        """Phân tích và in hướng của góc theo hệ tọa độ custom"""
        if abs(angle) < 5:
            direction = "Đông→Tây (E→W) ←"
        elif abs(angle - 180) < 5 or abs(angle + 180) < 5:
            direction = "Tây→Đông (W→E) →"
        elif 85 < angle < 95:
            direction = "Bắc→Nam (N→S) ↓"
        elif -95 < angle < -85:
            direction = "Nam→Bắc (S→N) ↑"
        elif 0 < angle < 90:
            direction = "Đông Nam (SE) ↘"
        elif 90 < angle < 180:
            direction = "Tây Nam (SW) ↙"
        elif -90 < angle < 0:
            direction = "Đông Bắc (NE) ↗"
        elif -180 < angle < -90:
            direction = "Tây Bắc (NW) ↖"
        else:
            direction = "Unknown"
        
        if self.debug and prefix:  # Chỉ in khi có prefix
            print(f"{prefix}: {direction}")
        
        return direction
    
    def define_robot_x_axis_direction(self) -> float:
        """
        Định nghĩa hướng trục X mong muốn của robot khi đặt load.
        
        Returns:
            float: Góc trục X robot (degrees) theo hệ tọa độ custom
        """
        # Có thể cấu hình dựa trên setup robot thực tế
        # Ví dụ: Robot muốn đặt load theo hướng Đông→Tây
        robot_x_direction = 0.0  # Đông→Tây (E→W)
        
        if self.debug:
            print(f"\n=== HƯỚNG TRỤC X ROBOT ===")
            print(f"Trục X robot mong muốn: {robot_x_direction:.1f}°")
            self.analyze_direction(robot_x_direction, "Robot X direction")
        
        return robot_x_direction
    
    def determine_placement_angle(self, pallet_id: int, region_id: int, layer: int) -> float:
        """
        Xác định góc đặt load dựa trên pallet layout pattern.
        
        Args:
            pallet_id: ID của pallet
            region_id: ID của region trong pallet
            layer: Layer đang xử lý (1 hoặc 2)
            
        Returns:
            float: Góc đặt load (degrees)
        """
        # Sử dụng hướng X robot làm chuẩn
        robot_x_direction = self.define_robot_x_axis_direction()
        
        # Pattern mẫu cho layer 1: đặt theo hướng robot X
        if layer == 1:
            return robot_x_direction  # Tất cả theo hướng robot X
        
        # Pattern mẫu cho layer 2: đặt xen kẽ
        elif layer == 2:
            if region_id % 2 == 1:  # Region lẻ
                return robot_x_direction  # Theo hướng robot X
            else:  # Region chẵn
                return self.normalize_angle(robot_x_direction + 90.0)  # Vuông góc với robot X
        
        # Mặc định
        return robot_x_direction
    
    def calculate_theta4_rotation(self, load_object: Dict[str, Any], target_placement_angle: float) -> Dict[str, Any]:
        """
        Tính toán góc xoay theta 4 cần thiết dựa trên trục X của load và hướng đặt mục tiêu.
        
        Args:
            load_object: Thông tin chi tiết về load object (từ analyze_object_dimensions_and_orientation)
            target_placement_angle: Góc đặt mục tiêu trên pallet (độ)
            
        Returns:
            Dict: Thông tin chi tiết về góc xoay cần thiết
        """
        # Lấy góc trục X của load (theo chiều dài)
        load_x_angle = load_object['x_axis_angle']
        
        # Tính góc xoay cần thiết để trục X load trùng với hướng đặt mục tiêu
        rotation_needed = target_placement_angle - load_x_angle
        
        # Chuẩn hóa góc xoay về [-180, 180]
        rotation_normalized = self.normalize_angle(rotation_needed)
        
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
        theta4_value = rotation_normalized
        
        result = {
            'load_object': load_object,
            'load_x_angle': load_x_angle,
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
            print(f"Load: {load_object['class_name']} (L={load_object['length']:.1f}, W={load_object['width']:.1f})")
            print(f"Trục X load hiện tại: {load_x_angle:.1f}°")
            self.analyze_direction(load_x_angle, "  Hướng trục X load")
            print(f"Góc đặt mục tiêu: {target_placement_angle:.1f}°")
            self.analyze_direction(target_placement_angle, "  Hướng đặt mục tiêu")
            print(f"Góc xoay cần thiết: {rotation_sign}{abs(rotation_normalized):.1f}°")
            print(f"Hướng xoay: {rotation_direction}")
            print(f"➤ LỆNH ROBOT: {result['theta4_command']}")
        
        return result
    
    def explain_coordinate_system(self):
        """
        Giải thích hệ tọa độ được sử dụng trong visualization.
        """
        if self.debug:
            print("\n" + "="*60)
            print("HỆ TỌA ĐỘ VÀ GÓC XOAY (CUSTOM SYSTEM)")
            print("="*60)
            print("1. HỆ TỌA ĐỘ CUSTOM:")
            print("   - Gốc (0,0): Góc trên-trái của ảnh")
            print("   - Trục X+: Hướng từ ĐÔNG sang TÂY (phải → trái) ←")
            print("   - Trục Y+: Hướng xuống dưới ↓")
            print("   - Trục X-: Hướng từ TÂY sang ĐÔNG (trái → phải) →")
            print("   - Trục Y-: Hướng lên trên ↑")
            print()
            print("2. GÓC XOAY (theo hệ tọa độ custom):")
            print("   - 0°: Hướng từ đông sang tây (phải → trái) ←")
            print("   - 90°: Hướng xuống dưới (bắc → nam) ↓")  
            print("   - 180° hoặc -180°: Hướng từ tây sang đông (trái → phải) →")
            print("   - -90°: Hướng lên trên (nam → bắc) ↑")
            print()
            print("3. HƯỚNG XOAY:")
            print("   - Góc dương (+): Counter-clockwise (ngược chiều kim đồng hồ)")
            print("   - Góc âm (-): Clockwise (cùng chiều kim đồng hồ)")
            print()
            print("4. THETA 4 ROBOT:")
            print("   - Theta 4 = Góc đặt mục tiêu - Góc hiện tại của load")
            print("   - Kết quả sẽ cho robot biết cần xoay bao nhiêu độ")
            print("="*60)

    def create_visualization(self, image: np.ndarray, object_analysis: List[Dict], 
                           theta4_calculations: List[Dict]) -> np.ndarray:
        """
        Tạo ảnh visualization với thông tin kích thước, hướng trục X và theta 4.
        
        Args:
            image: Ảnh gốc
            object_analysis: Thông tin phân tích objects (từ analyze_object_dimensions_and_orientation)
            theta4_calculations: Kết quả tính toán theta 4
            
        Returns:
            np.ndarray: Ảnh đã được vẽ thông tin
        """
        result_image = image.copy()
        
        # Màu sắc
        load_color = (0, 255, 0)    # Xanh lá cho load
        pallet_color = (255, 0, 0)  # Đỏ cho pallet
        
        # Vẽ objects với arrows chỉ hướng trục X (theo chiều dài)
        for obj in object_analysis:
            cx, cy = obj['center']
            x_axis_angle = obj['x_axis_angle']  # Góc trục X theo chiều dài
            class_name = obj['class_name']
            length = obj['length']
            width = obj['width']
            
            # Chọn màu dựa trên class
            if 'load' in class_name.lower():
                color = load_color
            else:
                color = pallet_color
            
            # Vẽ điểm center
            cv2.circle(result_image, (int(cx), int(cy)), 5, color, -1)
            
            # Vẽ vector trục X (theo chiều dài)
            # HỆ TỌA ĐỘ CUSTOM: X+ hướng từ đông sang tây (phải sang trái), Y+ hướng xuống
            arrow_length = 60
            end_x = cx - arrow_length * math.cos(math.radians(x_axis_angle))  # Đảo ngược trục X (East→West)
            end_y = cy - arrow_length * math.sin(math.radians(x_axis_angle))  # Đảo ngược trục Y (OpenCV)
            cv2.arrowedLine(result_image, (int(cx), int(cy)), 
                           (int(end_x), int(end_y)), color, 3)
            
            # Vẽ bounding box để hiển thị kích thước
            self.draw_oriented_bounding_box(result_image, obj, color)
            
            # Vẽ text thông tin
            text1 = f"{class_name}: L={length:.0f}, W={width:.0f}"
            text2 = f"X-axis: {x_axis_angle:.1f}°"
            self.draw_text_with_background(result_image, text1, 
                                         (int(cx) + 10, int(cy) - 20), color)
            self.draw_text_with_background(result_image, text2, 
                                         (int(cx) + 10, int(cy) - 5), color)
        
        # Vẽ thông tin theta 4 ở góc trên
        y_offset = 30
        for i, calc in enumerate(theta4_calculations):
            load_obj = calc['load_object']
            theta4_text = f"THETA4 #{i+1}: {calc['theta4_command']}"
            detail_text = f"  {load_obj['class_name']} (L={load_obj['length']:.0f}) {calc['load_x_angle']:.1f}°→{calc['target_angle']:.1f}°"
            
            self.draw_text_with_background(result_image, theta4_text, 
                                         (10, y_offset + i * 40), (255, 255, 255))
            self.draw_text_with_background(result_image, detail_text, 
                                         (10, y_offset + i * 40 + 15), (200, 200, 200))
        
        # Vẽ compass hệ tọa độ ở góc phải dưới
        self.draw_coordinate_compass(result_image)
        
        return result_image
    
    def draw_oriented_bounding_box(self, image: np.ndarray, obj: Dict[str, Any], color: Tuple[int, int, int]):
        """Vẽ oriented bounding box để hiển thị kích thước thực tế"""
        cx, cy = obj['center']
        length = obj['length']
        width = obj['width']
        angle = obj['x_axis_angle']
        
        # Tính các góc của bounding box
        cos_angle = math.cos(math.radians(angle))
        sin_angle = math.sin(math.radians(angle))
        
        # Các vector từ center đến các góc (trong hệ tọa độ local)
        half_length = length / 2
        half_width = width / 2
        
        # 4 góc trong hệ tọa độ local
        corners_local = [
            (-half_length, -half_width),  # Top-left
            (half_length, -half_width),   # Top-right
            (half_length, half_width),    # Bottom-right
            (-half_length, half_width)    # Bottom-left
        ]
        
        # Chuyển đổi sang hệ tọa độ image với custom coordinate system
        corners_image = []
        for lx, ly in corners_local:
            # Áp dụng rotation và custom coordinate system
            ix = cx - (lx * cos_angle - ly * sin_angle)  # Đảo ngược X
            iy = cy - (lx * sin_angle + ly * cos_angle)  # Đảo ngược Y
            corners_image.append((int(ix), int(iy)))
        
        # Vẽ bounding box
        for i in range(4):
            pt1 = corners_image[i]
            pt2 = corners_image[(i + 1) % 4]
            cv2.line(image, pt1, pt2, color, 2)
        
        # Vẽ đường chỉ chiều dài (trục X)
        length_start = corners_image[0]
        length_end = corners_image[1]
        cv2.line(image, length_start, length_end, color, 4)
    
    def draw_text_with_background(self, image: np.ndarray, text: str, 
                                position: Tuple[int, int], color: Tuple[int, int, int], 
                                bg_color: Tuple[int, int, int] = (0, 0, 0)):
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
    
    def draw_coordinate_compass(self, image: np.ndarray):
        """Vẽ compass hệ tọa độ ở góc ảnh"""
        h, w = image.shape[:2]
        
        # Vị trí compass (góc phải dưới)
        center_x = w - 80
        center_y = h - 80
        
        # Vẽ background cho compass
        cv2.rectangle(image, (center_x - 50, center_y - 50), 
                     (center_x + 50, center_y + 50), (0, 0, 0), -1)
        cv2.rectangle(image, (center_x - 50, center_y - 50), 
                     (center_x + 50, center_y + 50), (255, 255, 255), 2)
        
        # Vẽ các trục theo hệ tọa độ custom
        axis_length = 35
        
        # Trục X+ (màu đỏ, hướng từ đông sang tây: phải → trái)
        cv2.arrowedLine(image, (center_x, center_y), 
                       (center_x - axis_length, center_y), (0, 0, 255), 2)
        cv2.putText(image, "X+", (center_x - axis_length - 15, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(image, "E→W", (center_x - axis_length - 15, center_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Trục Y+ (màu xanh lá, hướng xuống: bắc → nam)
        cv2.arrowedLine(image, (center_x, center_y), 
                       (center_x, center_y + axis_length), (0, 255, 0), 2)
        cv2.putText(image, "Y+", (center_x + 5, center_y + axis_length + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(image, "N→S", (center_x + 5, center_y + axis_length + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Vẽ các góc chuẩn theo hệ tọa độ custom
        angle_radius = 25
        
        # 0° (từ đông sang tây - trái)
        cv2.circle(image, (center_x - angle_radius, center_y), 3, (255, 255, 255), -1)
        cv2.putText(image, "0°", (center_x - angle_radius - 25, center_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(image, "E→W", (center_x - angle_radius - 25, center_y + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)
        
        # 90° (xuống: bắc → nam)
        cv2.circle(image, (center_x, center_y + angle_radius), 3, (255, 255, 255), -1)
        cv2.putText(image, "90°", (center_x - 15, center_y + angle_radius + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(image, "N→S", (center_x - 15, center_y + angle_radius + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)
        
        # 180° (từ tây sang đông - phải)
        cv2.circle(image, (center_x + angle_radius, center_y), 3, (255, 255, 255), -1)
        cv2.putText(image, "180°", (center_x + angle_radius + 5, center_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(image, "W→E", (center_x + angle_radius + 5, center_y + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)
        
        # -90° (lên: nam → bắc)
        cv2.circle(image, (center_x, center_y - angle_radius), 3, (255, 255, 255), -1)
        cv2.putText(image, "-90°", (center_x - 15, center_y - angle_radius - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(image, "S→N", (center_x - 15, center_y - angle_radius - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)

def demo_single_image():
    """
    Demo phân tích góc xoay trên một ảnh đơn lẻ.
    """
    print("=== DEMO PHÂN TÍCH GÓC XOAY VÀ TÍNH TOÁN THETA 4 ===")
    print("Chọn ảnh để phân tích:")
    
    # Hiển thị ảnh có sẵn
    pallets_folder = "load_on_robot_images"
    if os.path.exists(pallets_folder):
        image_files = [f for f in os.listdir(pallets_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()
        
        for i, img_file in enumerate(image_files, 1):
            print(f"  {i}. {img_file}")
        print(f"  0. Nhập đường dẫn khác")
        
        choice = input(f"\nChọn ảnh (1-{len(image_files)}) hoặc 0: ")
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(image_files):
                image_path = os.path.join(pallets_folder, image_files[choice_num - 1])
            elif choice_num == 0:
                image_path = input("Nhập đường dẫn ảnh: ")
            else:
                print("Lựa chọn không hợp lệ, sử dụng ảnh đầu tiên")
                image_path = os.path.join(pallets_folder, image_files[0])
        except ValueError:
            print("Lựa chọn không hợp lệ, sử dụng ảnh đầu tiên")
            image_path = os.path.join(pallets_folder, image_files[0])
    else:
        image_path = input("Nhập đường dẫn ảnh: ")
    
    # Đọc ảnh
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    print(f"\nĐang xử lý ảnh: {image_path}")
    height, width = frame.shape[:2]
    print(f"Kích thước ảnh: {width} x {height}")
    
    # Khởi tạo các model
    print("\nKhởi tạo YOLO model...")
    yolo_model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.5)
    
    print("Khởi tạo Module Division...")
    divider = ModuleDivision()
    
    print("Khởi tạo Theta4 Calculator...")
    calculator = Theta4Calculator(debug=True)
    
    # Chọn layer để test
    layer_choice = input("\nChọn layer (1 hoặc 2, mặc định 1): ")
    try:
        layer = int(layer_choice) if layer_choice else 1
    except ValueError:
        layer = 1
    
    print(f"Sử dụng layer: {layer}")
    
    # Thực hiện detection
    print("\n" + "="*50)
    print("BƯỚC 1: YOLO DETECTION")
    print("="*50)
    
    start_time = time.time()
    detections = yolo_model.detect(frame)
    yolo_time = time.time() - start_time
    
    print(f"Thời gian YOLO: {yolo_time*1000:.2f} ms")
    print(f"Số objects phát hiện: {len(detections.get('bounding_boxes', []))}")
    
    if len(detections.get('bounding_boxes', [])) == 0:
        print("Không phát hiện object nào!")
        return
    
    # Thực hiện module division
    print("\n" + "="*50)
    print("BƯỚC 2: MODULE DIVISION")
    print("="*50)
    
    start_time = time.time()
    divided_result = divider.process_pallet_detections(detections, layer=layer)
    division_regions = divider.prepare_for_depth_estimation(divided_result)
    division_time = time.time() - start_time
    
    print(f"Thời gian Module Division: {division_time*1000:.2f} ms")
    print(f"Số regions được tạo: {len(division_regions)}")
    
    # Thực hiện phân tích góc xoay
    print("\n" + "="*50)
    print("BƯỚC 3: PHÂN TÍCH GÓC XOAY VÀ TÍNH TOÁN THETA 4")
    print("="*50)
    
    # Giải thích hệ tọa độ
    calculator.explain_coordinate_system()
    
    start_time = time.time()
    
    # Phân tích kích thước và hướng objects
    object_analysis = calculator.analyze_object_dimensions_and_orientation(detections)
    
    # Tìm load objects (không phải pallet)
    load_objects = [obj for obj in object_analysis if 'load' in obj['class_name'].lower()]
    pallet_objects = [obj for obj in object_analysis if 'load' not in obj['class_name'].lower()]
    
    print(f"\nPhân loại objects:")
    print(f"  Load objects: {len(load_objects)}")
    print(f"  Pallet objects: {len(pallet_objects)}")
    print(f"  Regions: {len(division_regions)}")
    
    # Tính toán theta 4 cho mỗi load
    theta4_calculations = []
    
    for i, load_obj in enumerate(load_objects):
        # Giả sử đặt vào region tương ứng
        if i < len(division_regions):
            region_data = division_regions[i]
            region_info = region_data.get('region_info', {})
            
            pallet_id = region_info.get('pallet_id', 1)
            region_id = region_info.get('region_id', i+1)
            
            # Xác định góc đặt dựa trên robot và layout
            target_angle = calculator.determine_placement_angle(pallet_id, region_id, layer)
            
            # Tính toán theta 4 dựa trên trục X của load
            theta4_calc = calculator.calculate_theta4_rotation(load_obj, target_angle)
            theta4_calc['target_region'] = region_info
            
            theta4_calculations.append(theta4_calc)
            
            print(f"\nLoad #{i+1} ({load_obj['class_name']}):")
            print(f"  Kích thước: L={load_obj['length']:.1f}, W={load_obj['width']:.1f}")
            print(f"  Đặt vào: Pallet {pallet_id}, Region {region_id}")
        else:
            print(f"\nLoad #{i+1}: Không có region để đặt")
    
    analysis_time = time.time() - start_time
    print(f"\nThời gian phân tích: {analysis_time*1000:.2f} ms")
    
    # Hiển thị kết quả chi tiết
    print("\n" + "="*50)
    print("KẾT QUẢ LỆNH THETA 4 CHO ROBOT IRB-460")
    print("="*50)
    
    if theta4_calculations:
        for i, calc in enumerate(theta4_calculations):
            load_obj = calc['load_object']
            print(f"\nLoad #{i+1} - {load_obj['class_name']}:")
            print(f"  Kích thước: L={load_obj['length']:.1f} x W={load_obj['width']:.1f} (tỷ lệ: {load_obj['aspect_ratio']:.2f})")
            print(f"  Trục X load hiện tại: {calc['load_x_angle']:.1f}°")
            calculator.analyze_direction(calc['load_x_angle'], "    Hướng trục X")
            print(f"  Góc đặt mục tiêu: {calc['target_angle']:.1f}°")
            calculator.analyze_direction(calc['target_angle'], "    Hướng đặt mục tiêu")
            print(f"  Góc xoay cần thiết: {calc['rotation_sign']}{abs(calc['rotation_normalized']):.1f}°")
            print(f"  Hướng xoay: {calc['rotation_direction']}")
            print(f"  ➤ LỆNH ROBOT: {calc['theta4_command']}")
    else:
        print("\nKhông tìm thấy load objects để tính toán theta 4!")
    
    # Tạo visualization
    print(f"\n--- TẠO VISUALIZATION ---")
    vis_image = calculator.create_visualization(frame, object_analysis, theta4_calculations)
    
    # Hiển thị ảnh
    print(f"\nHiển thị kết quả visualization...")
    
    # Ảnh detection gốc
    cv2.imshow("YOLO Detection", detections["annotated_frame"])
    
    # Ảnh phân tích góc xoay
    cv2.imshow("Theta4 Analysis", vis_image)
    
    print(f"\nẢnh đã được hiển thị. Nhấn phím bất kỳ để tiếp tục...")
    cv2.waitKey(0)
    
    # Lưu kết quả
    save_choice = input("\nBạn có muốn lưu kết quả? (y/n): ").lower()
    if save_choice in ['y', 'yes']:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Lưu ảnh visualization
        vis_output_path = f"theta4_analysis_{base_name}_layer{layer}.jpg"
        cv2.imwrite(vis_output_path, vis_image)
        print(f"Đã lưu visualization: {vis_output_path}")
        
        # Lưu báo cáo text
        report_path = f"theta4_report_{base_name}_layer{layer}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("BÁO CÁO TÍNH TOÁN THETA 4 CHO ROBOT IRB-460\n")
            f.write("="*60 + "\n")
            f.write(f"Ảnh: {image_path}\n")
            f.write(f"Layer: {layer}\n")
            f.write(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            f.write("OBJECTS PHÁT HIỆN:\n")
            f.write("-" * 20 + "\n")
            for obj in object_analysis:
                f.write(f"{obj['class_name']}: L={obj['length']:.1f} x W={obj['width']:.1f}\n")
                direction = calculator.analyze_direction(obj['x_axis_angle'], '')
                f.write(f"  Trục X: {obj['x_axis_angle']:.1f}° ({direction})\n")
            f.write("\n")
            
            f.write("LỆNH THETA 4:\n")
            f.write("-" * 20 + "\n")
            for i, calc in enumerate(theta4_calculations):
                load_obj = calc['load_object']
                f.write(f"Load #{i+1} - {load_obj['class_name']}:\n")
                f.write(f"  Kích thước: L={load_obj['length']:.1f} x W={load_obj['width']:.1f}\n")
                f.write(f"  Trục X hiện tại: {calc['load_x_angle']:.1f}°\n")
                f.write(f"  Góc đặt mục tiêu: {calc['target_angle']:.1f}°\n")
                f.write(f"  Lệnh: {calc['theta4_command']}\n")
                f.write(f"  Hướng xoay: {calc['rotation_direction']}\n")
                f.write("\n")
        
        print(f"Đã lưu báo cáo: {report_path}")
    
    cv2.destroyAllWindows()

def main():
    """
    Menu chính để chạy demo.
    """
    print("DEMO TÍNH TOÁN THETA 4 CHO ROBOT IRB-460")
    print("Tích hợp với YOLO Detection và Module Division")
    print("="*50)
    print()
    print("Chương trình này sẽ:")
    print("1. Phân tích góc xoay từ YOLO OBB detection")
    print("2. Sử dụng Module Division để xác định layout đặt load")
    print("3. Tính toán góc xoay theta 4 cần thiết cho robot IRB-460")
    print("4. Hiển thị visualization và lưu kết quả")
    print()
    print("Giải thích hệ quy chiếu CUSTOM (theo yêu cầu X: Đông→Tây):")
    print("- Image coordinates: (0,0) ở góc trên-trái")
    print("- Trục X+: Hướng từ ĐÔNG sang TÂY (phải → trái) ←")
    print("- Trục Y+: Hướng xuống dưới (bắc → nam) ↓")
    print("- 0°: Từ đông sang tây (E→W) ←")
    print("- 90°: Xuống dưới (N→S) ↓")
    print("- 180°: Từ tây sang đông (W→E) →")
    print("- -90°: Lên trên (S→N) ↑")
    print("- Visualization có compass hiển thị hệ tọa độ custom này")
    print()
    
    input("Nhấn Enter để bắt đầu...")
    demo_single_image()

if __name__ == "__main__":
    main() 