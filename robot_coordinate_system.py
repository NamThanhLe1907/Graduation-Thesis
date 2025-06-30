"""
Hệ thống tọa độ Robot cho Theta4 Calculator
- Sử dụng hệ quy chiếu robot: X (phải→trái), Y (trên→dưới)  
- Xác định phương X của object dựa vào cạnh chiều dài
- Chuyển đổi từ YOLO OBB format sang robot coordinates
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any

class RobotCoordinateSystem:
    """
    Xử lý chuyển đổi tọa độ và góc từ YOLO detection sang hệ quy chiếu robot.
    
    Hệ quy chiếu Robot:
    - X axis: Phải → Trái (Đông → Tây)
    - Y axis: Trên → Dưới (Bắc → Nam)
    - Góc dương: Counter-clockwise 
    - Range: [-180°, 180°]
    
    Quy ước object orientation:
    - Phương X của object = Hướng của cạnh chiều dài
    - Phương Y của object = Hướng của cạnh chiều rộng
    """
    
    def __init__(self, debug: bool = True):
        """
        Args:
            debug: Bật debug để hiển thị thông tin chi tiết
        """
        self.debug = debug
        
    def normalize_angle(self, angle_deg: float) -> float:
        """Chuẩn hóa góc về khoảng [-180, 180]"""
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg <= -180:
            angle_deg += 360
        return angle_deg
    
    def convert_image_to_robot_coordinates(self, image_x: float, image_y: float, 
                                         image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Chuyển đổi tọa độ từ image coordinates sang robot coordinates.
        
        Args:
            image_x, image_y: Tọa độ trong image (pixel)
            image_width, image_height: Kích thước ảnh
            
        Returns:
            Tuple[float, float]: Tọa độ trong robot coordinates (normalized [0,1])
        """
        # Normalize về [0,1]
        norm_x = image_x / image_width
        norm_y = image_y / image_height
        
        # Chuyển đổi sang robot coordinates
        # Image X (trái→phải) = Robot X (phải→trái) đảo ngược
        robot_x = 1.0 - norm_x  # Đảo ngược X
        robot_y = norm_y        # Giữ nguyên Y
        
        return robot_x, robot_y
    
    def convert_image_angle_to_robot_angle(self, image_angle_deg: float) -> float:
        """
        Chuyển đổi góc từ image coordinates sang robot coordinates.
        
        Args:
            image_angle_deg: Góc trong image coordinates
            
        Returns:
            float: Góc trong robot coordinates
        """
        # Đảo ngược trục X nên góc cũng phải điều chỉnh
        # Góc 0° trong image (hướng phải) = góc 180° trong robot (hướng trái)
        robot_angle = 180.0 - image_angle_deg
        
        return self.normalize_angle(robot_angle)
    
    def determine_object_main_axis(self, width: float, height: float, 
                                 yolo_angle_rad: float) -> Dict[str, Any]:
        """
        Xác định trục chính (phương X) của object dựa vào cạnh dài.
        
        Args:
            width: Chiều rộng từ YOLO OBB
            height: Chiều cao từ YOLO OBB  
            yolo_angle_rad: Góc xoay từ YOLO (radian)
            
        Returns:
            Dict với thông tin về trục chính của object
        """
        # Chuyển YOLO angle từ radian sang độ
        yolo_angle_deg = math.degrees(yolo_angle_rad)
        
        # Xác định cạnh nào dài hơn
        if width >= height:
            # Width là cạnh dài → Width direction = X axis của object
            main_axis_length = width
            secondary_axis_length = height
            main_axis_type = "width"
            
            # Góc của width direction chính là YOLO angle
            object_x_angle_image = yolo_angle_deg
            
        else:
            # Height là cạnh dài → Height direction = X axis của object  
            main_axis_length = height
            secondary_axis_length = width
            main_axis_type = "height"
            
            # Góc của height direction = YOLO angle + 90°
            object_x_angle_image = yolo_angle_deg + 90.0
        
        # Chuẩn hóa góc trong image coordinates
        object_x_angle_image = self.normalize_angle(object_x_angle_image)
        
        # Chuyển đổi sang robot coordinates
        object_x_angle_robot = self.convert_image_angle_to_robot_angle(object_x_angle_image)
        
        return {
            'main_axis_length': main_axis_length,
            'secondary_axis_length': secondary_axis_length,
            'main_axis_type': main_axis_type,
            'aspect_ratio': main_axis_length / secondary_axis_length,
            'yolo_angle_deg': yolo_angle_deg,
            'object_x_angle_image': object_x_angle_image,
            'object_x_angle_robot': object_x_angle_robot,
            'object_orientation': self.describe_robot_orientation(object_x_angle_robot)
        }
    
    def describe_robot_orientation(self, angle_deg: float) -> Dict[str, str]:
        """
        Mô tả hướng của object trong hệ tọa độ robot.
        
        Args:
            angle_deg: Góc trong robot coordinates
            
        Returns:
            Dict với mô tả hướng
        """
        angle_abs = abs(angle_deg)
        
        if -5 <= angle_deg <= 5:
            direction = "→ ĐÔNG"
            description = "Hướng Đông (X+ robot)"
            compass = "E"
        elif 175 <= angle_abs <= 180:
            direction = "← TÂY" 
            description = "Hướng Tây (X- robot)"
            compass = "W"
        elif 85 <= angle_deg <= 95:
            direction = "↑ BẮC"
            description = "Hướng Bắc (Y- robot)"
            compass = "N"
        elif -95 <= angle_deg <= -85:
            direction = "↓ NAM"
            description = "Hướng Nam (Y+ robot)" 
            compass = "S"
        elif 0 < angle_deg < 90:
            direction = "↗ ĐÔNG BẮC"
            description = "Hướng Đông Bắc"
            compass = "NE"
        elif 90 < angle_deg < 180:
            direction = "↖ TÂY BẮC"
            description = "Hướng Tây Bắc"
            compass = "NW"
        elif -90 < angle_deg < 0:
            direction = "↘ ĐÔNG NAM"
            description = "Hướng Đông Nam"
            compass = "SE"
        elif -180 < angle_deg < -90:
            direction = "↙ TÂY NAM"
            description = "Hướng Tây Nam"
            compass = "SW"
        else:
            direction = "? UNKNOWN"
            description = "Không xác định"
            compass = "?"
            
        return {
            'direction': direction,
            'description': description,
            'compass': compass,
            'quadrant': self.get_quadrant(angle_deg)
        }
    
    def get_quadrant(self, angle_deg: float) -> str:
        """Xác định góc phần tư trong robot coordinates"""
        if 0 <= angle_deg < 90:
            return "Quadrant I (NE)"
        elif 90 <= angle_deg < 180:
            return "Quadrant II (NW)"
        elif -180 <= angle_deg < -90:
            return "Quadrant III (SW)"
        elif -90 <= angle_deg < 0:
            return "Quadrant IV (SE)"
        else:
            return "Boundary"
    
    def analyze_object_from_yolo_obb(self, obb_data: List[float], 
                                   class_name: str = "object") -> Dict[str, Any]:
        """
        Phân tích hoàn chỉnh object từ YOLO OBB data.
        
        Args:
            obb_data: [cx, cy, width, height, angle_rad] từ YOLO
            class_name: Tên class của object
            
        Returns:
            Dict với thông tin phân tích đầy đủ
        """
        if len(obb_data) < 5:
            raise ValueError("OBB data cần có ít nhất 5 phần tử [cx, cy, width, height, angle]")
        
        cx, cy, width, height, angle_rad = obb_data[:5]
        
        # Phân tích trục chính
        main_axis_info = self.determine_object_main_axis(width, height, angle_rad)
        
        # Thông tin tổng hợp
        result = {
            'class_name': class_name,
            'center_image': (cx, cy),
            'size': (width, height),
            'yolo_angle_rad': angle_rad,
            'main_axis_info': main_axis_info,
            'object_x_direction': {
                'angle_robot': main_axis_info['object_x_angle_robot'],
                'orientation': main_axis_info['object_orientation']
            },
            'summary': {
                'main_axis': f"{main_axis_info['main_axis_type']} ({main_axis_info['main_axis_length']:.1f}px)",
                'aspect_ratio': f"{main_axis_info['aspect_ratio']:.2f}:1",
                'robot_orientation': main_axis_info['object_orientation']['description']
            }
        }
        
        if self.debug:
            self.print_object_analysis(result)
            
        return result
    
    def print_object_analysis(self, analysis: Dict[str, Any]):
        """In thông tin phân tích object để debug"""
        print(f"\n📦 OBJECT: {analysis['class_name']}")
        print(f"   Center: {analysis['center_image']}")
        print(f"   Size: {analysis['size'][0]:.1f} x {analysis['size'][1]:.1f}")
        print(f"   YOLO angle: {math.degrees(analysis['yolo_angle_rad']):.1f}°")
        print(f"   Main axis: {analysis['summary']['main_axis']}")
        print(f"   Aspect ratio: {analysis['summary']['aspect_ratio']}")
        print(f"   🧭 Object X direction: {analysis['object_x_direction']['orientation']['direction']}")
        print(f"   📐 Robot angle: {analysis['object_x_direction']['angle_robot']:.1f}°")
        print(f"   📍 Description: {analysis['summary']['robot_orientation']}")

def test_coordinate_system():
    """Test function để kiểm tra hệ tọa độ"""
    print("="*80)
    print("KIỂM TRA HỆ TỌA ĐỘ ROBOT")
    print("="*80)
    
    coords = RobotCoordinateSystem(debug=True)
    
    # Test cases với các ví dụ khác nhau
    test_cases = [
        {
            'name': 'Load nằm ngang (width > height)',
            'obb': [640, 360, 200, 100, 0],  # [cx, cy, w, h, angle_rad]
            'class': 'load'
        },
        {
            'name': 'Load đứng (height > width)', 
            'obb': [640, 360, 100, 200, 0],
            'class': 'load'
        },
        {
            'name': 'Load xoay 45° (width > height)',
            'obb': [640, 360, 200, 100, math.radians(45)],
            'class': 'load'
        },
        {
            'name': 'Pallet vuông',
            'obb': [640, 360, 150, 150, math.radians(30)],
            'class': 'pallet'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*20} TEST CASE {i}: {test['name']} {'='*20}")
        analysis = coords.analyze_object_from_yolo_obb(test['obb'], test['class'])

if __name__ == "__main__":
    test_coordinate_system() 