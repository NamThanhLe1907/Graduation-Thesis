"""
Calculator cho góc xoay theta 4 robot IRB-460.
Tích hợp với YOLO detection và module division để tính toán góc xoay cần thiết.
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
            print(f"\n=== PHÂN TÍCH GÓC XOAY YOLO ===")
            print(f"Số objects phát hiện: {len(obb_boxes)}")
        
        for i, obb in enumerate(obb_boxes):
            if len(obb) >= 5:  # [cx, cy, width, height, angle]
                cx, cy, width, height, angle_rad = obb[:5]
                
                # Chuyển từ radian sang độ
                angle_deg = math.degrees(angle_rad)
                
                # Chuẩn hóa góc về [-180, 180]
                angle_normalized = self.normalize_angle(angle_deg)
                
                object_info = {
                    'object_id': i,
                    'class_name': class_names[i] if i < len(class_names) else f"object_{i}",
                    'confidence': confidences[i] if i < len(confidences) else 1.0,
                    'center': (cx, cy),
                    'size': (width, height),
                    'angle_rad': angle_rad,
                    'angle_deg': angle_deg,
                    'angle_normalized': angle_normalized
                }
                
                results.append(object_info)
                
                if self.debug:
                    print(f"\nObject {i} ({object_info['class_name']}):")
                    print(f"  Center: ({cx:.1f}, {cy:.1f})")
                    print(f"  Size: {width:.1f} x {height:.1f}")
                    print(f"  Góc gốc: {angle_deg:.1f}°")
                    print(f"  Góc chuẩn hóa: {angle_normalized:.1f}°")
                    
                    # Phân tích hướng
                    if abs(angle_normalized) < 5:
                        direction = "Horizontal (→)"
                    elif 85 < abs(angle_normalized) < 95:
                        direction = "Vertical (↑/↓)"
                    elif 0 < angle_normalized < 90:
                        direction = "NE (↗)"
                    elif 90 < angle_normalized < 180:
                        direction = "NW (↖)"
                    elif -90 < angle_normalized < 0:
                        direction = "SE (↘)"
                    elif -180 < angle_normalized < -90:
                        direction = "SW (↙)"
                    else:
                        direction = "Unknown"
                    
                    print(f"  Hướng: {direction}")
        
        return results
    
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
        # Pattern mẫu cho layer 1: đặt theo hàng ngang
        if layer == 1:
            return 0.0  # Tất cả đặt nằm ngang
        
        # Pattern mẫu cho layer 2: đặt xen kẽ
        elif layer == 2:
            if region_id % 2 == 1:  # Region lẻ
                return 0.0  # Nằm ngang
            else:  # Region chẵn
                return 90.0  # Thẳng đứng
        
        # Mặc định
        return 0.0
    
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
    
    def create_visualization(self, image: np.ndarray, yolo_angles: List[Dict], 
                           theta4_calculations: List[Dict]) -> np.ndarray:
        """
        Tạo ảnh visualization với thông tin góc xoay và theta 4.
        
        Args:
            image: Ảnh gốc
            yolo_angles: Thông tin góc từ YOLO
            theta4_calculations: Kết quả tính toán theta 4
            
        Returns:
            np.ndarray: Ảnh đã được vẽ thông tin
        """
        result_image = image.copy()
        
        # Màu sắc
        load_color = (0, 255, 0)    # Xanh lá cho load
        pallet_color = (255, 0, 0)  # Đỏ cho pallet
        
        # Vẽ objects với arrows chỉ hướng
        for obj in yolo_angles:
            cx, cy = obj['center']
            angle = obj['angle_normalized']
            class_name = obj['class_name']
            
            # Chọn màu dựa trên class
            if 'load' in class_name.lower():
                color = load_color
            else:
                color = pallet_color
            
            # Vẽ điểm center
            cv2.circle(result_image, (int(cx), int(cy)), 5, color, -1)
            
            # Vẽ vector hướng
            length = 50
            end_x = cx + length * math.cos(math.radians(angle))
            end_y = cy + length * math.sin(math.radians(angle))
            cv2.arrowedLine(result_image, (int(cx), int(cy)), 
                           (int(end_x), int(end_y)), color, 3)
            
            # Vẽ text thông tin
            text = f"{class_name}: {angle:.1f}°"
            self.draw_text_with_background(result_image, text, 
                                         (int(cx) + 10, int(cy) - 10), color)
        
        # Vẽ thông tin theta 4 ở góc trên
        y_offset = 30
        for i, calc in enumerate(theta4_calculations):
            theta4_text = f"THETA4 #{i+1}: {calc['theta4_command']}"
            self.draw_text_with_background(result_image, theta4_text, 
                                         (10, y_offset + i * 30), (255, 255, 255))
            
            rotation_text = f"  Xoay {calc['rotation_direction']}"
            self.draw_text_with_background(result_image, rotation_text, 
                                         (10, y_offset + i * 30 + 20), (200, 200, 200))
        
        return result_image
    
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

def demo_single_image():
    """
    Demo phân tích góc xoay trên một ảnh đơn lẻ.
    """
    print("=== DEMO PHÂN TÍCH GÓC XOAY VÀ TÍNH TOÁN THETA 4 ===")
    print("Chọn ảnh để phân tích:")
    
    # Hiển thị ảnh có sẵn
    pallets_folder = "images_pallets2"
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
    yolo_model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)
    
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
    
    start_time = time.time()
    
    # Phân tích góc YOLO
    yolo_angles = calculator.analyze_yolo_angles(detections)
    
    # Tìm load objects (không phải pallet)
    load_objects = [obj for obj in yolo_angles if 'load' in obj['class_name'].lower()]
    pallet_objects = [obj for obj in yolo_angles if 'load' not in obj['class_name'].lower()]
    
    print(f"\nPhân loại objects:")
    print(f"  Load objects: {len(load_objects)}")
    print(f"  Pallet objects: {len(pallet_objects)}")
    print(f"  Regions: {len(division_regions)}")
    
    # Tính toán theta 4 cho mỗi load
    theta4_calculations = []
    
    for i, load_obj in enumerate(load_objects):
        load_angle = load_obj['angle_normalized']
        
        # Giả sử đặt vào region tương ứng
        if i < len(division_regions):
            region_data = division_regions[i]
            region_info = region_data.get('region_info', {})
            
            pallet_id = region_info.get('pallet_id', 1)
            region_id = region_info.get('region_id', i+1)
            
            # Xác định góc đặt
            target_angle = calculator.determine_placement_angle(pallet_id, region_id, layer)
            
            # Tính toán theta 4
            theta4_calc = calculator.calculate_theta4_rotation(load_angle, target_angle)
            theta4_calc['load_object'] = load_obj
            theta4_calc['target_region'] = region_info
            
            theta4_calculations.append(theta4_calc)
            
            print(f"\nLoad #{i+1}:")
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
            print(f"\nLoad #{i+1}:")
            print(f"  Góc hiện tại trên băng tải: {calc['load_angle']:.1f}°")
            print(f"  Góc đặt mục tiêu trên pallet: {calc['target_angle']:.1f}°")
            print(f"  Góc xoay cần thiết: {calc['rotation_sign']}{abs(calc['rotation_normalized']):.1f}°")
            print(f"  Hướng xoay: {calc['rotation_direction']}")
            print(f"  ➤ LỆNH ROBOT: {calc['theta4_command']}")
    else:
        print("\nKhông tìm thấy load objects để tính toán theta 4!")
    
    # Tạo visualization
    print(f"\n--- TẠO VISUALIZATION ---")
    vis_image = calculator.create_visualization(frame, yolo_angles, theta4_calculations)
    
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
            for obj in yolo_angles:
                f.write(f"{obj['class_name']}: {obj['angle_normalized']:.1f}°\n")
            f.write("\n")
            
            f.write("LỆNH THETA 4:\n")
            f.write("-" * 20 + "\n")
            for i, calc in enumerate(theta4_calculations):
                f.write(f"Load #{i+1}: {calc['theta4_command']}\n")
                f.write(f"  Xoay: {calc['rotation_direction']}\n")
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
    print("Giải thích hệ quy chiếu:")
    print("- Image coordinates: (0,0) ở góc trên-trái")
    print("- Góc dương: Counter-clockwise (ngược chiều kim đồng hồ)")
    print("- Góc âm: Clockwise (cùng chiều kim đồng hồ)")
    print("- Range: [-180°, 180°]")
    print()
    
    input("Nhấn Enter để bắt đầu...")
    demo_single_image()

if __name__ == "__main__":
    main() 