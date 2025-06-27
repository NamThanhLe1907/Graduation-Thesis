"""
Theta4 Calculator với Module Division Integration
Tích hợp YOLO detection, module division và tính toán góc xoay theta4 cho robot IRB-460.

Chức năng chính:
1. Phát hiện pallets và loads bằng YOLO
2. Chia pallets thành regions theo Module Division
3. Mapping loads với regions phù hợp
4. Tính toán góc xoay theta4 cho từng load
5. Visualization regions và góc xoay

HỆ TỌA ĐỘ CUSTOM:
- Trục X+: Hướng từ ĐÔNG sang TÂY (phải → trái) ←
- Trục Y+: Hướng xuống dưới (bắc → nam) ↓  
- 0°: Từ đông sang tây (E→W) ←
- 90°: Xuống dưới (N→S) ↓
- 180°: Từ tây sang đông (W→E) →
- -90°: Lên trên (S→N) ↑
"""
import cv2
import os
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Any

# Import từ hệ thống hiện có
try:
    from detection import YOLOTensorRT
    from detection.utils.module_division import ModuleDivision
except ImportError:
    print("Lỗi import! Hãy đảm bảo đang chạy từ thư mục gốc của project")
    exit(1)

# Đường dẫn model
ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "best.engine")

class Theta4WithModuleDivision:
    """
    Class chính tích hợp Module Division và Theta4 Calculator.
    """
    
    def __init__(self, debug: bool = True):
        """
        Args:
            debug: Bật chế độ debug để hiển thị thông tin chi tiết
        """
        self.debug = debug
        self.module_divider = ModuleDivision(debug=debug)
        
        # Cấu hình target classes
        self.pallet_classes = [2.0]  # Class ID cho pallet
        self.load_classes = [0.0, 1.0]  # Class ID cho loads
        
    def normalize_angle(self, angle_deg: float) -> float:
        """Chuẩn hóa góc về khoảng [-180, 180]"""
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg <= -180:
            angle_deg += 360
        return angle_deg
    
    def analyze_object_dimensions_and_orientation(self, detections: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """
        Phân tích kích thước và hướng của objects từ YOLO OBB detection.
        Phân loại thành pallets và loads.
        
        Args:
            detections: Kết quả detection từ YOLO
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (pallet_objects, load_objects)
        """
        pallet_objects = []
        load_objects = []
        
        # Lấy thông tin từ detections
        obb_boxes = detections.get('obb_boxes', [])
        class_names = detections.get('class_names', [])
        classes = detections.get('classes', [])
        confidences = detections.get('confidences', [])
        
        if self.debug:
            print(f"\n=== PHÂN TÍCH OBJECTS ===")
            print(f"Số objects phát hiện: {len(obb_boxes)}")
        
        for i, obb in enumerate(obb_boxes):
            if len(obb) >= 5:  # [cx, cy, width, height, angle]
                cx, cy, width, height, angle_rad = obb[:5]
                
                # Thông tin cơ bản
                class_id = classes[i] if i < len(classes) else -1
                class_name = class_names[i] if i < len(class_names) else f"object_{i}"
                confidence = confidences[i] if i < len(confidences) else 1.0
                
                # Tính góc và kích thước
                angle_deg = math.degrees(angle_rad)
                angle_normalized = self.normalize_angle(angle_deg)
                
                # Xác định trục chính (chiều dài)
                if width >= height:
                    length = width
                    width_minor = height
                    major_axis_angle = angle_normalized
                else:
                    length = height
                    width_minor = width
                    major_axis_angle = self.normalize_angle(angle_normalized + 90)
                
                # Tạo object info
                object_info = {
                    'object_id': i,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'center': (cx, cy),
                    'length': length,
                    'width': width_minor,
                    'aspect_ratio': length / width_minor,
                    'original_angle_deg': angle_normalized,
                    'x_axis_angle': major_axis_angle,  # Góc trục X theo chiều dài
                    'original_obb': obb
                }
                
                # Phân loại dựa trên class_id
                if class_id in self.pallet_classes:
                    pallet_objects.append(object_info)
                    if self.debug:
                        print(f"PALLET {i}: {class_name} - L={length:.1f}, W={width_minor:.1f}, angle={major_axis_angle:.1f}°")
                elif class_id in self.load_classes:
                    load_objects.append(object_info)
                    if self.debug:
                        print(f"LOAD {i}: {class_name} - L={length:.1f}, W={width_minor:.1f}, angle={major_axis_angle:.1f}°")
                else:
                    # Unknown class, thêm vào loads theo mặc định
                    load_objects.append(object_info)
                    if self.debug:
                        print(f"UNKNOWN {i}: {class_name} - L={length:.1f}, W={width_minor:.1f}, angle={major_axis_angle:.1f}°")
        
        if self.debug:
            print(f"Kết quả phân loại: {len(pallet_objects)} pallets, {len(load_objects)} loads")
        
        return pallet_objects, load_objects
    
    def divide_pallets_into_regions(self, detections: Dict[str, Any], layer: int = 1) -> Dict[str, Any]:
        """
        Chia pallets thành regions sử dụng Module Division.
        
        Args:
            detections: Kết quả detection từ YOLO
            layer: Layer để chia (1 hoặc 2)
            
        Returns:
            Dict: Kết quả chia regions với thông tin chi tiết
        """
        if self.debug:
            print(f"\n=== CHIA PALLETS THÀNH REGIONS (Layer {layer}) ===")
        
        # Sử dụng Module Division để chia pallets
        divided_result = self.module_divider.process_pallet_detections(
            detections, 
            layer=layer, 
            target_classes=self.pallet_classes
        )
        
        # Chuẩn bị dữ liệu regions
        regions_data = self.module_divider.prepare_for_depth_estimation(divided_result)
        
        if self.debug:
            print(f"Tổng số regions: {len(regions_data)}")
            for i, region in enumerate(regions_data):
                info = region['region_info']
                center = region['center']
                print(f"  Region {i+1}: Pallet {info['pallet_id']}, Region {info['region_id']}, Center=({center[0]:.1f}, {center[1]:.1f})")
        
        return {
            'divided_result': divided_result,
            'regions_data': regions_data,
            'total_regions': len(regions_data),
            'layer': layer
        }
    
    def map_loads_to_regions(self, load_objects: List[Dict], regions_data: List[Dict]) -> List[Dict]:
        """
        Mapping loads với regions dựa trên proximity và availability.
        
        Args:
            load_objects: Danh sách load objects
            regions_data: Danh sách regions từ module division
            
        Returns:
            List[Dict]: Danh sách mapping load-region
        """
        if self.debug:
            print(f"\n=== MAPPING LOADS VỚI REGIONS ===")
            print(f"Loads: {len(load_objects)}, Regions: {len(regions_data)}")
        
        mappings = []
        used_regions = set()
        
        for load_idx, load_obj in enumerate(load_objects):
            load_center = np.array(load_obj['center'])
            best_region = None
            min_distance = float('inf')
            
            # Tìm region gần nhất chưa được sử dụng
            for region_idx, region in enumerate(regions_data):
                if region_idx in used_regions:
                    continue
                    
                region_center = np.array(region['center'])
                distance = np.linalg.norm(load_center - region_center)
                
                if distance < min_distance:
                    min_distance = distance
                    best_region = (region_idx, region)
            
            if best_region is not None:
                region_idx, region = best_region
                used_regions.add(region_idx)
                
                mapping = {
                    'load_object': load_obj,
                    'target_region': region,
                    'region_index': region_idx,
                    'distance': min_distance,
                    'mapping_success': True
                }
                
                if self.debug:
                    load_info = load_obj
                    region_info = region['region_info']
                    print(f"  Load {load_idx} ({load_info['class_name']}) → Region {region_idx} (Pallet {region_info['pallet_id']}, Region {region_info['region_id']})")
                    print(f"    Distance: {min_distance:.1f}px")
            else:
                mapping = {
                    'load_object': load_obj,
                    'target_region': None,
                    'region_index': -1,
                    'distance': -1,
                    'mapping_success': False
                }
                
                if self.debug:
                    print(f"  Load {load_idx} ({load_obj['class_name']}) → NO AVAILABLE REGION")
            
            mappings.append(mapping)
        
        successful_mappings = sum(1 for m in mappings if m['mapping_success'])
        if self.debug:
            print(f"Mapping thành công: {successful_mappings}/{len(load_objects)}")
        
        return mappings
    
    def determine_placement_angle_for_region(self, region_info: Dict, layer: int) -> float:
        """
        Xác định góc đặt load dựa trên region và layer.
        
        Args:
            region_info: Thông tin region
            layer: Layer hiện tại
            
        Returns:
            float: Góc đặt mục tiêu (degrees)
        """
        region_id = region_info.get('region_id', 1)
        pallet_id = region_info.get('pallet_id', 1)
        
        # Hướng chuẩn của robot (có thể cấu hình)
        robot_base_angle = 0.0  # Đông→Tây (E→W)
        
        if layer == 1:
            # Layer 1: Tất cả regions theo hướng chuẩn
            placement_angle = robot_base_angle
        elif layer == 2:
            # Layer 2: Xen kẽ hướng
            if region_id % 2 == 1:  # Region lẻ
                placement_angle = robot_base_angle
            else:  # Region chẵn
                placement_angle = self.normalize_angle(robot_base_angle + 90.0)
        else:
            placement_angle = robot_base_angle
        
        if self.debug:
            print(f"    Placement angle cho Region {region_id} (Layer {layer}): {placement_angle:.1f}°")
        
        return placement_angle
    
    def calculate_theta4_for_mapping(self, mapping: Dict, layer: int) -> Dict[str, Any]:
        """
        Tính toán theta4 cho một mapping load-region.
        
        Args:
            mapping: Mapping load-region
            layer: Layer hiện tại
            
        Returns:
            Dict: Kết quả tính toán theta4
        """
        load_obj = mapping['load_object']
        
        if not mapping['mapping_success']:
            return {
                'mapping': mapping,
                'theta4_success': False,
                'error': 'No target region available',
                'theta4_command': 'NO PLACEMENT POSSIBLE'
            }
        
        region = mapping['target_region']
        region_info = region['region_info']
        
        # Xác định góc đặt cho region này
        target_angle = self.determine_placement_angle_for_region(region_info, layer)
        
        # Lấy góc hiện tại của load
        load_current_angle = load_obj['x_axis_angle']
        
        # Tính góc xoay cần thiết
        rotation_needed = target_angle - load_current_angle
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
        
        result = {
            'mapping': mapping,
            'theta4_success': True,
            'load_current_angle': load_current_angle,
            'target_angle': target_angle,
            'rotation_needed': rotation_needed,
            'rotation_normalized': rotation_normalized,
            'rotation_direction': rotation_direction,
            'rotation_sign': rotation_sign,
            'theta4_value': rotation_normalized,
            'theta4_command': f"THETA4 = {rotation_sign}{abs(rotation_normalized):.1f}°",
            'region_info': region_info
        }
        
        if self.debug:
            load_name = load_obj['class_name']
            print(f"\n  THETA4 cho {load_name}:")
            print(f"    Load angle: {load_current_angle:.1f}° → Target: {target_angle:.1f}°")
            print(f"    Rotation: {rotation_sign}{abs(rotation_normalized):.1f}° ({rotation_direction})")
            print(f"    Command: {result['theta4_command']}")
        
        return result
    
    def process_full_pipeline(self, detections: Dict[str, Any], layer: int = 1) -> Dict[str, Any]:
        """
        Thực hiện toàn bộ pipeline từ detection đến theta4 calculation.
        
        Args:
            detections: Kết quả YOLO detection
            layer: Layer để chia pallet
            
        Returns:
            Dict: Kết quả hoàn chỉnh
        """
        if self.debug:
            print(f"\n" + "="*60)
            print(f"FULL PIPELINE: THETA4 WITH MODULE DIVISION (Layer {layer})")
            print(f"="*60)
        
        start_time = time.time()
        
        # Bước 1: Phân tích objects
        pallet_objects, load_objects = self.analyze_object_dimensions_and_orientation(detections)
        
        # Bước 2: Chia pallets thành regions
        regions_result = self.divide_pallets_into_regions(detections, layer)
        
        # Bước 3: Mapping loads với regions
        mappings = self.map_loads_to_regions(load_objects, regions_result['regions_data'])
        
        # Bước 4: Tính toán theta4 cho từng mapping
        if self.debug:
            print(f"\n=== TÍNH TOÁN THETA4 ===")
        
        theta4_results = []
        for i, mapping in enumerate(mappings):
            if self.debug:
                print(f"\nMapping {i+1}:")
            
            theta4_calc = self.calculate_theta4_for_mapping(mapping, layer)
            theta4_results.append(theta4_calc)
        
        processing_time = time.time() - start_time
        
        # Tổng hợp kết quả
        result = {
            'pallet_objects': pallet_objects,
            'load_objects': load_objects,
            'regions_result': regions_result,
            'mappings': mappings,
            'theta4_results': theta4_results,
            'layer': layer,
            'processing_time': processing_time,
            'summary': {
                'total_pallets': len(pallet_objects),
                'total_loads': len(load_objects),
                'total_regions': regions_result['total_regions'],
                'successful_mappings': sum(1 for m in mappings if m['mapping_success']),
                'successful_theta4': sum(1 for t in theta4_results if t['theta4_success'])
            }
        }
        
        if self.debug:
            print(f"\n=== SUMMARY ===")
            print(f"Processing time: {processing_time*1000:.2f} ms")
            print(f"Pallets: {result['summary']['total_pallets']}")
            print(f"Loads: {result['summary']['total_loads']}")
            print(f"Regions: {result['summary']['total_regions']}")
            print(f"Successful mappings: {result['summary']['successful_mappings']}")
            print(f"Successful theta4: {result['summary']['successful_theta4']}")
        
        return result
    
    def create_comprehensive_visualization(self, image: np.ndarray, pipeline_result: Dict) -> np.ndarray:
        """
        Tạo visualization hoàn chỉnh với regions và theta4 info.
        
        Args:
            image: Ảnh gốc
            pipeline_result: Kết quả từ process_full_pipeline
            
        Returns:
            np.ndarray: Ảnh visualization
        """
        result_image = image.copy()
        
        # Màu sắc
        pallet_color = (255, 0, 0)      # Đỏ cho pallet
        load_color = (0, 255, 0)        # Xanh lá cho load
        region_color = (0, 255, 255)    # Vàng cho region boundaries
        mapping_color = (255, 0, 255)   # Magenta cho mapping lines
        
        # Vẽ pallet objects
        for pallet in pipeline_result['pallet_objects']:
            self.draw_object_with_orientation(result_image, pallet, pallet_color, "PALLET")
        
        # Vẽ regions
        regions_data = pipeline_result['regions_result']['regions_data']
        for i, region in enumerate(regions_data):
            self.draw_region_boundary(result_image, region, region_color, i+1)
        
        # Vẽ load objects và mappings
        theta4_results = pipeline_result['theta4_results']
        for i, theta4_result in enumerate(theta4_results):
            mapping = theta4_result['mapping']
            load_obj = mapping['load_object']
            
            # Vẽ load với thông tin theta4
            self.draw_object_with_orientation(result_image, load_obj, load_color, "LOAD")
            
            # Vẽ đường mapping nếu thành công
            if mapping['mapping_success']:
                target_region = mapping['target_region']
                load_center = load_obj['center']
                region_center = target_region['center']
                
                cv2.line(result_image, 
                        (int(load_center[0]), int(load_center[1])),
                        (int(region_center[0]), int(region_center[1])),
                        mapping_color, 2)
                
                # Vẽ theta4 command gần load
                theta4_text = theta4_result.get('theta4_command', 'NO THETA4')
                self.draw_text_with_background(result_image, theta4_text,
                                             (int(load_center[0]) + 10, int(load_center[1]) + 30),
                                             (255, 255, 255))
        
        # Vẽ summary info ở góc trên
        self.draw_summary_info(result_image, pipeline_result)
        
        # Vẽ coordinate compass
        self.draw_coordinate_compass(result_image)
        
        return result_image
    
    def draw_object_with_orientation(self, image: np.ndarray, obj: Dict, color: Tuple[int, int, int], obj_type: str):
        """Vẽ object với orientation arrow"""
        cx, cy = obj['center']
        x_axis_angle = obj['x_axis_angle']
        length = obj['length']
        width = obj['width']
        
        # Vẽ center point
        cv2.circle(image, (int(cx), int(cy)), 5, color, -1)
        
        # Vẽ orientation arrow (theo hệ tọa độ custom)
        arrow_length = 50
        end_x = cx - arrow_length * math.cos(math.radians(x_axis_angle))
        end_y = cy - arrow_length * math.sin(math.radians(x_axis_angle))
        cv2.arrowedLine(image, (int(cx), int(cy)), 
                       (int(end_x), int(end_y)), color, 3)
        
        # Vẽ thông tin
        text1 = f"{obj_type}: L={length:.0f}, W={width:.0f}"
        text2 = f"Angle: {x_axis_angle:.1f}°"
        self.draw_text_with_background(image, text1, 
                                     (int(cx) + 10, int(cy) - 20), color)
        self.draw_text_with_background(image, text2, 
                                     (int(cx) + 10, int(cy) - 5), color)
    
    def draw_region_boundary(self, image: np.ndarray, region: Dict, color: Tuple[int, int, int], region_num: int):
        """Vẽ boundary của region"""
        if 'corners' in region:
            # Vẽ OBB corners
            corners = region['corners']
            corners_int = [(int(x), int(y)) for x, y in corners]
            cv2.polylines(image, [np.array(corners_int)], True, color, 2)
        else:
            # Vẽ regular bounding box
            bbox = region['bbox']
            x1, y1, x2, y2 = [int(x) for x in bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ region number
        center = region['center']
        self.draw_text_with_background(image, f"R{region_num}", 
                                     (int(center[0]) - 10, int(center[1])), 
                                     color, (0, 0, 0))
    
    def draw_summary_info(self, image: np.ndarray, pipeline_result: Dict):
        """Vẽ thông tin summary"""
        summary = pipeline_result['summary']
        layer = pipeline_result['layer']
        
        y_offset = 20
        line_height = 20
        
        info_lines = [
            f"Layer: {layer}",
            f"Pallets: {summary['total_pallets']} | Loads: {summary['total_loads']}",
            f"Regions: {summary['total_regions']} | Mappings: {summary['successful_mappings']}",
            f"Theta4 Success: {summary['successful_theta4']}/{summary['total_loads']}",
            f"Processing: {pipeline_result['processing_time']*1000:.1f}ms"
        ]
        
        for i, line in enumerate(info_lines):
            self.draw_text_with_background(image, line,
                                         (10, y_offset + i * line_height),
                                         (255, 255, 255), (0, 0, 0))
    
    def draw_text_with_background(self, image: np.ndarray, text: str, 
                                position: Tuple[int, int], color: Tuple[int, int, int], 
                                bg_color: Tuple[int, int, int] = (0, 0, 0)):
        """Vẽ text với background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        cv2.rectangle(image, 
                     (position[0] - 2, position[1] - text_size[1] - 2),
                     (position[0] + text_size[0] + 2, position[1] + 2),
                     bg_color, -1)
        
        cv2.putText(image, text, position, font, font_scale, color, thickness)
    
    def draw_coordinate_compass(self, image: np.ndarray):
        """Vẽ compass hệ tọa độ custom"""
        h, w = image.shape[:2]
        center_x = w - 80
        center_y = h - 80
        
        # Background
        cv2.rectangle(image, (center_x - 50, center_y - 50), 
                     (center_x + 50, center_y + 50), (0, 0, 0), -1)
        cv2.rectangle(image, (center_x - 50, center_y - 50), 
                     (center_x + 50, center_y + 50), (255, 255, 255), 2)
        
        # Trục X+ (đỏ, E→W)
        cv2.arrowedLine(image, (center_x, center_y), 
                       (center_x - 35, center_y), (0, 0, 255), 2)
        cv2.putText(image, "X+", (center_x - 50, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Trục Y+ (xanh lá, N→S)
        cv2.arrowedLine(image, (center_x, center_y), 
                       (center_x, center_y + 35), (0, 255, 0), 2)
        cv2.putText(image, "Y+", (center_x + 5, center_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

def demo_theta4_with_module_division():
    """
    Demo hoàn chỉnh theta4 calculation với module division.
    """
    print("=== DEMO THETA4 WITH MODULE DIVISION ===")
    print("Tích hợp YOLO Detection + Module Division + Theta4 Calculation")
    print()
    
    # Chọn ảnh
    pallets_folder = "load_on_robot_images"
    if os.path.exists(pallets_folder):
        image_files = [f for f in os.listdir(pallets_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()
        
        print("Chọn ảnh để phân tích:")
        for i, img_file in enumerate(image_files, 1):
            print(f"  {i}. {img_file}")
        print(f"  0. Nhập đường dẫn khác")
        
        choice = input(f"\nChọn ảnh (1-{len(image_files)}) hoặc 0: ")
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(image_files):
                image_path = os.path.join(pallets_folder, image_files[choice_num - 1])
            elif choice_num == 0:
                image_path = input("Nhập đường dẫn khác: ")
            else:
                image_path = os.path.join(pallets_folder, image_files[0])
        except ValueError:
            image_path = os.path.join(pallets_folder, image_files[0])
    else:
        image_path = input("Nhập đường dẫn ảnh: ")
    
    # Đọc ảnh
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    print(f"\nĐang xử lý ảnh: {image_path}")
    
    # Chọn layer
    layer_choice = input("Chọn layer (1 hoặc 2, mặc định 1): ")
    try:
        layer = int(layer_choice) if layer_choice else 1
    except ValueError:
        layer = 1
    
    # Khởi tạo models
    print(f"\nKhởi tạo models...")
    yolo_model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.5)
    theta4_calculator = Theta4WithModuleDivision(debug=True)
    
    # Thực hiện YOLO detection
    print(f"\nThực hiện YOLO detection...")
    start_time = time.time()
    detections = yolo_model.detect(frame)
    yolo_time = time.time() - start_time
    print(f"YOLO time: {yolo_time*1000:.2f} ms")
    
    if len(detections.get('bounding_boxes', [])) == 0:
        print("Không phát hiện object nào!")
        return
    
    # Thực hiện full pipeline
    print(f"\nThực hiện full pipeline...")
    pipeline_result = theta4_calculator.process_full_pipeline(detections, layer)
    
    # Tạo visualization
    print(f"\nTạo visualization...")
    vis_image = theta4_calculator.create_comprehensive_visualization(frame, pipeline_result)
    
    # Hiển thị kết quả
    print(f"\n" + "="*60)
    print(f"KẾT QUẢ THETA4 COMMANDS")
    print(f"="*60)
    
    theta4_results = pipeline_result['theta4_results']
    for i, result in enumerate(theta4_results):
        if result['theta4_success']:
            load_obj = result['mapping']['load_object']
            region_info = result['region_info']
            
            print(f"\nLoad #{i+1} - {load_obj['class_name']}:")
            print(f"  Target: Pallet {region_info['pallet_id']}, Region {region_info['region_id']}")
            print(f"  Current angle: {result['load_current_angle']:.1f}°")
            print(f"  Target angle: {result['target_angle']:.1f}°")
            print(f"  ➤ ROBOT COMMAND: {result['theta4_command']}")
        else:
            load_obj = result['mapping']['load_object']
            print(f"\nLoad #{i+1} - {load_obj['class_name']}: {result['error']}")
    
    # Hiển thị ảnh
    cv2.imshow("YOLO Detection", detections["annotated_frame"])
    cv2.imshow("Theta4 + Module Division", vis_image)
    
    print(f"\nNhấn phím bất kỳ để tiếp tục...")
    cv2.waitKey(0)
    
    # Lưu kết quả
    save_choice = input("\nBạn có muốn lưu kết quả? (y/n): ").lower()
    if save_choice in ['y', 'yes']:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Lưu ảnh
        output_path = f"theta4_module_division_{base_name}_layer{layer}.jpg"
        cv2.imwrite(output_path, vis_image)
        print(f"Đã lưu: {output_path}")
        
        # Lưu báo cáo
        report_path = f"theta4_module_division_report_{base_name}_layer{layer}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("BÁO CÁO THETA4 WITH MODULE DIVISION\n")
            f.write("="*60 + "\n")
            f.write(f"Ảnh: {image_path}\n")
            f.write(f"Layer: {layer}\n")
            f.write(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing time: {pipeline_result['processing_time']*1000:.2f} ms\n")
            f.write("\n")
            
            summary = pipeline_result['summary']
            f.write("SUMMARY:\n")
            f.write(f"  Pallets: {summary['total_pallets']}\n")
            f.write(f"  Loads: {summary['total_loads']}\n")
            f.write(f"  Regions: {summary['total_regions']}\n")
            f.write(f"  Successful mappings: {summary['successful_mappings']}\n")
            f.write(f"  Successful theta4: {summary['successful_theta4']}\n")
            f.write("\n")
            
            f.write("THETA4 COMMANDS:\n")
            f.write("-" * 20 + "\n")
            for i, result in enumerate(theta4_results):
                if result['theta4_success']:
                    load_obj = result['mapping']['load_object']
                    region_info = result['region_info']
                    f.write(f"Load #{i+1} - {load_obj['class_name']}:\n")
                    f.write(f"  Target: Pallet {region_info['pallet_id']}, Region {region_info['region_id']}\n")
                    f.write(f"  Command: {result['theta4_command']}\n")
                    f.write(f"  Rotation: {result['rotation_direction']}\n")
                    f.write("\n")
        
        print(f"Đã lưu báo cáo: {report_path}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo_theta4_with_module_division() 