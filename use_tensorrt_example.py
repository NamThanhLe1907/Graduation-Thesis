"""
Ví dụ sử dụng model TensorRT cho phát hiện đối tượng
"""
import cv2
import time
import os
import threading
import numpy as np
from detection import (YOLOTensorRT,
                       ProcessingPipeline,
                       CameraInterface,
                       DepthEstimator,
                       ModuleDivision,
                       RegionManager)
from region_division_plc_integration import RegionDivisionPLCIntegration

# Đường dẫn tới file model - sử dụng .pt thay vì .engine để tránh lỗi version
ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "utils", "detection", "best.engine")
# Cấu hình hiển thị depth - mặc định là False để tránh lag
SHOW_DEPTH = os.environ.get('SHOW_DEPTH', 'false').lower() in ('true', '1', 'yes')
# Cấu hình hiển thị theta4 - mặc định là False để tránh lag
SHOW_THETA4 = os.environ.get('SHOW_THETA4', 'true').lower() in ('true', '1', 'yes')
# Cấu hình hiển thị regions - mặc định là True 
SHOW_REGIONS = os.environ.get('SHOW_REGIONS', 'true').lower() in ('true', '1', 'yes')

def draw_rotated_boxes_with_depth(image, detections, depth_results=None, thickness=2):
    """
    Vẽ rotated bounding boxes với thông tin depth lên ảnh.
    
    Args:
        image: Ảnh để vẽ lên
        detections: Kết quả detection từ YOLO (chứa corners)
        depth_results: Kết quả depth estimation (optional)
        thickness: Độ dày đường viền
        
    Returns:
        np.ndarray: Ảnh đã được vẽ boxes
    """
    result_image = image.copy()
    
    # Màu sắc mặc định cho các boxes
    default_colors = [
        (0, 255, 0),    # Xanh lá
        (255, 0, 0),    # Đỏ
        (0, 0, 255),    # Xanh dương
        (255, 255, 0),  # Vàng
        (255, 0, 255),  # Tím
        (0, 255, 255),  # Cyan
    ]
    
    # Kiểm tra xem có corners không
    if 'corners' in detections and detections['corners']:
        # Sử dụng rotated bounding boxes (corners)
        corners_list = detections['corners']
        
        for i, corners in enumerate(corners_list):
            # Chọn màu
            color = default_colors[i % len(default_colors)]
            
            # Vẽ rotated box bằng cv2.polylines
            pts = np.array(corners, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result_image, [pts], True, color, thickness)
            
            # Thêm thông tin depth nếu có
            if depth_results and i < len(depth_results):
                depth_info = depth_results[i]
                mean_depth = depth_info.get('mean_depth', 0.0)
                
                # Tìm điểm trên cùng bên trái để đặt text
                corners_array = np.array(corners)
                min_y_idx = np.argmin(corners_array[:, 1])
                text_x, text_y = corners_array[min_y_idx]
                text_y = max(text_y - 5, 10)  # Đảm bảo không vẽ ra ngoài ảnh
                
                # Vẽ text depth
                cv2.putText(result_image, f"{mean_depth:.1f}m", 
                           (int(text_x), int(text_y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Vẽ background cho text để dễ đọc
                text_size = cv2.getTextSize(f"{mean_depth:.1f}m", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(result_image, 
                             (int(text_x) - 2, int(text_y) - text_size[1] - 2),
                             (int(text_x) + text_size[0] + 2, int(text_y) + 2),
                             (0, 0, 0), -1)
                cv2.putText(result_image, f"{mean_depth:.1f}m", 
                           (int(text_x), int(text_y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    elif 'bounding_boxes' in detections and detections['bounding_boxes']:
        # Fallback: sử dụng regular bounding boxes nếu không có corners
        print("[WARNING] Không có corners, sử dụng regular bounding boxes")
        bboxes = detections['bounding_boxes']
        
        for i, bbox in enumerate(bboxes):
            color = default_colors[i % len(default_colors)]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Vẽ regular box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # Thêm thông tin depth nếu có
            if depth_results and i < len(depth_results):
                depth_info = depth_results[i]
                mean_depth = depth_info.get('mean_depth', 0.0)
                cv2.putText(result_image, f"{mean_depth:.1f}m", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_image

def draw_depth_regions_with_rotated_boxes(image, depth_results):
    """
    Vẽ depth regions với rotated bounding boxes (cho pipeline camera).
    Phân biệt pallet regions và non-pallet objects.
    
    Args:
        image: Ảnh để vẽ lên
        depth_results: Kết quả depth từ pipeline
        
    Returns:
        np.ndarray: Ảnh đã được vẽ
    """
    result_image = image.copy()
    
    # Màu sắc cho các region
    pallet_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Xanh, đỏ, xanh dương cho pallet regions
    non_pallet_color = (255, 255, 0)  # Vàng cho non-pallet objects
    
    for i, region_data in enumerate(depth_results):
        # Lấy thông tin từ depth result
        region_info = region_data.get('region_info', {})
        position = region_data.get('position', {})
        
        # Phân biệt pallet và non-pallet
        pallet_id = region_info.get('pallet_id', 0)
        is_pallet = pallet_id > 0
        
        if is_pallet:
            # Pallet: Chọn màu dựa trên region_id
            region_id = region_info.get('region_id', 1)
            color = pallet_colors[(region_id - 1) % len(pallet_colors)]
            thickness = 2
        else:
            # Non-pallet: Sử dụng màu vàng, đường viền dày hơn
            color = non_pallet_color
            thickness = 3
        
        # Ưu tiên sử dụng corners nếu có (từ rotated boxes)
        if 'corners' in region_data and region_data['corners']:
            corners = region_data['corners']
            pts = np.array(corners, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result_image, [pts], True, color, thickness)
            
            # Tìm điểm để đặt text - sử dụng điểm có y nhỏ nhất
            corners_array = np.array(corners)
            min_y_idx = np.argmin(corners_array[:, 1])
            text_x, text_y = corners_array[min_y_idx]
            text_y = max(text_y - 5, 10)
            
        else:
            # Fallback: sử dụng bbox nếu không có corners
            bbox = region_data.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
                text_x, text_y = x1, y1 - 5
            else:
                print(f"[WARNING] Region không có corners hoặc bbox hợp lệ")
                continue
        
        # Hiển thị thông tin region và depth
        depth_z = position.get('z', 0.0)
        
        if is_pallet:
            # Pallet: hiển thị thông tin region
            region_id = region_info.get('region_id', 1)
            text = f"P{pallet_id}R{region_id}: {depth_z:.1f}m"
        else:
            # Non-pallet: hiển thị class
            object_class = region_info.get('object_class', 'Unknown')
            text = f"C{object_class}: {depth_z:.1f}m"
        
        # Vẽ background cho text để dễ đọc
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(result_image, 
                     (int(text_x) - 2, int(text_y) - text_size[1] - 2),
                     (int(text_x) + text_size[0] + 2, int(text_y) + 2),
                     (0, 0, 0), -1)
        
        cv2.putText(result_image, text, (int(text_x), int(text_y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_image

def draw_theta4_visualization(image, detections_with_theta4):
    """
    Vẽ theta4 visualization với regions, loads và rotation angles từ pipeline.
    
    Args:
        image: Ảnh gốc
        detections_with_theta4: Kết quả detection từ pipeline bao gồm theta4_result
        
    Returns:
        np.ndarray: Ảnh đã được vẽ với theta4 info
    """
    result_image = image.copy()
    
    # Lấy theta4 result từ detection
    theta4_result = detections_with_theta4.get('theta4_result')
    if theta4_result is None:
        # Nếu không có theta4 result, chỉ hiển thị thông báo
        cv2.putText(result_image, "No Theta4 calculation available", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return result_image
    
    # Màu sắc
    pallet_color = (255, 0, 0)      # Đỏ cho pallet
    load_color = (0, 255, 0)        # Xanh lá cho load
    region_color = (0, 255, 255)    # Vàng cho region boundaries
    mapping_color = (255, 0, 255)   # Magenta cho mapping lines
    
    # Vẽ regions từ module division
    regions_data = theta4_result['regions_result']['regions_data']
    for i, region in enumerate(regions_data):
        draw_region_boundary_theta4(result_image, region, region_color, i+1)
    
    # Vẽ load objects và theta4 mappings
    theta4_results = theta4_result['theta4_results']
    for i, theta4_calc in enumerate(theta4_results):
        mapping = theta4_calc['mapping']
        load_obj = mapping['load_object']
        
        # Vẽ load với orientation
        draw_object_with_orientation_theta4(result_image, load_obj, load_color, "LOAD")
        
        # Vẽ đường mapping và theta4 command nếu thành công
        if mapping['mapping_success'] and theta4_calc['theta4_success']:
            target_region = mapping['target_region']
            load_center = load_obj['center']
            region_center = target_region['center']
            
            # Vẽ đường mapping
            cv2.line(result_image, 
                    (int(load_center[0]), int(load_center[1])),
                    (int(region_center[0]), int(region_center[1])),
                    mapping_color, 2)
            
            # Vẽ theta4 command gần load
            theta4_command = theta4_calc.get('theta4_command', 'NO THETA4')
            rotation_direction = theta4_calc.get('rotation_direction', '')
            
            # Text theta4 command
            draw_text_with_background_theta4(result_image, theta4_command,
                                           (int(load_center[0]) + 15, int(load_center[1]) + 35),
                                           (255, 255, 255), (0, 0, 0))
            
            # Text hướng xoay (ngắn gọn)
            direction_short = "CW" if "Clockwise" in rotation_direction else ("CCW" if "Counter" in rotation_direction else "OK")
            draw_text_with_background_theta4(result_image, f"({direction_short})",
                                           (int(load_center[0]) + 15, int(load_center[1]) + 55),
                                           (255, 255, 0), (0, 0, 0))
    
    # Vẽ pallet objects
    for pallet in theta4_result['pallet_objects']:
        draw_object_with_orientation_theta4(result_image, pallet, pallet_color, "PALLET")
    
    # Vẽ summary info ở góc trên
    draw_summary_info_theta4(result_image, theta4_result)
    
    # Vẽ coordinate compass
    draw_coordinate_compass_theta4(result_image)
    
    return result_image

def draw_object_with_orientation_theta4(image, obj, color, obj_type):
    """Vẽ object với orientation arrow cho theta4 visualization"""
    import math
    
    cx, cy = obj['center']
    x_axis_angle = obj['x_axis_angle']
    length = obj['length']
    width = obj['width']
    
    # Vẽ center point
    cv2.circle(image, (int(cx), int(cy)), 4, color, -1)
    
    # Vẽ orientation arrow (theo hệ tọa độ custom)
    arrow_length = 30
    end_x = cx - arrow_length * math.cos(math.radians(x_axis_angle))
    end_y = cy - arrow_length * math.sin(math.radians(x_axis_angle))
    cv2.arrowedLine(image, (int(cx), int(cy)), 
                   (int(end_x), int(end_y)), color, 2)
    
    # Vẽ thông tin object
    text1 = f"{obj_type}: {length:.0f}x{width:.0f}"
    text2 = f"{x_axis_angle:.1f}°"
    draw_text_with_background_theta4(image, text1, 
                                   (int(cx) + 8, int(cy) - 25), color, (0, 0, 0))
    draw_text_with_background_theta4(image, text2, 
                                   (int(cx) + 8, int(cy) - 10), color, (0, 0, 0))

def draw_region_boundary_theta4(image, region, color, region_num):
    """Vẽ boundary của region cho theta4 visualization"""
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
    draw_text_with_background_theta4(image, f"R{region_num}", 
                                   (int(center[0]) - 10, int(center[1])), 
                                   color, (0, 0, 0))

def draw_summary_info_theta4(image, theta4_result):
    """Vẽ thông tin summary cho theta4 visualization"""
    summary = theta4_result['summary']
    layer = theta4_result['layer']
    
    y_offset = 20
    line_height = 18
    
    info_lines = [
        f"Theta4 Layer: {layer}",
        f"P:{summary['total_pallets']} L:{summary['total_loads']} R:{summary['total_regions']}",
        f"Map:{summary['successful_mappings']} θ4:{summary['successful_theta4']}",
        f"Time:{theta4_result['processing_time']*1000:.0f}ms"
    ]
    
    for i, line in enumerate(info_lines):
        draw_text_with_background_theta4(image, line,
                                       (10, y_offset + i * line_height),
                                       (255, 255, 255), (0, 0, 0))

def draw_text_with_background_theta4(image, text, position, color, bg_color=(0, 0, 0)):
    """Vẽ text với background cho theta4 visualization"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    cv2.rectangle(image, 
                 (position[0] - 2, position[1] - text_size[1] - 2),
                 (position[0] + text_size[0] + 2, position[1] + 2),
                 bg_color, -1)
    
    cv2.putText(image, text, position, font, font_scale, color, thickness)

def draw_coordinate_compass_theta4(image):
    """Vẽ compass hệ tọa độ custom cho theta4 visualization"""
    h, w = image.shape[:2]
    center_x = w - 60
    center_y = h - 60
    
    # Background
    cv2.rectangle(image, (center_x - 35, center_y - 35), 
                 (center_x + 35, center_y + 35), (0, 0, 0), -1)
    cv2.rectangle(image, (center_x - 35, center_y - 35), 
                 (center_x + 35, center_y + 35), (255, 255, 255), 1)
    
    # Trục X+ (đỏ, E→W)
    cv2.arrowedLine(image, (center_x, center_y), 
                   (center_x - 25, center_y), (0, 0, 255), 2)
    cv2.putText(image, "X+", (center_x - 35, center_y + 3), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # Trục Y+ (xanh lá, N→S)
    cv2.arrowedLine(image, (center_x, center_y), 
                   (center_x, center_y + 25), (0, 255, 0), 2)
    cv2.putText(image, "Y+", (center_x + 3, center_y + 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

def draw_region_visualization(image, detections_with_regions):
    """
    Vẽ regions và detections theo regions từ pipeline.
    
    Args:
        image: Ảnh gốc
        detections_with_regions: Kết quả detection từ pipeline bao gồm region_filtered
        
    Returns:
        np.ndarray: Ảnh đã được vẽ với regions và detections
    """
    result_image = image.copy()
    
    # Lấy region_filtered từ detections
    region_filtered = detections_with_regions.get('region_filtered')
    if not region_filtered:
        return result_image
    
    # Khởi tạo RegionManager để vẽ regions (chỉ để vẽ, không xử lý)
    temp_region_manager = RegionManager()
    
    # Vẽ tất cả regions trước
    result_image = temp_region_manager.draw_regions(result_image, show_labels=True)
    
    # Vẽ detections theo từng region
    for region_name, region_data in region_filtered['regions'].items():
        if not region_data['bounding_boxes']:
            continue
        
        region_info = region_data['region_info']
        color = region_info['color']
        bboxes = region_data['bounding_boxes']
        classes = region_data['classes']
        corners_list = region_data.get('corners', [])
        
        # Vẽ detections trong region này
        for i, bbox in enumerate(bboxes):
            # Ưu tiên vẽ corners nếu có
            if i < len(corners_list) and corners_list[i]:
                corners = corners_list[i]
                pts = np.array(corners, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(result_image, [pts], True, color, 3)
                
                # Vẽ điểm center
                center_x = int(np.mean([p[0] for p in corners]))
                center_y = int(np.mean([p[1] for p in corners]))
            else:
                # Fallback: vẽ regular bbox
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
            
            # Vẽ center point
            cv2.circle(result_image, (center_x, center_y), 5, color, -1)
            
            # Vẽ thông tin class
            class_id = classes[i] if i < len(classes) else 0
            class_names = {0: 'L', 1: 'L2', 2: 'P'}  # Tên ngắn
            class_name = class_names.get(class_id, str(int(class_id)))
            
            # Vẽ text với background
            text = f"{region_name}:{class_name}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_image,
                        (center_x - text_size[0]//2 - 2, center_y - 25 - text_size[1] - 2),
                        (center_x + text_size[0]//2 + 2, center_y - 25 + 2),
                        (0, 0, 0), -1)
            cv2.putText(result_image, text,
                      (center_x - text_size[0]//2, center_y - 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Vẽ unassigned detections nếu có
    unassigned_data = region_filtered.get('unassigned', {})
    if unassigned_data.get('bounding_boxes'):
        unassigned_color = (128, 128, 128)  # Xám cho unassigned
        bboxes = unassigned_data['bounding_boxes']
        classes = unassigned_data.get('classes', [])
        corners_list = unassigned_data.get('corners', [])
        
        for i, bbox in enumerate(bboxes):
            # Ưu tiên vẽ corners nếu có
            if i < len(corners_list) and corners_list[i]:
                corners = corners_list[i]
                pts = np.array(corners, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(result_image, [pts], True, unassigned_color, 2)
                
                center_x = int(np.mean([p[0] for p in corners]))
                center_y = int(np.mean([p[1] for p in corners]))
            else:
                # Fallback: vẽ regular bbox
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(result_image, (x1, y1), (x2, y2), unassigned_color, 2)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
            
            # Vẽ center point
            cv2.circle(result_image, (center_x, center_y), 3, unassigned_color, -1)
            
            # Vẽ thông tin class
            class_id = classes[i] if i < len(classes) else 0
            class_names = {0: 'L', 1: 'L2', 2: 'P'}
            class_name = class_names.get(class_id, str(int(class_id)))
            
            text = f"UNASSIGNED:{class_name}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(result_image,
                        (center_x - text_size[0]//2 - 2, center_y - 20 - text_size[1] - 2),
                        (center_x + text_size[0]//2 + 2, center_y - 20 + 2),
                        (0, 0, 0), -1)
            cv2.putText(result_image, text,
                      (center_x - text_size[0]//2, center_y - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Vẽ thống kê ở góc trên bên trái
    y_offset = 10
    line_height = 20
    
    region_counts = {}
    for region_name, region_data in region_filtered['regions'].items():
        count = len(region_data['bounding_boxes'])
        if count > 0:
            region_counts[region_name] = count
    
    unassigned_count = len(region_filtered['unassigned']['bounding_boxes'])
    
    stats_lines = [
        "=== REGION STATISTICS ===",
        f"Total Detections: {len(detections_with_regions.get('bounding_boxes', []))}",
    ]
    
    for region_name, count in region_counts.items():
        stats_lines.append(f"{region_name}: {count}")
    
    if unassigned_count > 0:
        stats_lines.append(f"Unassigned: {unassigned_count}")
    
    for i, line in enumerate(stats_lines):
        # Vẽ background cho text
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(result_image,
                    (10 - 2, y_offset + i * line_height - text_size[1] - 2),
                    (10 + text_size[0] + 2, y_offset + i * line_height + 2),
                    (0, 0, 0), -1)
        
        # Chọn màu text
        if i == 0:  # Header
            color = (255, 255, 0)  # Vàng
        elif "Unassigned" in line:
            color = (128, 128, 128)  # Xám
        else:
            color = (255, 255, 255)  # Trắng
        
        cv2.putText(result_image, line,
                  (10, y_offset + i * line_height),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return result_image

def demo_single_image():
    """Thử nghiệm với một ảnh đơn lẻ"""
    print("Thử nghiệm phát hiện đối tượng với TensorRT trên một ảnh đơn lẻ")
    
    # Khởi tạo model YOLO
    model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)
    
    # Khởi tạo model Depth (sử dụng chung config với camera)
    depth_model = create_depth()
    print(f"Depth model enable: {depth_model.enable}")
    
    # Khởi tạo Module Division
    divider = ModuleDivision()
    print("Module Division đã được khởi tạo")
    
    # Hiển thị ảnh có sẵn từ folder images_pallets2
    print("\nẢnh có sẵn trong folder images_pallets2:")
    pallets_folder = "images_pallets2"
    if os.path.exists(pallets_folder):
        image_files = [f for f in os.listdir(pallets_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()  # Sắp xếp theo thứ tự
        for i, img_file in enumerate(image_files, 1):
            print(f"  {i}. {img_file}")
        print(f"  0. Nhập đường dẫn khác")
        
        choice = input(f"\nChọn ảnh (1-{len(image_files)}) hoặc 0 để nhập đường dẫn khác: ")
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(image_files):
                image_path = os.path.join(pallets_folder, image_files[choice_num - 1])
                print(f"Đã chọn: {image_path}")
            elif choice_num == 0:
                image_path = input("Nhập đường dẫn tới ảnh thử nghiệm: ")
                if not image_path:
                    image_path = "test.jpg"  # Ảnh mặc định
            else:
                print("Lựa chọn không hợp lệ, sử dụng ảnh đầu tiên")
                image_path = os.path.join(pallets_folder, image_files[0])
        except ValueError:
            print("Lựa chọn không hợp lệ, sử dụng ảnh đầu tiên")
            image_path = os.path.join(pallets_folder, image_files[0])
    else:
        # Đọc ảnh thử nghiệm theo cách cũ nếu không tìm thấy folder
        image_path = input("Nhập đường dẫn tới ảnh thử nghiệm: ")
        if not image_path:
            image_path = "test.jpg"  # Ảnh mặc định
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return
    
    # Hiển thị thông tin ảnh
    height, width = frame.shape[:2]
    print(f"\nThông tin ảnh:")
    print(f"  Đường dẫn: {image_path}")
    print(f"  Kích thước: {width}x{height}")
    print(f"  Kích thước file: {os.path.getsize(image_path)} bytes")
    
    # Đo thời gian xử lý YOLO
    start_time = time.time()
    
    # Thực hiện phát hiện YOLO
    detections = model.detect(frame)
    
    yolo_time = time.time()
    
    # Chia pallet thành các vùng nhỏ và thực hiện depth estimation
    depth_results = None
    region_depth_results = []
    if depth_model.enable and len(detections['bounding_boxes']) > 0:
        print("Đang chia pallet thành các vùng nhỏ...")
        
        # Chia pallet thành các vùng sử dụng Module Division
        divided_result = divider.process_pallet_detections(detections, layer=1)
        depth_regions = divider.prepare_for_depth_estimation(divided_result)
        
        print(f"Đã chia thành {len(depth_regions)} vùng")
        
        print("Đang xử lý depth estimation cho từng vùng...")
        # Thực hiện depth estimation cho từng vùng
        for i, region in enumerate(depth_regions):
            bbox = region['bbox']
            region_info = region['region_info']
            
            # Ước tính độ sâu cho bbox này
            region_depth = depth_model.estimate_depth(frame, [bbox])
            
            # Tạo kết quả chi tiết cho region
            if region_depth and len(region_depth) > 0:
                depth_info = region_depth[0]  # Lấy kết quả đầu tiên
                result = {
                    'region_info': region_info,
                    'bbox': bbox,
                    'center': region['center'],
                    'depth': depth_info,
                    'position': {
                        'x': region['center'][0],
                        'y': region['center'][1], 
                        'z': depth_info.get('mean_depth', 0.0) if isinstance(depth_info, dict) else 0.0
                    }
                }
                
                # Thêm corners nếu có (để vẽ rotated boxes)
                if 'corners' in region:
                    result['corners'] = region['corners']
                
                # Thêm corners gốc của pallet nếu có
                if 'original_corners' in region:
                    result['original_corners'] = region['original_corners']
                
                region_depth_results.append(result)
        
        # Giữ lại depth_results cũ để tương thích với code hiển thị
        if region_depth_results:
            depth_results = [r['depth'] for r in region_depth_results]
        
    depth_time = time.time()
    
    # Hiển thị kết quả
    print(f"Thời gian xử lý YOLO: {(yolo_time - start_time) * 1000:.2f} ms")
    if depth_model.enable:
        print(f"Thời gian xử lý Depth: {(depth_time - yolo_time) * 1000:.2f} ms")
        print(f"Tổng thời gian: {(depth_time - start_time) * 1000:.2f} ms")
    print(f"Đã phát hiện {len(detections['bounding_boxes'])} đối tượng")
    
    # Hiển thị thông tin depth nếu có
    if region_depth_results and len(region_depth_results) > 0:
        print("Thông tin độ sâu theo vùng:")
        for i, result in enumerate(region_depth_results):
            region_info = result['region_info']
            depth_info = result['depth']
            position = result['position']
            
            pallet_id = region_info.get('pallet_id', 1)
            region_id = region_info.get('region_id', 1)
            layer = region_info.get('layer', 1)
            
            if isinstance(depth_info, dict):
                mean_depth = depth_info.get('mean_depth', 0.0)
                min_depth = depth_info.get('min_depth', 0.0) 
                max_depth = depth_info.get('max_depth', 0.0)
            else:
                mean_depth = min_depth = max_depth = 0.0
            
            print(f"  Pallet {pallet_id}, Vùng {region_id} (Layer {layer}): {mean_depth:.2f}m (min: {min_depth:.2f}m, max: {max_depth:.2f}m)")
            print(f"    Tọa độ pixel: X={position['x']:.1f}, Y={position['y']:.1f}, Z={position['z']:.2f}m")
            
            # Hiển thị thông tin 3D nếu có camera calibration
            if 'position_3d_camera' in result:
                pos_3d = result['position_3d_camera']
                print(f"    Tọa độ 3D (camera): X={pos_3d['X']:.3f}m, Y={pos_3d['Y']:.3f}m, Z={pos_3d['Z']:.3f}m")
            
            if 'real_size' in result:
                real_size = result['real_size']
                print(f"    Kích thước thực: {real_size['width_m']:.3f}m x {real_size['height_m']:.3f}m (diện tích: {real_size['area_m2']:.3f}m²)")
    elif depth_results and len(depth_results) > 0:
        # Fallback cho trường hợp không có region results
        print("Thông tin độ sâu (không chia vùng):")
        for i, result in enumerate(depth_results):
            if isinstance(result, dict):
                print(f"  Đối tượng {i+1}: {result.get('mean_depth', 0.0):.2f}m (min: {result.get('min_depth', 0.0):.2f}m, max: {result.get('max_depth', 0.0):.2f}m)")
            else:
                print(f"  Đối tượng {i+1}: Không có thông tin depth")
    
    # Hiển thị ảnh detection từ YOLO
    cv2.imshow("Kết quả phát hiện", detections["annotated_frame"])
    
    # Hiển thị rotated boxes với depth information
    if detections['corners'] or detections['bounding_boxes']:
        depth_viz = draw_rotated_boxes_with_depth(frame, detections, depth_results)
        cv2.imshow("Rotated Boxes với Depth Information", depth_viz)
        
        # Hiển thị depth regions nếu có chia vùng
        if region_depth_results:
            region_viz = draw_depth_regions_with_rotated_boxes(frame, region_depth_results)
            cv2.imshow("Depth Regions (Module Division)", region_viz)
    
    print("\nẢnh đã được hiển thị, vui lòng nhấn phím bất kỳ để tiếp tục")
    cv2.waitKey(0)
    # Lưu kết quả
    save_choice = input("\nBạn có muốn lưu kết quả? (y/n): ").lower()
    if save_choice in ['y', 'yes']:
        # Tạo tên file output
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        detection_output_path = f"result_{base_name}.jpg"
        cv2.imwrite(detection_output_path, detections["annotated_frame"])
        print(f"Đã lưu kết quả detection tại: {detection_output_path}")
        
        # Lưu rotated boxes với depth nếu có
        if detections['corners'] or detections['bounding_boxes']:
            depth_viz = draw_rotated_boxes_with_depth(frame, detections, depth_results)
            depth_output_path = f"rotated_depth_{base_name}.jpg"
            cv2.imwrite(depth_output_path, depth_viz)
            print(f"Đã lưu kết quả rotated boxes với depth tại: {depth_output_path}")
            
            # Lưu depth regions nếu có chia vùng
            if region_depth_results:
                region_viz = draw_depth_regions_with_rotated_boxes(frame, region_depth_results)
                region_output_path = f"depth_regions_{base_name}.jpg"
                cv2.imwrite(region_output_path, region_viz)
                print(f"Đã lưu kết quả depth regions tại: {region_output_path}")
    
    cv2.destroyAllWindows()

def demo_batch_images():
    """Thử nghiệm với tất cả ảnh trong folder images_pallets2"""
    print("Thử nghiệm phát hiện đối tượng với TensorRT trên tất cả ảnh trong folder")
    
    pallets_folder = "images_pallets"
    if not os.path.exists(pallets_folder):
        print(f"Không tìm thấy folder {pallets_folder}")
        return
    
    # Khởi tạo model YOLO
    model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)
    
    # Khởi tạo model Depth (sử dụng chung config với camera)
    depth_model = create_depth()
    print(f"Depth model enable: {depth_model.enable}")
    
    # Khởi tạo Module Division
    divider = ModuleDivision()
    print("Module Division đã được khởi tạo")
    
    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(pallets_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    if not image_files:
        print("Không có ảnh nào trong folder")
        return
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    
    # Tạo folder kết quả
    output_folder = "batch_results"
    os.makedirs(output_folder, exist_ok=True)
    
    # Tạo subfolder cho rotated boxes với depth
    rotated_folder = os.path.join(output_folder, "rotated_depth")
    os.makedirs(rotated_folder, exist_ok=True)
    
    # Tạo subfolder cho depth regions 
    regions_folder = os.path.join(output_folder, "depth_regions")
    os.makedirs(regions_folder, exist_ok=True)
    
    total_time = 0
    total_yolo_time = 0
    total_depth_time = 0
    successful_detections = 0
    successful_depth_detections = 0
    
    for i, img_file in enumerate(image_files, 1):
        image_path = os.path.join(pallets_folder, img_file)
        print(f"\n[{i}/{len(image_files)}] Xử lý: {img_file}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  Không thể đọc ảnh: {img_file}")
            continue
        
        # Đo thời gian xử lý YOLO
        start_time = time.time()
        
        # Thực hiện phát hiện YOLO
        detections = model.detect(frame)
        
        yolo_time = time.time()
        yolo_process_time = (yolo_time - start_time) * 1000
        total_yolo_time += yolo_process_time
        
        # Chia pallet thành các vùng nhỏ và thực hiện depth estimation
        depth_results = None
        region_depth_results = []
        depth_process_time = 0
        if depth_model.enable and len(detections['bounding_boxes']) > 0:
            # Chia pallet thành các vùng sử dụng Module Division
            divided_result = divider.process_pallet_detections(detections, layer=2)
            depth_regions = divider.prepare_for_depth_estimation(divided_result)
            
            # Thực hiện depth estimation cho từng vùng
            for region in depth_regions:
                bbox = region['bbox']
                region_info = region['region_info']
                
                # Ước tính độ sâu cho bbox này
                region_depth = depth_model.estimate_depth(frame, [bbox])
                
                # Tạo kết quả chi tiết cho region
                if region_depth and len(region_depth) > 0:
                    depth_info = region_depth[0]  # Lấy kết quả đầu tiên
                    result = {
                        'region_info': region_info,
                        'bbox': bbox,
                        'center': region['center'],
                        'depth': depth_info,
                        'position': {
                            'x': region['center'][0],
                            'y': region['center'][1], 
                            'z': depth_info.get('mean_depth', 0.0) if isinstance(depth_info, dict) else 0.0
                        }
                    }
                    
                    # Thêm corners nếu có (để vẽ rotated boxes)
                    if 'corners' in region:
                        result['corners'] = region['corners']
                    
                    # Thêm corners gốc của pallet nếu có
                    if 'original_corners' in region:
                        result['original_corners'] = region['original_corners']
                    
                    region_depth_results.append(result)
            
            # Giữ lại depth_results cũ để tương thích với code hiển thị
            if region_depth_results:
                depth_results = [r['depth'] for r in region_depth_results]
            
            depth_end_time = time.time()
            depth_process_time = (depth_end_time - yolo_time) * 1000
            total_depth_time += depth_process_time
        
        total_process_time = yolo_process_time + depth_process_time
        total_time += total_process_time
        
        # Hiển thị kết quả
        num_objects = len(detections['bounding_boxes'])
        print(f"  Thời gian YOLO: {yolo_process_time:.2f} ms")
        if depth_model.enable:
            print(f"  Thời gian Depth: {depth_process_time:.2f} ms")
            print(f"  Tổng thời gian: {total_process_time:.2f} ms")
        print(f"  Đã phát hiện: {num_objects} đối tượng")
        
        if num_objects > 0:
            successful_detections += 1
        
        # Hiển thị thông tin depth nếu có
        if region_depth_results and len(region_depth_results) > 0:
            successful_depth_detections += 1
            print(f"  Thông tin độ sâu theo vùng ({len(region_depth_results)} vùng):")
            for j, result in enumerate(region_depth_results):
                region_info = result['region_info']
                depth_info = result['depth']
                position = result['position']
                
                pallet_id = region_info.get('pallet_id', 1)
                region_id = region_info.get('region_id', 1)
                layer = region_info.get('layer', 1)
                
                if isinstance(depth_info, dict):
                    mean_depth = depth_info.get('mean_depth', 0.0)
                else:
                    mean_depth = 0.0
                
                print(f"    P{pallet_id}R{region_id}L{layer}: {mean_depth:.2f}m")
        elif depth_results and len(depth_results) > 0:
            # Fallback cho trường hợp không có region results
            successful_depth_detections += 1
            print(f"  Thông tin độ sâu (không chia vùng):")
            for j, result in enumerate(depth_results):
                if isinstance(result, dict):
                    print(f"    Đối tượng {j+1}: {result.get('mean_depth', 0.0):.2f}m")
                else:
                    print(f"    Đối tượng {j+1}: Không có thông tin depth")
        
        # Lưu kết quả detection
        base_name = os.path.splitext(img_file)[0]
        detection_output_path = os.path.join(output_folder, f"result_{base_name}.jpg")
        cv2.imwrite(detection_output_path, detections["annotated_frame"])
        print(f"  Đã lưu detection: {detection_output_path}")
        
        # Lưu kết quả rotated boxes với depth
        if detections['corners'] or detections['bounding_boxes']:
            depth_viz = draw_rotated_boxes_with_depth(frame, detections, depth_results)
            rotated_output_path = os.path.join(rotated_folder, f"rotated_depth_{base_name}.jpg")
            cv2.imwrite(rotated_output_path, depth_viz)
            print(f"  Đã lưu rotated boxes với depth: {rotated_output_path}")
            
            # Lưu depth regions nếu có chia vùng
            if region_depth_results:
                region_viz = draw_depth_regions_with_rotated_boxes(frame, region_depth_results)
                region_output_path = os.path.join(regions_folder, f"depth_regions_{base_name}.jpg")
                cv2.imwrite(region_output_path, region_viz)
                print(f"  Đã lưu depth regions: {region_output_path}")
    
    # Thống kê tổng kết
    print(f"\n=== THỐNG KÊ TỔNG KẾT ===")
    print(f"Tổng số ảnh xử lý: {len(image_files)}")
    print(f"Ảnh có phát hiện đối tượng: {successful_detections}")
    print(f"Tỉ lệ phát hiện thành công: {successful_detections/len(image_files)*100:.1f}%")
    
    if depth_model.enable:
        print(f"Ảnh có thông tin depth: {successful_depth_detections}")
        print(f"Tỉ lệ depth thành công: {successful_depth_detections/len(image_files)*100:.1f}%")
        print(f"Thời gian YOLO trung bình: {total_yolo_time/len(image_files):.2f} ms/ảnh")
        print(f"Thời gian Depth trung bình: {total_depth_time/len(image_files):.2f} ms/ảnh")
    
    print(f"Thời gian tổng trung bình: {total_time/len(image_files):.2f} ms/ảnh")
    print(f"Kết quả detection đã được lưu trong folder: {output_folder}")
    print(f"Kết quả rotated boxes với depth đã được lưu trong folder: {rotated_folder}")
    print(f"Kết quả depth regions (Module Division) đã được lưu trong folder: {regions_folder}")

# Di chuyển các hàm factory ra ngoài hàm demo_camera để có thể pickle
def create_camera():
    camera = CameraInterface(camera_index=0)
    camera.initialize()
    return camera

def create_yolo():
    return YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.55)

def create_depth():
    # Cho phép chạy depth model trên CPU hoặc tắt hoàn toàn
    use_device = os.environ.get('DEPTH_DEVICE', 'cuda')  # 'cuda', 'cpu' hoặc 'off' 
    enable_depth = use_device.lower() != 'off'
    
    # Lấy loại model: regular hoặc metric
    model_type = os.environ.get('DEPTH_TYPE', 'metric').lower()  # 'regular' hoặc 'metric'
    
    # Lấy kích thước model
    model_size = os.environ.get('DEPTH_MODEL', 'small').lower()  # 'large', 'base', 'small'
    
    # Lấy loại scene cho metric depth
    scene_type = os.environ.get('DEPTH_SCENE', 'indoor').lower()  # 'indoor' hoặc 'outdoor'
    
    # Kích thước input
    input_size_str = os.environ.get('DEPTH_SIZE', '640x640')
    input_size = None
    if input_size_str:
        try:
            w, h = map(int, input_size_str.split('x'))
            input_size = (w, h)
        except:
            print(f"[Factory] Không thể phân tích DEPTH_SIZE: {input_size_str}, sử dụng kích thước gốc")
    
    # Bỏ qua frame
    skip_frames_str = os.environ.get('DEPTH_SKIP', '50')
    try:
        skip_frames = int(skip_frames_str)
    except:
        skip_frames = 0
    
    # Camera calibration settings
    use_calibration = os.environ.get('USE_CAMERA_CALIBRATION', 'True').lower() in ('true', '1', 'yes')
    calibration_file = os.environ.get('CAMERA_CALIBRATION_FILE', 'camera_params.npz')
    
    if use_device.lower() == 'off':
        # print(f"[Factory] Đã tắt depth model để tiết kiệm tài nguyên")
        return DepthEstimator(device='cpu', enable=False, use_camera_calibration=use_calibration, camera_calibration_file=calibration_file)
    
    # print(f"[Factory] Khởi tạo depth model trên thiết bị: {use_device}")
    # print(f"[Factory] Model type: {model_type}, Size: {model_size}")
    # if model_type == 'metric':
    #     print(f"[Factory] Scene type: {scene_type}")
    # print(f"[Factory] Camera calibration: {'Bật' if use_calibration else 'Tắt'}")
    # if use_calibration:
    #     print(f"[Factory] Calibration file: {calibration_file}")
    
    # Tạo DepthEstimator dựa trên loại model
    if model_type == 'metric':
        return DepthEstimator.create_metric(
            scene_type=scene_type,
            model_size=model_size,
            device=use_device, 
            enable=enable_depth,
            input_size=input_size,
            skip_frames=skip_frames,
            use_camera_calibration=use_calibration,
            camera_calibration_file=calibration_file
        )
    else:
        return DepthEstimator.create_regular(
            model_size=model_size,
            device=use_device, 
            enable=enable_depth,
            input_size=input_size,
            skip_frames=skip_frames,
            use_camera_calibration=use_calibration,
            camera_calibration_file=calibration_file
        )

def demo_camera():
    """Thử nghiệm với camera thời gian thực"""
    # print("Thử nghiệm phát hiện đối tượng với TensorRT trên camera thời gian thực")
    global SHOW_DEPTH, SHOW_THETA4, SHOW_REGIONS
    
    # ⭐ PLC INTEGRATION SETUP ⭐
    enable_plc = os.environ.get('ENABLE_PLC', 'true').lower() in ('true', '1', 'yes')
    plc_ip = os.environ.get('PLC_IP', '192.168.0.1')
    
    if enable_plc:
        print(f"🏭 PLC Integration: ENABLED (IP: {plc_ip})")
        print("   💡 Nhấn 'n' để send regions vào PLC thật sự!")
    else:
        print(f"🏭 PLC Integration: DISABLED")
        print("   💡 Để enable: set ENABLE_PLC=true")
        print("   💡 Set IP: set PLC_IP=192.168.1.100")
    
    # ⭐ GIẢI THÍCH CÁC TABS VISUALIZATION ⭐
    print("🎯 HƯỚNG DẪN CÁC TABS VISUALIZATION:")
    print("1. 📺 TAB CHÍNH: 'Phát hiện đối tượng với TensorRT'")
    print("   - Hiển thị: YOLO detection bounding boxes")
    print("   - Thông tin: FPS, số objects, theta4 success, robot coordinates")
    print("   - Đây là tab luôn hiển thị")
    print()
    print("2. 🧠 TAB THETA4: 'Theta4 Calculation & Regions' (Nhấn 't' để bật/tắt)")
    print("   - Hiển thị: Theta4 rotation calculations")
    print("   - Bao gồm: Load objects, pallet regions, rotation angles, mapping lines")
    print("   - Có compass tọa độ và theta4 commands")
    print()
    print("3. 🗺️ TAB REGIONS: 'Region Processing & Detections' (Nhấn 'r' để bật/tắt)")
    print("   - Hiển thị: Region boundaries (loads, pallets1, pallets2)")
    print("   - Bao gồm: Detections được filter theo regions, unassigned objects")
    print("   - Có statistics về số objects trong mỗi region")
    print()
    print("4. 📏 TAB DEPTH: 'Rotated Depth Regions' (Nhấn 'd' để bật/tắt)")
    print("   - Hiển thị: Depth estimation với rotated bounding boxes")
    print("   - Bao gồm: Pallet regions (P1R1, P1R2, P1R3) với depth values")
    print("   - Sequential regions được hiển thị ở đây!")
    print()
    print("🔑 QUAN TRỌNG: SEQUENTIAL REGIONS hiển thị chủ yếu ở:")
    print("   📏 TAB DEPTH - Nơi bạn sẽ thấy P1R1, P1R2, P1R3 với depth values")
    print("   🗺️ TAB REGIONS - Nơi bạn thấy region boundaries và filtering")
    print()
    print("⌨️ KEYBOARD CONTROLS:")
    print("   'q': Thoát  |  'd': Depth tab  |  't': Theta4 tab  |  'r': Regions tab")
    print("   'h': Help (hướng dẫn chi tiết)  |  's': Sequence status")
    print()
    
    # Hiển thị tùy chọn depth, theta4 và regions
    # print(f"Hiển thị depth map: {'BẬT' if SHOW_DEPTH else 'TẮT'} (Dùng 'd' để bật/tắt)")
    # print(f"Hiển thị theta4 info: {'BẬT' if SHOW_THETA4 else 'TẮT'} (Dùng 't' để bật/tắt)")
    # print(f"Hiển thị regions: {'BẬT' if SHOW_REGIONS else 'TẮT'} (Dùng 'r' để bật/tắt)")
    
    # Khởi tạo pipeline với các factory function đã được định nghĩa ở cấp module
    pipeline = ProcessingPipeline(
        camera_factory=create_camera,
        yolo_factory=create_yolo,
        depth_factory=create_depth,
        enable_plc=enable_plc,
        plc_ip=plc_ip
    )
    
    # Biến để lưu frame depth, theta4 và regions cuối cùng
    last_depth_viz = None
    last_depth_time = 0
    last_theta4_viz = None
    last_theta4_time = 0
    last_regions_viz = None
    last_regions_time = 0
    skip_counter = 0
    max_skip = 5  # Bỏ qua tối đa 5 frames khi xử lý không kịp
    
    # Khởi động pipeline
    if pipeline.start(timeout=60.0):
        print("Pipeline đã khởi động thành công!")
        
        # ⭐ CONNECT PLC IF ENABLED ⭐
        if enable_plc:
            print(f"🔌 Đang kết nối PLC...")
            plc_connected = pipeline.connect_plc()
            if plc_connected:
                print(f"✅ Kết nối PLC thành công!")
            else:
                print(f"❌ Kết nối PLC thất bại! Tiếp tục mà không PLC...")
        
        # print("\nPhím điều khiển:")
        # print("  'q': Thoát")
        # print("  'd': Bật/tắt hiển thị depth map")
        # print("  't': Bật/tắt hiển thị theta4 calculation")
        # print("  'r': Bật/tắt hiển thị regions processing")
        pass
        
        try:
            # Vòng lặp hiển thị kết quả
            fps_counter = 0
            fps_time = time.time()
            fps = 0.0  # Khởi tạo FPS
            
            while True:
                start_loop = time.time()
                
                # Lấy kết quả detection mới nhất
                detection_result = pipeline.get_latest_detection()
                if not detection_result:
                    # Nếu không có kết quả detection, chờ một chút
                    time.sleep(0.01)
                    continue
                
                frame, detections = detection_result
                
                # Nếu xử lý quá chậm, tăng skip_counter
                if time.time() - start_loop > 0.1:  # Quá 100ms
                    skip_counter += 1
                    if skip_counter >= max_skip:
                        # Bỏ qua hiển thị để giảm tải
                        skip_counter = 0
                        continue
                else:
                    skip_counter = 0  # Reset nếu xử lý nhanh
                
                # Tính FPS
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()
                    
                    # Cập nhật thông tin thống kê
                    stats = pipeline.get_stats()
                # Vẽ FPS lên frame detection
                display_frame = detections["annotated_frame"].copy()
                
                # Vẽ FPS với background đen để dễ đọc
                fps_text = f"FPS: {fps:.1f}"
                text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.rectangle(display_frame, 
                             (10 - 5, 30 - text_size[1] - 5),
                             (10 + text_size[0] + 5, 30 + 5),
                             (0, 0, 0), -1)
                cv2.putText(display_frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Thêm thông tin số objects
                num_objects = len(detections.get('bounding_boxes', []))
                objects_text = f"Objects: {num_objects}"
                cv2.putText(display_frame, objects_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Thêm thông tin theta4 nếu có
                theta4_result = detections.get('theta4_result')
                if theta4_result:
                    theta4_success = theta4_result['summary']['successful_theta4']
                    total_loads = theta4_result['summary']['total_loads']
                    theta4_text = f"Theta4: {theta4_success}/{total_loads}"
                    cv2.putText(display_frame, theta4_text, (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # ⭐ HIỂN THỊ ROBOT COORDINATES ⭐
                robot_coords = detections.get('robot_coordinates', [])
                if robot_coords:
                    coords_text = f"Robot Coords: {len(robot_coords)}"
                    cv2.putText(display_frame, coords_text, (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # ⭐ DEBUG: KIỂM TRA PALLET REGIONS DATA ⭐
                pallet_regions = detections.get('pallet_regions', [])
                if pallet_regions:
                    regions_text = f"Pallet Regions: {len(pallet_regions)}"
                    cv2.putText(display_frame, regions_text, (10, 190), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    
                    # Debug log (mỗi 10 frames để sync với pipeline)
                    if fps_counter % 10 == 0:
                        print(f"[DEBUG] Received pallet_regions: {len(pallet_regions)}")
                        for i, region in enumerate(pallet_regions):
                            region_info = region.get('region_info', {})
                            bbox = region.get('bbox', [])
                            corners = region.get('corners', [])
                            print(f"  Region {i}: P{region_info.get('pallet_id')}R{region_info.get('region_id')} bbox={[int(x) for x in bbox]} corners={len(corners) > 0}")
                else:
                    # Hiển thị thông báo không có regions
                    if fps_counter % 10 == 0:
                        print(f"[DEBUG] No pallet_regions data received")
                    
                    # In ra console cho từng object - LOG ROBOT COORDINATES (mỗi 10 frames)
                    if len(robot_coords) > 0 and fps_counter % 10 == 0:
                        print(f"[ROBOT COORDS] Frame {fps_counter}: {len(robot_coords)} objects")
                        for coord in robot_coords:
                            class_name = coord['class']
                            pixel = coord['camera_pixel']
                            robot_pos = coord['robot_coordinates']
                            cam_3d = coord.get('camera_3d')
                            
                            print(f"   {class_name}: Pixel({pixel['x']},{pixel['y']}) → Robot(X={robot_pos['x']:.2f}, Y={robot_pos['y']:.2f})")
                            if cam_3d:
                                print(f"      (Camera3D: X={cam_3d['X']:.3f}, Y={cam_3d['Y']:.3f}, Z={cam_3d['Z']:.3f})")
                
                # Hiển thị kết quả detection với FPS
                cv2.imshow("Phát hiện đối tượng với TensorRT", display_frame)
                
                # Xử lý depth chỉ khi SHOW_DEPTH được bật
                if SHOW_DEPTH:
                    # Chỉ lấy depth mới sau mỗi 0.5 giây
                    if time.time() - last_depth_time > 0.5:
                        depth_result = pipeline.get_latest_depth()
                        if depth_result:
                            frame_depth, depth_results = depth_result
                            
                            # Sử dụng helper function để vẽ rotated boxes với depth
                            depth_viz = draw_depth_regions_with_rotated_boxes(frame_depth, depth_results)
                            
                            # Lưu lại để tái sử dụng
                            last_depth_viz = depth_viz
                            last_depth_time = time.time()
                    
                    # Hiển thị depth từ lần xử lý gần nhất
                    if last_depth_viz is not None:
                        cv2.imshow("Rotated Depth Regions", last_depth_viz)
                
                # ⭐ XỬ LÝ THETA4 CHỈ KHI SHOW_THETA4 ĐƯỢC BẬT ⭐
                if SHOW_THETA4:
                    # Chỉ cập nhật theta4 visualization sau mỗi 0.3 giây để tránh lag
                    if time.time() - last_theta4_time > 0.3:
                        theta4_result = detections.get('theta4_result')
                        if theta4_result:
                            # Sử dụng function theta4 visualization
                            theta4_viz = draw_theta4_visualization(frame, detections)
                            
                            # Lưu lại để tái sử dụng
                            last_theta4_viz = theta4_viz
                            last_theta4_time = time.time()
                    
                    # Hiển thị theta4 từ lần xử lý gần nhất
                    if last_theta4_viz is not None:
                        cv2.imshow("Theta4 Calculation & Regions", last_theta4_viz)
                
                # ⭐ XỬ LÝ REGIONS CHỈ KHI SHOW_REGIONS ĐƯỢC BẬT ⭐
                if SHOW_REGIONS:
                    # Chỉ cập nhật regions visualization sau mỗi 0.2 giây để tránh lag
                    if time.time() - last_regions_time > 0.2:
                        region_filtered = detections.get('region_filtered')
                        if region_filtered:
                            # Sử dụng function regions visualization
                            regions_viz = draw_region_visualization(frame, detections)
                            
                            # Lưu lại để tái sử dụng
                            last_regions_viz = regions_viz
                            last_regions_time = time.time()
                    
                    # Hiển thị regions từ lần xử lý gần nhất
                    if last_regions_viz is not None:
                        cv2.imshow("Region Processing & Detections", last_regions_viz)
                
                # Xử lý phím nhấn
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    # Bật/tắt hiển thị depth
                    SHOW_DEPTH = not SHOW_DEPTH
                    # print(f"Hiển thị depth map: {'BẬT' if SHOW_DEPTH else 'TẮT'}")
                    if not SHOW_DEPTH:
                        cv2.destroyWindow("Rotated Depth Regions")
                elif key == ord('t'):
                    # Bật/tắt hiển thị theta4
                    SHOW_THETA4 = not SHOW_THETA4
                    # print(f"Hiển thị theta4 calculation: {'BẬT' if SHOW_THETA4 else 'TẮT'}")
                    if not SHOW_THETA4:
                        cv2.destroyWindow("Theta4 Calculation & Regions")
                elif key == ord('h'):
                    # Show help
                    print("\n📚 [HELP] HƯỚNG DẪN SỬ DỤNG:")
                    print("=== KEYBOARD CONTROLS ===")
                    print("'q': Thoát")
                    print("'d': Bật/tắt TAB DEPTH (📏 'Rotated Depth Regions')")
                    print("'t': Bật/tắt TAB THETA4 (🧠 'Theta4 Calculation & Regions')")
                    print("'r': Bật/tắt TAB REGIONS (🗺️ 'Region Processing & Detections')")
                    print("'h': Hiển thị help này")
                    print()
                    print("=== SEQUENTIAL CONTROLS ===")
                    if enable_plc:
                        print("'n': ⭐ SEND REGIONS TO PLC ⭐ (Real PLC sending)")
                    else:
                        print("'n': Next region (demo only - enable PLC to send real data)")
                    print("'c': Complete region (demo only)")
                    print("'s': Show sequence status")
                    print("'x': Reset sequence (demo only)")
                    print("'z': Show depth info")
                    print()
                    print("=== PLC INTEGRATION ===")
                    print(f"PLC Status: {'🟢 ENABLED' if enable_plc else '🔴 DISABLED'}")
                    if enable_plc:
                        plc_integration = pipeline.get_plc_integration()
                        if plc_integration and plc_integration.plc_connected:
                            print(f"PLC Connection: ✅ CONNECTED (IP: {plc_ip})")
                        else:
                            print(f"PLC Connection: ❌ DISCONNECTED (IP: {plc_ip})")
                    else:
                        print("To enable: set ENABLE_PLC=true")
                    print()
                    print("=== TABS VISUALIZATION ===")
                    print("📺 TAB CHÍNH: Luôn hiển thị YOLO detections")
                    print("📏 TAB DEPTH: Sequential regions P1R1, P1R2, P1R3")
                    print("🗺️ TAB REGIONS: Region boundaries & filtering")
                    print("🧠 TAB THETA4: Rotation calculations")
                    print()
                    print("💡 QUAN TRỌNG:")
                    print("- Sequential regions hiển thị chủ yếu ở TAB DEPTH")
                    print("- Sequential logic chạy tự động khi có pallets & loads")
                    print("- Keyboard controls chỉ là demo, actual logic chạy tự động")
                        
                # ⭐ SEQUENTIAL REGION CONTROLS - ADDED FOR PLAN IMPLEMENTATION ⭐
                elif key == ord('n'):
                    # ⭐ SEND REGIONS TO PLC ⭐
                    print("\n🚀 [PLC SENDING] Manual trigger: Send regions to PLC...")
                    print(f"🏭 PLC Integration enabled: {enable_plc}")
                    
                    if not enable_plc:
                        print("   ❌ PLC not enabled! Set ENABLE_PLC=true to enable.")
                        continue
                    
                    try:
                        # ⭐ STEP 1: DIRECT PLC ACCESS (PRIMARY METHOD) ⭐
                        print("   🎯 Using direct PLC access method...")
                        plc_integration = pipeline.get_plc_integration()
                        
                        if not plc_integration:
                            print("   ❌ PLC integration not available!")
                            continue
                        
                        if not plc_integration.plc_connected:
                            print("   🔌 PLC not connected, attempting to connect...")
                            connected = plc_integration.connect_plc()
                            if not connected:
                                print("   ❌ Failed to connect to PLC!")
                                continue
                            else:
                                print("   ✅ PLC connected successfully!")
                        
                        # ⭐ STEP 2: GET CURRENT FRAME DATA ⭐
                        print("   📡 Getting current frame data...")
                        
                        # Try multiple times with short intervals to catch current frame
                        detection_result = None
                        for attempt in range(5):
                            detection_result = pipeline.get_latest_detection(timeout=0.1)
                            if detection_result:
                                break
                            print(f"      Attempt {attempt+1}/5: Waiting for frame...")
                            time.sleep(0.05)  # Short wait
                        
                        if not detection_result:
                            print("   ⚠️ No current frame data available, using last known data...")
                            # Use any available regions from pipeline state
                            if hasattr(plc_integration, 'last_regions_data') and plc_integration.last_regions_data:
                                print("   📋 Using cached regions data from PLC integration...")
                                success = plc_integration.send_regions_to_plc()
                                if success:
                                    print("   ✅ Successfully sent cached regions to PLC!")
                                    
                                    # Show BAG PALLET STATUS
                                    bag_status = plc_integration.get_bag_pallet_status()
                                    print(f"   📦 BAG PALLET TRACKING:")
                                    print(f"      bag_pallet_1 = {bag_status['bag_pallet_1']}")
                                    print(f"      bag_pallet_2 = {bag_status['bag_pallet_2']}")
                                    print(f"      Active regions: {bag_status['active_regions_count']}")
                                    
                                    # Show regions
                                    for region_name, region_data in bag_status['current_regions'].items():
                                        if region_data:
                                            pallet_id = region_data['pallet_id']
                                            region_id = region_data['region_id']
                                            robot_coords = region_data['robot_coords']
                                            print(f"      {region_name}: P{pallet_id}R{region_id} → Px={robot_coords['px']:.2f}, Py={robot_coords['py']:.2f}")
                                            
                                    # READ BACK FROM PLC
                                    print("   🔍 Reading back from PLC to verify...")
                                    plc_data = plc_integration.read_regions_from_plc()
                                    if plc_data:
                                        print("   📊 PLC Memory Content:")
                                        for region_name, data in plc_data.items():
                                            print(f"      {region_name}: Px={data['px']:.2f} (DB26.{data['px_offset']}), Py={data['py']:.2f} (DB26.{data['py_offset']})")
                                else:
                                    print("   ❌ Failed to send cached regions to PLC")
                            else:
                                print("   ❌ No cached regions data available")
                            continue
                        
                        # ⭐ STEP 3: PROCESS AND SEND TO PLC ⭐
                        frame, detections = detection_result
                        print("   🔄 Processing current frame for PLC sending...")
                        
                        regions_data, success = plc_integration.process_detection_and_send_to_plc(detections, layer=1)
                        
                        if success:
                            print("   ✅ Successfully sent regions to PLC!")
                            print(f"   📋 Processed {len(regions_data)} regions")
                            
                            # ⭐ SHOW BAG PALLET TRACKING STATUS ⭐
                            bag_status = plc_integration.get_bag_pallet_status()
                            print(f"   📦 BAG PALLET TRACKING:")
                            print(f"      bag_pallet_1 = {bag_status['bag_pallet_1']}")
                            print(f"      bag_pallet_2 = {bag_status['bag_pallet_2']}")
                            print(f"      Active regions: {bag_status['active_regions_count']}")
                            
                            # ⭐ SHOW DETAILED REGION DATA ⭐
                            for region_name, region_data in bag_status['current_regions'].items():
                                if region_data:
                                    pallet_id = region_data['pallet_id']
                                    region_id = region_data['region_id']
                                    robot_coords = region_data['robot_coords']
                                    print(f"      {region_name}: P{pallet_id}R{region_id} → Px={robot_coords['px']:.2f}, Py={robot_coords['py']:.2f}")
                            
                            # ⭐ READ BACK FROM PLC TO VERIFY ⭐
                            print("   🔍 Reading back from PLC to verify...")
                            plc_data = plc_integration.read_regions_from_plc()
                            if plc_data:
                                print("   📊 PLC Memory Content:")
                                for region_name, data in plc_data.items():
                                    print(f"      {region_name}: Px={data['px']:.2f} (DB26.{data['px_offset']}), Py={data['py']:.2f} (DB26.{data['py_offset']})")
                        else:
                            print("   ❌ Failed to send regions to PLC")
                            if regions_data:
                                print(f"   📋 Regions were processed ({len(regions_data)}) but PLC sending failed")
                        
                    except Exception as e:
                        print(f"   ❌ Error in PLC sending process: {e}")
                        import traceback
                        traceback.print_exc()
                        
                elif key == ord('c'):
                    # Complete current region (robot hoàn thành, chuyển sang region tiếp theo)
                    print("\n✅ [SEQUENCE] Robot completed current region...")
                    try:
                        sequencer = pipeline.get_region_sequencer()
                        if sequencer and sequencer.is_available():
                            sequencer.mark_region_completed()
                            # SequencerProxy sẽ explain về manual completion
                        else:
                            print("   ⚠️ RegionSequencer đang khởi tạo hoặc chưa có detections")
                            print("   💡 Hãy chờ một chút để system phát hiện pallets")
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
                        
                elif key == ord('s'):
                    # Show sequence status
                    print("\n📊 [SEQUENCE] Current Status:")
                    try:
                        sequencer = pipeline.get_region_sequencer()
                        if sequencer and sequencer.is_available():
                            status = sequencer.get_queue_status()
                            print(f"   Status: {status['status']}")
                            if status['current_pallet']:
                                print(f"   Current Pallet: P{status['current_pallet']}")
                            print(f"   Progress: {status['progress']}")
                            print(f"   Sequence: {status['sequence']}")
                            print(f"   Completed: {status['completed_count']}")
                            print(f"   Remaining: {status['remaining_count']}")
                            
                            # Show remaining regions
                            if status['remaining_regions']:
                                print("   Remaining regions:")
                                for region in status['remaining_regions']:
                                    marker = "→ " if region.get('is_current') else "  "
                                    print(f"     {marker}Region {region.get('region_id', '?')} (seq {region.get('sequence_order', '?')})")
                            
                            print("   ℹ️ Sequential sending đang chạy tự động trong background")
                        else:
                            print("   ⚠️ RegionSequencer đang khởi tạo hoặc chưa có detections")
                            print("   💡 Hãy chờ một chút để system phát hiện pallets")
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
                        
                elif key == ord('x'):
                    # Reset sequence
                    print("\n🔄 [SEQUENCE] Resetting sequence...")
                    try:
                        sequencer = pipeline.get_region_sequencer()
                        if sequencer and sequencer.is_available():
                            sequencer.reset_sequence()
                            # SequencerProxy sẽ explain về manual reset
                        else:
                            print("   ⚠️ RegionSequencer đang khởi tạo hoặc chưa có detections")
                            print("   💡 Hãy chờ một chút để system phát hiện pallets")
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
                        
                elif key == ord('z'):
                    # Show depth info (Z values)
                    print("\n🔍 [DEPTH INFO] Current depth information:")
                    try:
                        depth_result = pipeline.get_latest_depth()
                        if depth_result:
                            frame_depth, depth_results = depth_result
                            print(f"   Found {len(depth_results)} depth regions:")
                            for i, region in enumerate(depth_results):
                                region_info = region.get('region_info', {})
                                position = region.get('position', {})
                                z_value = position.get('z', 0.0)
                                pallet_id = region_info.get('pallet_id', 0)
                                region_id = region_info.get('region_id', 0)
                                
                                if pallet_id > 0:
                                    print(f"     P{pallet_id}R{region_id}: Z={z_value:.3f}m")
                                else:
                                    obj_class = region_info.get('object_class', 'Unknown')
                                    print(f"     {obj_class}: Z={z_value:.3f}m")
                        else:
                            print("   ⚠️ No depth results available")
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
                elif key == ord('r'):
                    # Bật/tắt hiển thị regions
                    SHOW_REGIONS = not SHOW_REGIONS
                    # print(f"Hiển thị regions: {'BẬT' if SHOW_REGIONS else 'TẮT'}")
                    if not SHOW_REGIONS:
                        cv2.destroyWindow("Region Processing & Detections")
                        
        except KeyboardInterrupt:
            # print("Đã nhận tín hiệu ngắt từ bàn phím")
            pass
        finally:
            # Dừng pipeline
            pipeline.stop()
            cv2.destroyAllWindows()
            # print("Pipeline đã dừng")
    else:
        # print("Không thể khởi động pipeline!")
        # Kiểm tra lỗi
        for error in pipeline.errors:
            print(f"Lỗi: {error}")

def demo_sequential_region_sending():
    """Demo Sequential Region Sending với BAG PALLET TRACKING"""
    print("🚀 Demo Sequential Region Sending với BAG PALLET TRACKING")
    print("Hệ thống sẽ:")
    print("1. Phát hiện pallets và xác định workspace region (pallets1/pallets2)")
    print("2. Chia pallets thành 3 regions cho từng pallet")
    print("3. Gửi từng region một theo thứ tự, chờ robot hoàn thành")
    print("4. Theo dõi bag pallet tracking (load nào gắp vào pallet nào)")
    print()
    
    # Khởi tạo components
    model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)
    region_plc = RegionDivisionPLCIntegration(debug=True)
    
    # Test mode (không cần kết nối PLC thật)
    use_mock_plc = input("Sử dụng mock PLC? (y/n, mặc định y): ").lower()
    if use_mock_plc != 'n':
        print("🔧 Sử dụng mock PLC mode cho demo")
        region_plc.plc_connected = True  # Mock connection
        
        # Mock write function
        def mock_write_db26_real(offset, value):
            print(f"    [MOCK PLC] Write DB26.{offset} = {value:.2f}")
            return True
        region_plc.plc_comm.write_db26_real = mock_write_db26_real
    else:
        # Kết nối PLC thật
        region_plc.connect_plc()
    
    if not region_plc.plc_connected:
        print("❌ Không thể kết nối PLC, thoát demo")
        return
    
    # Demo với camera hoặc ảnh
    input_mode = input("Chọn input: (1) Camera real-time, (2) Ảnh từ folder (mặc định 2): ")
    
    if input_mode == "1":
        demo_sequential_with_camera(model, region_plc)
    else:
        demo_sequential_with_images(model, region_plc)

def demo_sequential_with_camera(model, region_plc):
    """Demo sequential sending với camera real-time"""
    print("\n📹 Demo với camera real-time")
    print("Phím điều khiển:")
    print("  'q': Thoát")
    print("  'n': Gửi region tiếp theo (manual)")
    print("  'c': Robot hoàn thành (complete signal)")
    print("  's': Hiển thị status")
    print("  'a': Bật/tắt auto sending")
    
    camera = CameraInterface(camera_index=0)
    if not camera.initialize():
        print("❌ Không thể khởi tạo camera")
        return
    
    try:
        frame_count = 0
        last_detection_time = 0
        
        while True:
            ret, frame = camera.read()
            if not ret:
                print("❌ Không thể đọc frame từ camera")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Chỉ detect mỗi 2 giây để tránh spam
            if current_time - last_detection_time > 2.0:
                print(f"\n🔍 Frame {frame_count}: Đang phát hiện...")
                
                # YOLO detection
                detections = model.detect(frame)
                num_objects = len(detections.get('bounding_boxes', []))
                
                if num_objects > 0:
                    print(f"   Phát hiện {num_objects} objects")
                    
                    # Xử lý sequential region sending
                    regions_data, _ = region_plc.process_detection_and_send_to_plc(detections, layer=1)
                    
                    if regions_data:
                        print(f"   Đã tạo {len(regions_data)} regions và organize vào queues")
                    
                    last_detection_time = current_time
                else:
                    print(f"   Không phát hiện objects nào")
            
            # Tạo visualization
            if region_plc.last_regions_data:
                display_frame = region_plc.create_visualization(frame)
            else:
                display_frame = frame.copy()
                # Vẽ thông tin status cơ bản
                cv2.putText(display_frame, "Sequential Region Sending Demo", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(display_frame, "Waiting for detections...", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Sequential Region Sending Demo", display_frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Manual send next region
                print("\n🤖 Manual trigger: Gửi region tiếp theo...")
                success = region_plc.manual_send_next()
                print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
            elif key == ord('c'):
                # Robot completed signal
                print("\n🤖✅ Robot completed signal...")
                success = region_plc.robot_completed_signal()
                print(f"   Next send result: {'✅ Success' if success else '❌ No more regions'}")
            elif key == ord('s'):
                # Show status
                status = region_plc.get_queue_status()
                print(f"\n📊 SYSTEM STATUS:")
                print(f"   Robot: {status['robot_status']}")
                print(f"   Queues: pallets1={status['queue_counts']['pallets1']}, pallets2={status['queue_counts']['pallets2']}")
                print(f"   Total pending: {status['total_pending']}")
                print(f"   Stats: Sent {status['stats']['total_regions_sent']} regions")
                print(f"   BAG Tracking: pallet1={status['bag_pallet_1']}, pallet2={status['bag_pallet_2']}")
                if status['current_sending_region']:
                    current = status['current_sending_region']
                    print(f"   Current sending: {current['workspace_region']} P{current['pallet_id']}R{current['region_id']}")
            elif key == ord('a'):
                # Toggle auto sending
                region_plc.auto_send_enabled = not region_plc.auto_send_enabled
                print(f"\n🔄 Auto sending: {'ENABLED' if region_plc.auto_send_enabled else 'DISABLED'}")
            
    finally:
        camera.release()
        cv2.destroyAllWindows()
        region_plc.disconnect_plc()

def demo_sequential_with_images(model, region_plc):
    """Demo sequential sending với ảnh từ folder"""
    print("\n🖼️ Demo với ảnh từ folder")
    
    pallets_folder = "images_pallets2"
    if not os.path.exists(pallets_folder):
        print(f"❌ Không tìm thấy folder {pallets_folder}")
        return
    
    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(pallets_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    if not image_files:
        print("❌ Không có ảnh nào trong folder")
        return
    
    print(f"📁 Tìm thấy {len(image_files)} ảnh")
    for i, img_file in enumerate(image_files, 1):
        print(f"  {i}. {img_file}")
    
    try:
        for i, img_file in enumerate(image_files, 1):
            image_path = os.path.join(pallets_folder, img_file)
            print(f"\n[{i}/{len(image_files)}] 🔍 Xử lý: {img_file}")
            
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"   ❌ Không thể đọc ảnh: {img_file}")
                continue
            
            # YOLO detection
            detections = model.detect(frame)
            num_objects = len(detections.get('bounding_boxes', []))
            
            print(f"   Phát hiện {num_objects} objects")
            
            if num_objects > 0:
                # Xử lý sequential region sending
                regions_data, _ = region_plc.process_detection_and_send_to_plc(detections, layer=1)
                
                if regions_data:
                    print(f"   ✅ Đã tạo {len(regions_data)} regions")
                    
                    # Show detailed regions
                    for region in regions_data:
                        workspace_region = region['workspace_region']
                        pallet_id = region['pallet_id']
                        region_id = region['region_id']
                        robot_coords = region['robot_coordinates']
                        print(f"      {workspace_region} P{pallet_id}R{region_id}: Px={robot_coords['px']:.2f}, Py={robot_coords['py']:.2f}")
                    
                    # Hiển thị visualization
                    visualization = region_plc.create_visualization(frame)
                    
                    cv2.imshow(f"Sequential Demo - {img_file}", visualization)
                    cv2.imshow(f"Original Detection - {img_file}", detections["annotated_frame"])
                    
                    print(f"   📋 Queue status:")
                    status = region_plc.get_queue_status()
                    for workspace_region, count in status['queue_counts'].items():
                        if count > 0:
                            print(f"      {workspace_region}: {count} regions pending")
                    
                    print(f"   🤖 Robot status: {status['robot_status']}")
                    if status['current_sending_region']:
                        current = status['current_sending_region']
                        print(f"   ⚡ Currently sending: {current['workspace_region']} P{current['pallet_id']}R{current['region_id']}")
                    
                    print(f"\n   Nhấn phím để điều khiển:")
                    print(f"     'n': Send next region")
                    print(f"     'c': Robot completed")
                    print(f"     'space': Next image")
                    print(f"     'q': Quit")
                    
                    # Wait for user input
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('n'):
                            # Send next
                            success = region_plc.manual_send_next()
                            print(f"      {'✅ Sent next region' if success else '❌ No more regions to send'}")
                            if success:
                                new_status = region_plc.get_queue_status()
                                if new_status['current_sending_region']:
                                    current = new_status['current_sending_region']
                                    print(f"      Now sending: {current['workspace_region']} P{current['pallet_id']}R{current['region_id']}")
                        elif key == ord('c'):
                            # Robot completed
                            success = region_plc.robot_completed_signal()
                            print(f"      🤖✅ Robot completed. {'Next region sent' if success else 'All regions completed'}")
                            if success:
                                new_status = region_plc.get_queue_status()
                                if new_status['current_sending_region']:
                                    current = new_status['current_sending_region']
                                    print(f"      Now sending: {current['workspace_region']} P{current['pallet_id']}R{current['region_id']}")
                        elif key == ord(' '):
                            # Next image
                            break
                        elif key == ord('q'):
                            cv2.destroyAllWindows()
                            return
                else:
                    print(f"   ❌ Không tạo được regions")
            else:
                print(f"   ⚠️ Không có objects để xử lý")
                
                # Hiển thị ảnh gốc
                cv2.imshow(f"No Detection - {img_file}", frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
            
            cv2.destroyAllWindows()
    
    finally:
        region_plc.disconnect_plc()
        print(f"\n📊 FINAL STATISTICS:")
        final_status = region_plc.get_queue_status()
        print(f"   Total regions processed: {final_status['stats']['total_regions_detected']}")
        print(f"   Total regions sent: {final_status['stats']['total_regions_sent']}")
        print(f"   Pallets1 sent: {final_status['stats']['pallets1_sent']}")
        print(f"   Pallets2 sent: {final_status['stats']['pallets2_sent']}")
        print(f"   Final BAG tracking: pallet1={final_status['bag_pallet_1']}, pallet2={final_status['bag_pallet_2']}")

if __name__ == "__main__":
    print("Demo sử dụng model TensorRT với Rotated Bounding Boxes, Module Division, Theta4 Calculation và Region Processing")
    print("1. Thử nghiệm với ảnh đơn lẻ (có Module Division + depth estimation)")
    print("2. Thử nghiệm với camera thời gian thực (có Module Division + depth estimation + Theta4 + Regions)")
    print("3. Thử nghiệm với tất cả ảnh trong folder images_pallets2 (có Module Division + depth estimation)")
    print("4. 🚀 Sequential Region Sending với BAG PALLET TRACKING (Tính năng mới!)")
    print("\nTÍNH NĂNG MỚI - SEQUENTIAL REGION SENDING (Option 4):")
    print("- Phát hiện pallets và xác định workspace region (pallets1/pallets2)")  
    print("- Chia pallets thành 3 regions và gửi từng region một theo thứ tự")
    print("- BAG PALLET TRACKING: Theo dõi load nào gắp vào pallet nào")
    print("- Sequential sending: Chờ robot hoàn thành trước khi gửi region tiếp theo")
    print("- PLC Communication với DB26 offsets riêng cho từng workspace region")
    print("\nTÍNH NĂNG REGION PROCESSING:")
    print("- Chia không gian làm việc thành 3 regions cố định: loads, pallets1, pallets2")  
    print("- Mỗi region có thể có offset riêng cho robot coordinates")
    print("- Detections được filter và group theo regions")
    print("- Sử dụng phím 'r' để bật/tắt hiển thị regions trong real-time")
    print("\nTÍNH NĂNG THETA4 CALCULATION:")
    print("- Chế độ camera (2) hiện có tính toán góc xoay theta4 cho robot")
    print("- Sử dụng phím 't' để bật/tắt hiển thị theta4 trong real-time")
    print("- Theta4 sẽ hiển thị góc cần xoay cho từng load để đặt vào regions")
    print("- Bao gồm visualization với mapping lines và rotation commands")
    print("\nGhi chú:")
    print("- Tất cả các demo đều sử dụng Module Division để chia pallet thành các vùng nhỏ")
    print("- Depth estimation được thực hiện cho từng vùng riêng biệt")
    print("- Theta4 calculation chỉ hoạt động khi có loads (class 0,1) và regions từ pallets (class 2)")
    print("- Sequential Region Sending (4) là tính năng hoàn chỉnh cho robot gắp theo thứ tự")
    print("- Tất cả các demo đều sử dụng chung cấu hình depth model")
    print("Bạn có thể đặt các biến môi trường để điều khiển các tính năng:")
    print("\n🏭 PLC INTEGRATION:")
    print("  ENABLE_PLC: Bật/tắt PLC integration")
    print("    - ENABLE_PLC=true     # Bật PLC integration")
    print("    - ENABLE_PLC=false    # Tắt PLC integration (mặc định)")
    print("  PLC_IP: IP address của PLC")
    print("    - PLC_IP=192.168.0.1  # IP PLC (mặc định)")
    print("    - PLC_IP=192.168.1.100 # Example custom IP")
    print("\n📏 DEPTH MODEL:")
    print("  DEPTH_DEVICE: Thiết bị chạy depth model")
    print("    - DEPTH_DEVICE=cuda   # Chạy trên GPU (mặc định nếu có CUDA)")
    print("    - DEPTH_DEVICE=cpu    # Chạy trên CPU")
    print("    - DEPTH_DEVICE=off    # Tắt hoàn toàn depth model (mặc định)")
    print("\n  DEPTH_TYPE: Loại mô hình depth")
    print("    - DEPTH_TYPE=regular  # Regular depth model (normalized output)")
    print("    - DEPTH_TYPE=metric   # Metric depth model (output in meters)")
    print("\n  DEPTH_MODEL: Kích thước mô hình để tăng tốc độ")
    print("    - DEPTH_MODEL=large   # Mô hình lớn, chất lượng cao, chậm nhất")
    print("    - DEPTH_MODEL=base    # Mô hình vừa, cân bằng tốc độ/chất lượng")
    print("    - DEPTH_MODEL=small   # Mô hình nhỏ, tốc độ nhanh nhất (mặc định)")
    print("\n  DEPTH_SCENE: Loại cảnh (chỉ cho metric depth)")
    print("    - DEPTH_SCENE=indoor  # Cảnh trong nhà (mặc định)")
    print("    - DEPTH_SCENE=outdoor # Cảnh ngoài trời")
    print("\n  DEPTH_SIZE: Kích thước đầu vào (W,H) để tăng tốc")
    print("    - DEPTH_SIZE=640x480  # Ví dụ: 640x480")
    print("\n  DEPTH_SKIP: Số frame bỏ qua giữa các lần xử lý")
    print("    - DEPTH_SKIP=5        # Ví dụ: Chỉ xử lý 1 frame trong mỗi 6 frames")
    print("\n  SHOW_DEPTH: Bật/tắt hiển thị depth map")
    print("    - SHOW_DEPTH=true     # Hiển thị depth map (có thể gây lag)")
    print("    - SHOW_DEPTH=false    # Tắt hiển thị depth map (mặc định)")
    print("\n  SHOW_THETA4: Bật/tắt hiển thị theta4 calculation")
    print("    - SHOW_THETA4=true    # Hiển thị theta4 info (có thể gây lag)")
    print("    - SHOW_THETA4=false   # Tắt hiển thị theta4 info (mặc định)")
    print("\n  SHOW_REGIONS: Bật/tắt hiển thị region processing")
    print("    - SHOW_REGIONS=true   # Hiển thị regions và detections theo regions (mặc định)")
    print("    - SHOW_REGIONS=false  # Tắt hiển thị regions")
    print("\n  USE_CAMERA_CALIBRATION: Bật/tắt camera calibration")
    print("    - USE_CAMERA_CALIBRATION=true    # Sử dụng camera calibration (mặc định)")
    print("    - USE_CAMERA_CALIBRATION=false   # Tắt camera calibration")
    print("\n  CAMERA_CALIBRATION_FILE: Đường dẫn file camera calibration")
    print("    - CAMERA_CALIBRATION_FILE=camera_params.npz  # File calibration (mặc định)")
    print("\n  Ví dụ với Theta4 enabled:")
    print("    set SHOW_THETA4=true && python use_tensorrt_example.py")
    print("\n  Ví dụ Regular Depth với Camera Calibration và Theta4:")
    print("    set DEPTH_DEVICE=cuda && set DEPTH_TYPE=regular && set DEPTH_MODEL=small && set USE_CAMERA_CALIBRATION=true && set SHOW_THETA4=true && python use_tensorrt_example.py")
    print("\n  Ví dụ Metric Depth (Indoor) với Camera Calibration và Theta4:")
    print("    set DEPTH_DEVICE=cuda && set DEPTH_TYPE=metric && set DEPTH_SCENE=indoor && set DEPTH_MODEL=base && set USE_CAMERA_CALIBRATION=true && set SHOW_THETA4=true && python use_tensorrt_example.py")
    print("\n  Ví dụ Metric Depth (Outdoor) không có Camera Calibration:")
    print("    set DEPTH_DEVICE=cuda && set DEPTH_TYPE=metric && set DEPTH_SCENE=outdoor && set DEPTH_MODEL=small && set USE_CAMERA_CALIBRATION=false && python use_tensorrt_example.py")
    print("\n  Ví dụ với file calibration tùy chỉnh:")
    print("    set USE_CAMERA_CALIBRATION=true && set CAMERA_CALIBRATION_FILE=my_camera_calib.npz && python use_tensorrt_example.py")
    print("\n  Ví dụ với Regions và Theta4 enabled:")
    print("    set SHOW_REGIONS=true && set SHOW_THETA4=true && python use_tensorrt_example.py")
    print("\n  Ví dụ full features (Depth + Theta4 + Regions):")
    print("    set DEPTH_DEVICE=cuda && set DEPTH_TYPE=metric && set SHOW_DEPTH=true && set SHOW_THETA4=true && set SHOW_REGIONS=true && python use_tensorrt_example.py")
    print()
    print("🏭 PLC INTEGRATION EXAMPLES:")
    print("  Ví dụ với PLC enabled (demo mode):")
    print("    set ENABLE_PLC=true && python use_tensorrt_example.py")
    print()
    print("  Ví dụ với PLC thật (custom IP):")
    print("    set ENABLE_PLC=true && set PLC_IP=192.168.1.100 && python use_tensorrt_example.py")
    print()
    print("  Ví dụ FULL FEATURES với PLC:")
    print("    set ENABLE_PLC=true && set PLC_IP=192.168.0.1 && set DEPTH_DEVICE=cuda && set SHOW_THETA4=true && set SHOW_REGIONS=true && python use_tensorrt_example.py")
    print()
    print("💡 KHI PLC ENABLED:")
    print("  - Nhấn 'n' để THẬT SỰ gửi regions vào PLC DB26")
    print("  - BAG PALLET TRACKING sẽ hoạt động")
    print("  - Regions được map theo: loads→DB26.0/4, pallets1→DB26.12/16, pallets2→DB26.24/28")
    print()
    
    choice = input("Chọn chế độ (1/2/3/4): ")
    
    if choice == "1":
        demo_single_image()
    elif choice == "2":
        demo_camera()
    elif choice == "3":
        demo_batch_images()
    elif choice == "4":
        demo_sequential_region_sending()
    else:
        print("Lựa chọn không hợp lệ!") 