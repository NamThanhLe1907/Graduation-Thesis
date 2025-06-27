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
                       ModuleDivision)

# Đường dẫn tới file model - sử dụng .pt thay vì .engine để tránh lỗi version
ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "utils", "detection", "best.engine")
# Cấu hình hiển thị depth - mặc định là False để tránh lag
SHOW_DEPTH = os.environ.get('SHOW_DEPTH', 'false').lower() in ('true', '1', 'yes')

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
        print(f"[Factory] Đã tắt depth model để tiết kiệm tài nguyên")
        return DepthEstimator(device='cpu', enable=False, use_camera_calibration=use_calibration, camera_calibration_file=calibration_file)
    
    print(f"[Factory] Khởi tạo depth model trên thiết bị: {use_device}")
    print(f"[Factory] Model type: {model_type}, Size: {model_size}")
    if model_type == 'metric':
        print(f"[Factory] Scene type: {scene_type}")
    print(f"[Factory] Camera calibration: {'Bật' if use_calibration else 'Tắt'}")
    if use_calibration:
        print(f"[Factory] Calibration file: {calibration_file}")
    
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
    print("Thử nghiệm phát hiện đối tượng với TensorRT trên camera thời gian thực")
    global SHOW_DEPTH
    
    # Hiển thị tùy chọn depth
    print(f"Hiển thị depth map: {'BẬT' if SHOW_DEPTH else 'TẮT'} (Dùng 'd' để bật/tắt)")
    
    # Khởi tạo pipeline với các factory function đã được định nghĩa ở cấp module
    pipeline = ProcessingPipeline(
        camera_factory=create_camera,
        yolo_factory=create_yolo,
        depth_factory=create_depth
    )
    
    # Biến để lưu frame depth cuối cùng
    last_depth_viz = None
    last_depth_time = 0
    skip_counter = 0
    max_skip = 5  # Bỏ qua tối đa 3 frames khi xử lý không kịp
    
    # Khởi động pipeline
    if pipeline.start(timeout=60.0):
        print("Pipeline đã khởi động thành công!")
        
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
                        # Bỏ qua hiển thị depth để giảm tải
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
                
                # Xử lý phím nhấn
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    # Bật/tắt hiển thị depth
                    SHOW_DEPTH = not SHOW_DEPTH
                    print(f"Hiển thị depth map: {'BẬT' if SHOW_DEPTH else 'TẮT'}")
                    if not SHOW_DEPTH:
                        cv2.destroyWindow("Rotated Depth Regions")
                        
        except KeyboardInterrupt:
            print("Đã nhận tín hiệu ngắt từ bàn phím")
        finally:
            # Dừng pipeline
            pipeline.stop()
            cv2.destroyAllWindows()
            print("Pipeline đã dừng")
    else:
        print("Không thể khởi động pipeline!")
        # Kiểm tra lỗi
        for error in pipeline.errors:
            print(f"Lỗi: {error}")

if __name__ == "__main__":
    print("Demo sử dụng model TensorRT với Rotated Bounding Boxes và Module Division")
    print("1. Thử nghiệm với ảnh đơn lẻ (có Module Division + depth estimation)")
    print("2. Thử nghiệm với camera thời gian thực (có Module Division + depth estimation)")
    print("3. Thử nghiệm với tất cả ảnh trong folder images_pallets2 (có Module Division + depth estimation)")
    print("\nGhi chú:")
    print("- Tất cả các demo đều sử dụng Module Division để chia pallet thành các vùng nhỏ")
    print("- Depth estimation được thực hiện cho từng vùng riêng biệt")
    print("- Tất cả các demo đều sử dụng chung cấu hình depth model")
    print("Bạn có thể đặt các biến môi trường để điều khiển depth model:")
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
    print("\n  USE_CAMERA_CALIBRATION: Bật/tắt camera calibration")
    print("    - USE_CAMERA_CALIBRATION=true    # Sử dụng camera calibration (mặc định)")
    print("    - USE_CAMERA_CALIBRATION=false   # Tắt camera calibration")
    print("\n  CAMERA_CALIBRATION_FILE: Đường dẫn file camera calibration")
    print("    - CAMERA_CALIBRATION_FILE=camera_params.npz  # File calibration (mặc định)")
    print("\n  Ví dụ Regular Depth với Camera Calibration:")
    print("    set DEPTH_DEVICE=cuda && set DEPTH_TYPE=regular && set DEPTH_MODEL=small && set USE_CAMERA_CALIBRATION=true && python use_tensorrt_example.py")
    print("\n  Ví dụ Metric Depth (Indoor) với Camera Calibration:")
    print("    set DEPTH_DEVICE=cuda && set DEPTH_TYPE=metric && set DEPTH_SCENE=indoor && set DEPTH_MODEL=base && set USE_CAMERA_CALIBRATION=true && python use_tensorrt_example.py")
    print("\n  Ví dụ Metric Depth (Outdoor) không có Camera Calibration:")
    print("    set DEPTH_DEVICE=cuda && set DEPTH_TYPE=metric && set DEPTH_SCENE=outdoor && set DEPTH_MODEL=small && set USE_CAMERA_CALIBRATION=false && python use_tensorrt_example.py")
    print("\n  Ví dụ với file calibration tùy chỉnh:")
    print("    set USE_CAMERA_CALIBRATION=true && set CAMERA_CALIBRATION_FILE=my_camera_calib.npz && python use_tensorrt_example.py")
    print()
    
    choice = input("Chọn chế độ (1/2/3): ")
    
    if choice == "1":
        demo_single_image()
    elif choice == "2":
        demo_camera()
    elif choice == "3":
        demo_batch_images()
    else:
        print("Lựa chọn không hợp lệ!") 