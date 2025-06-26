"""
Demo phân tích góc xoay và tính toán theta 4 cho robot IRB-460.
Tích hợp với hệ thống YOLO detection và module division hiện có.
"""
import cv2
import os
import numpy as np
import time
from detection import (YOLOTensorRT, ModuleDivision)
from detection.utils.rotation_analyzer import RotationAnalyzer

# Đường dẫn model (sử dụng chung với use_tensorrt_example.py)
ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "utils", "detection", "best.engine")

def demo_rotation_analysis_single_image():
    """
    Demo phân tích góc xoay trên một ảnh đơn lẻ.
    """
    print("=== DEMO PHÂN TÍCH GÓC XOAY VÀ TÍNH TOÁN THETA 4 ===")
    print("Chọn ảnh để phân tích:")
    
    # Hiển thị ảnh có sẵn
    pallets_folder = "images_pallets"
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
    
    print("Khởi tạo Rotation Analyzer...")
    analyzer = RotationAnalyzer(debug=True)
    
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
    analysis_result = analyzer.process_complete_analysis(detections, division_regions, layer)
    analysis_time = time.time() - start_time
    
    print(f"Thời gian phân tích: {analysis_time*1000:.2f} ms")
    
    # Hiển thị kết quả chi tiết
    print("\n" + "="*50)
    print("KẾT QUẢ PHÂN TÍCH CHI TIẾT")
    print("="*50)
    
    summary = analysis_result['summary']
    print(f"Tổng số objects: {summary['total_objects']}")
    print(f"Số load objects: {summary['load_count']}")
    print(f"Số pallet objects: {summary['pallet_count']}")
    print(f"Số regions: {summary['regions_count']}")
    print(f"Số tính toán theta 4: {summary['theta4_count']}")
    
    # Hiển thị thông tin theta 4
    if analysis_result['theta4_calculations']:
        print(f"\n--- LỆNH THETA 4 CHO ROBOT IRB-460 ---")
        for i, calc in enumerate(analysis_result['theta4_calculations']):
            print(f"\nLoad #{i+1}:")
            print(f"  Góc hiện tại trên băng tải: {calc['load_angle']:.1f}°")
            print(f"  Góc đặt mục tiêu trên pallet: {calc['target_angle']:.1f}°")
            print(f"  Góc xoay cần thiết: {calc['rotation_sign']}{abs(calc['rotation_normalized']):.1f}°")
            print(f"  Hướng xoay: {calc['rotation_direction']}")
            print(f"  ➤ LỆNH ROBOT: {calc['theta4_command']}")
            
            # Thông tin region đích
            target_region = calc['target_region']
            print(f"  Đặt vào: Pallet {target_region['pallet_id']}, Region {target_region['region_id']}")
    else:
        print("\nKhông tìm thấy load objects để tính toán theta 4!")
    
    # Tạo visualization
    print(f"\n--- TẠO VISUALIZATION ---")
    vis_image = analyzer.create_visualization(frame, analysis_result)
    
    # Hiển thị ảnh
    print(f"\nHiển thị kết quả visualization...")
    
    # Ảnh detection gốc
    cv2.imshow("YOLO Detection", detections["annotated_frame"])
    
    # Ảnh phân tích góc xoay
    cv2.imshow("Rotation Analysis", vis_image)
    
    print(f"\nẢnh đã được hiển thị. Nhấn phím bất kỳ để tiếp tục...")
    cv2.waitKey(0)
    
    # Lưu kết quả
    save_choice = input("\nBạn có muốn lưu kết quả? (y/n): ").lower()
    if save_choice in ['y', 'yes']:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Lưu ảnh visualization
        vis_output_path = f"rotation_analysis_{base_name}_layer{layer}.jpg"
        cv2.imwrite(vis_output_path, vis_image)
        print(f"Đã lưu visualization: {vis_output_path}")
        
        # Lưu ảnh detection
        detection_output_path = f"detection_{base_name}_layer{layer}.jpg"
        cv2.imwrite(detection_output_path, detections["annotated_frame"])
        print(f"Đã lưu detection: {detection_output_path}")
        
        # Lưu báo cáo text
        report_path = f"rotation_report_{base_name}_layer{layer}.txt"
        save_text_report(analysis_result, report_path, image_path, layer)
        print(f"Đã lưu báo cáo: {report_path}")
    
    cv2.destroyAllWindows()

def demo_rotation_analysis_batch():
    """
    Demo phân tích góc xoay trên tất cả ảnh trong folder.
    """
    print("=== DEMO PHÂN TÍCH GÓC XOAY BATCH ===")
    
    pallets_folder = "images_pallets"
    if not os.path.exists(pallets_folder):
        print(f"Không tìm thấy folder: {pallets_folder}")
        return
    
    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(pallets_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    if not image_files:
        print("Không có ảnh nào trong folder!")
        return
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    
    # Chọn layer
    layer_choice = input("Chọn layer (1 hoặc 2, mặc định 1): ")
    try:
        layer = int(layer_choice) if layer_choice else 1
    except ValueError:
        layer = 1
    
    # Khởi tạo models
    print(f"\nKhởi tạo models...")
    yolo_model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)
    divider = ModuleDivision()
    analyzer = RotationAnalyzer(debug=False)  # Tắt debug cho batch
    
    # Tạo folder kết quả
    output_folder = f"rotation_analysis_results_layer{layer}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Thống kê
    total_time = 0
    successful_analysis = 0
    total_theta4_commands = 0
    
    print(f"\nBắt đầu xử lý {len(image_files)} ảnh...")
    
    for i, img_file in enumerate(image_files, 1):
        image_path = os.path.join(pallets_folder, img_file)
        print(f"\n[{i}/{len(image_files)}] Xử lý: {img_file}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  Không thể đọc ảnh!")
            continue
        
        start_time = time.time()
        
        try:
            # YOLO detection
            detections = yolo_model.detect(frame)
            
            if len(detections.get('bounding_boxes', [])) == 0:
                print(f"  Không phát hiện object nào")
                continue
            
            # Module division
            divided_result = divider.process_pallet_detections(detections, layer=layer)
            division_regions = divider.prepare_for_depth_estimation(divided_result)
            
            # Phân tích góc xoay
            analysis_result = analyzer.process_complete_analysis(detections, division_regions, layer)
            
            process_time = time.time() - start_time
            total_time += process_time
            
            # Thống kê kết quả
            summary = analysis_result['summary']
            theta4_count = summary['theta4_count']
            total_theta4_commands += theta4_count
            
            print(f"  Objects: {summary['total_objects']}, Loads: {summary['load_count']}, Regions: {summary['regions_count']}")
            print(f"  Theta4 commands: {theta4_count}")
            print(f"  Thời gian: {process_time*1000:.1f} ms")
            
            if theta4_count > 0:
                successful_analysis += 1
                
                # Hiển thị lệnh theta 4
                for j, calc in enumerate(analysis_result['theta4_calculations']):
                    print(f"    Load {j+1}: {calc['theta4_command']}")
            
            # Tạo và lưu visualization
            vis_image = analyzer.create_visualization(frame, analysis_result)
            
            base_name = os.path.splitext(img_file)[0]
            vis_output_path = os.path.join(output_folder, f"rotation_{base_name}.jpg")
            cv2.imwrite(vis_output_path, vis_image)
            
            # Lưu báo cáo text
            report_path = os.path.join(output_folder, f"report_{base_name}.txt")
            save_text_report(analysis_result, report_path, image_path, layer)
            
        except Exception as e:
            print(f"  Lỗi xử lý: {e}")
            continue
    
    # Thống kê tổng kết
    print(f"\n" + "="*50)
    print(f"THỐNG KÊ TỔNG KẾT")
    print(f"="*50)
    print(f"Tổng số ảnh xử lý: {len(image_files)}")
    print(f"Ảnh phân tích thành công: {successful_analysis}")
    print(f"Tỉ lệ thành công: {successful_analysis/len(image_files)*100:.1f}%")
    print(f"Tổng số lệnh theta 4: {total_theta4_commands}")
    print(f"Thời gian trung bình: {total_time/len(image_files)*1000:.1f} ms/ảnh")
    print(f"Kết quả được lưu trong: {output_folder}")

def save_text_report(analysis_result: dict, file_path: str, image_path: str, layer: int):
    """
    Lưu báo cáo chi tiết dưới dạng text file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("BÁO CÁO PHÂN TÍCH GÓC XOAY VÀ TÍNH TOÁN THETA 4\n")
        f.write("="*60 + "\n")
        f.write(f"Ảnh: {image_path}\n")
        f.write(f"Layer: {layer}\n")
        f.write(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        # Tổng quan
        summary = analysis_result['summary']
        f.write("TỔNG QUAN:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Tổng số objects: {summary['total_objects']}\n")
        f.write(f"Load objects: {summary['load_count']}\n")
        f.write(f"Pallet objects: {summary['pallet_count']}\n")
        f.write(f"Regions: {summary['regions_count']}\n")
        f.write(f"Lệnh theta 4: {summary['theta4_count']}\n")
        f.write("\n")
        
        # Chi tiết objects
        f.write("CHI TIẾT OBJECTS:\n")
        f.write("-" * 20 + "\n")
        for i, obj in enumerate(analysis_result['yolo_angles']):
            f.write(f"Object {i+1} ({obj['class_name']}):\n")
            f.write(f"  Center: ({obj['center'][0]:.1f}, {obj['center'][1]:.1f})\n")
            f.write(f"  Size: {obj['size'][0]:.1f} x {obj['size'][1]:.1f}\n")
            f.write(f"  Góc: {obj['angle_normalized']:.1f}°\n")
            f.write(f"  Hướng: {obj['angle_info']['direction']}\n")
            f.write("\n")
        
        # Chi tiết regions
        f.write("CHI TIẾT REGIONS:\n")
        f.write("-" * 20 + "\n")
        for layout in analysis_result['layout_info']:
            f.write(f"Pallet {layout['pallet_id']}, Region {layout['region_id']}:\n")
            f.write(f"  Center: ({layout['center'][0]:.1f}, {layout['center'][1]:.1f})\n")
            f.write(f"  Góc đặt: {layout['placement_angle']:.1f}°\n")
            f.write("\n")
        
        # Lệnh theta 4
        f.write("LỆNH THETA 4 CHO ROBOT IRB-460:\n")
        f.write("-" * 40 + "\n")
        if analysis_result['theta4_calculations']:
            for i, calc in enumerate(analysis_result['theta4_calculations']):
                f.write(f"Load #{i+1}:\n")
                f.write(f"  Góc hiện tại: {calc['load_angle']:.1f}°\n")
                f.write(f"  Góc đích: {calc['target_angle']:.1f}°\n")
                f.write(f"  Xoay: {calc['rotation_sign']}{abs(calc['rotation_normalized']):.1f}°\n")
                f.write(f"  Hướng: {calc['rotation_direction']}\n")
                f.write(f"  ➤ LỆNH: {calc['theta4_command']}\n")
                
                target = calc['target_region']
                f.write(f"  Đặt vào: Pallet {target['pallet_id']}, Region {target['region_id']}\n")
                f.write("\n")
        else:
            f.write("Không có lệnh theta 4 nào được tạo.\n")

def main():
    """
    Menu chính để chọn chế độ demo.
    """
    print("DEMO PHÂN TÍCH GÓC XOAY VÀ TÍNH TOÁN THETA 4")
    print("Tích hợp với YOLO Detection và Module Division")
    print("="*50)
    print("1. Phân tích ảnh đơn lẻ (chi tiết)")
    print("2. Phân tích batch tất cả ảnh (nhanh)")
    print()
    print("Lưu ý:")
    print("- Chương trình sẽ phân tích góc xoay từ YOLO OBB detection")
    print("- Sử dụng Module Division để xác định layout đặt load")
    print("- Tính toán góc xoay theta 4 cần thiết cho robot IRB-460")
    print("- Hiển thị visualization với arrows và thông tin chi tiết")
    print()
    
    choice = input("Chọn chế độ (1/2): ")
    
    if choice == "1":
        demo_rotation_analysis_single_image()
    elif choice == "2":
        demo_rotation_analysis_batch()
    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main() 