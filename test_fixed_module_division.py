"""
Test script để kiểm tra module division đã được sửa chữa.
Kiểm tra xem layer 1 có chia đúng theo chiều dài (12cm) và layer 2 chia theo chiều ngang (10cm) không.
"""
import cv2
import os
import numpy as np
from detection import YOLOTensorRT, ModuleDivision

# Đường dẫn model 
ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "utils", "detection", "best.engine")

def test_fixed_module_division():
    """
    Test module division với ảnh rotation_image_14.jpg để kiểm tra sửa chữa.
    """
    print("=== TEST MODULE DIVISION ĐÃ ĐƯỢC SỬA CHỮA ===")
    
    # Đọc ảnh từ rotation_analysis_results_layer1/rotation_image_14.jpg
    image_path = "rotation_analysis_results_layer1/rotation_image_19.jpg"
    
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh: {image_path}")
        return
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    print(f"Đang test với ảnh: {image_path}")
    height, width = frame.shape[:2]
    print(f"Kích thước ảnh: {width} x {height}")
    
    # Khởi tạo models với debug mode
    print("\nKhởi tạo YOLO model...")
    yolo_model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)
    
    print("Khởi tạo Module Division với debug mode...")
    divider = ModuleDivision(debug=True)
    
    # Thực hiện detection
    print("\n" + "="*60)
    print("BƯỚC 1: YOLO DETECTION")
    print("="*60)
    
    detections = yolo_model.detect(frame)
    print(f"Số objects phát hiện: {len(detections.get('bounding_boxes', []))}")
    
    if len(detections.get('bounding_boxes', [])) == 0:
        print("Không phát hiện object nào!")
        return
    
    # Test với cả 2 layer
    for layer in [1, 2]:
        print("\n" + "="*60)
        print(f"BƯỚC 2: MODULE DIVISION - LAYER {layer}")
        print("="*60)
        
        divided_result = divider.process_pallet_detections(detections, layer=layer)
        division_regions = divider.prepare_for_depth_estimation(divided_result)
        
        print(f"Số regions được tạo: {len(division_regions)}")
        
        # Hiển thị thông tin chi tiết từng region
        for i, region in enumerate(division_regions):
            region_info = region['region_info']
            print(f"\n  Region {i+1}:")
            print(f"    Region ID: {region_info['region_id']}")
            print(f"    Pallet ID: {region_info['pallet_id']}")
            print(f"    Layer: {region_info['layer']}")
            print(f"    Center: ({region['center'][0]:.1f}, {region['center'][1]:.1f})")
            
            # Thông tin thêm từ corners nếu có
            if 'corners' in region:
                print(f"    Có corners: 4 điểm")
            
            # Thông tin division direction nếu có
            if 'corners' in region and len(divided_result['divided_regions']) > i:
                orig_region = divided_result['divided_regions'][i]
                if 'division_direction' in orig_region:
                    print(f"    Hướng chia: {orig_region['division_direction']}")
                if 'real_dimension_direction' in orig_region:
                    print(f"    Theo trục thực tế: {orig_region['real_dimension_direction']}")
    
    # Hiển thị ảnh kết quả
    print(f"\n" + "="*60)
    print("HIỂN THỊ KẾT QUẢ")
    print("="*60)
    
    cv2.imshow("Original Image", frame)
    cv2.imshow("YOLO Detection", detections["annotated_frame"])
    
    print(f"\nNhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"\n=== HOÀN THÀNH TEST ===")

if __name__ == "__main__":
    test_fixed_module_division() 