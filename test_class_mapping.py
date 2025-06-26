"""
Test script để kiểm tra class mapping hoạt động đúng.
"""
import cv2
import os
from detection import YOLOTensorRT
from detection.utils.rotation_analyzer import RotationAnalyzer

def test_class_mapping():
    """Test xem class mapping có hoạt động không"""
    
    # Khởi tạo model
    ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "utils", "detection", "best.engine")
    
    try:
        yolo_model = YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.25)
        analyzer = RotationAnalyzer(debug=True)
        
        # Test với ảnh mẫu
        test_image_path = "images_pallets"
        if os.path.exists(test_image_path):
            image_files = [f for f in os.listdir(test_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                image_path = os.path.join(test_image_path, image_files[15])
                
                frame = cv2.imread(image_path)
                if frame is not None:
                    print(f"Testing với ảnh: {image_path}")
                    
                    # Thực hiện detection
                    detections = yolo_model.detect(frame)
                    
                    print(f"\n=== KẾT QUẢ DETECTION ===")
                    print(f"Số objects: {len(detections.get('bounding_boxes', []))}")
                    print(f"Class IDs: {detections.get('classes', [])}")
                    print(f"Class Names: {detections.get('class_names', [])}")
                    print(f"Confidences: {detections.get('confidences', [])}")
                    
                    # Test rotation analyzer
                    if detections.get('class_names'):
                        yolo_angles = analyzer.analyze_yolo_angles(detections)
                        
                        print(f"\n=== FILTER TEST ===")
                        load_objects = [obj for obj in yolo_angles if 'load' in obj['class_name'].lower()]
                        pallet_objects = [obj for obj in yolo_angles if 'pallet' in obj['class_name'].lower()]
                        
                        print(f"Load objects found: {len(load_objects)}")
                        print(f"Pallet objects found: {len(pallet_objects)}")
                        
                        for obj in yolo_angles:
                            print(f"  Object: {obj['class_name']} (confidence: {obj['confidence']:.2f})")
                    else:
                        print("❌ Không có class_names!")
                else:
                    print("❌ Không thể đọc ảnh")
            else:
                print("❌ Không có ảnh trong folder")
        else:
            print("❌ Folder images_pallets2 không tồn tại")
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    test_class_mapping() 