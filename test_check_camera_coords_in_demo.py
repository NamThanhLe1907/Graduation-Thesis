"""
Script kiểm tra robot coordinates trong demo camera
"""
import cv2
import time
from detection.pipeline import ProcessingPipeline
from detection.camera import CameraInterface  
from detection.utils.tensorrt_yolo import YOLOTensorRT
from detection.utils.depth import DepthEstimator
import os

# Factory functions (copy từ use_tensorrt_example.py)
def create_camera():
    camera = CameraInterface(camera_index=0)
    camera.initialize()
    return camera

def create_yolo():
    ENGINE_PATH = "best.engine"  # Adjust path as needed
    return YOLOTensorRT(engine_path=ENGINE_PATH, conf=0.55)

def create_depth():
    return DepthEstimator(device='cpu', enable=False)  # Tắt depth để test nhanh

def test_robot_coordinates_in_pipeline():
    """Test robot coordinates từ pipeline."""
    print("🔍 TESTING ROBOT COORDINATES IN PIPELINE")
    print("="*60)
    
    # Tạo pipeline
    pipeline = ProcessingPipeline(
        camera_factory=create_camera,
        yolo_factory=create_yolo, 
        depth_factory=create_depth,
        max_queue_size=3
    )
    
    # Khởi động pipeline
    if not pipeline.start(timeout=60.0):
        print("❌ Không thể khởi động pipeline!")
        return
    
    print("✅ Pipeline started successfully!")
    print("📱 Đợi phát hiện objects để xem robot coordinates...")
    
    try:
        frame_count = 0
        while frame_count < 20:  # Test 20 frames
            # Lấy kết quả detection
            detection_result = pipeline.get_latest_detection()
            
            if detection_result:
                frame, detections = detection_result
                frame_count += 1
                
                print(f"\n📸 Frame {frame_count}:")
                
                # ⭐ KIỂM TRA ROBOT COORDINATES ⭐
                robot_coords = detections.get('robot_coordinates', [])
                
                if robot_coords:
                    print(f"   ✅ CÓ {len(robot_coords)} robot coordinates:")
                    
                    for i, coord in enumerate(robot_coords, 1):
                        class_name = coord['class']
                        confidence = coord['confidence']
                        pixel = coord['camera_pixel']
                        robot_pos = coord['robot_coordinates']
                        cam_3d = coord.get('camera_3d')
                        
                        print(f"      {i}. {class_name} (conf: {confidence:.2f})")
                        print(f"         Pixel: ({pixel['x']}, {pixel['y']})")
                        print(f"         Robot: X={robot_pos['x']:8.2f}, Y={robot_pos['y']:8.2f}")
                        if cam_3d:
                            print(f"         Camera 3D: X={cam_3d['X']:8.3f}, Y={cam_3d['Y']:8.3f}, Z={cam_3d['Z']:8.3f}")
                else:
                    print("   ❌ KHÔNG có robot coordinates")
                
                # Kiểm tra các data khác
                print(f"   📊 Other data:")
                print(f"      - Bounding boxes: {len(detections.get('bounding_boxes', []))}")
                print(f"      - Classes: {detections.get('classes', [])}")
                print(f"      - Theta4 result: {'CÓ' if detections.get('theta4_result') else 'KHÔNG'}")
                print(f"      - Divided result: {'CÓ' if detections.get('divided_result') else 'KHÔNG'}")
                
                # Hiển thị frame
                cv2.imshow("Test Robot Coordinates", detections["annotated_frame"])
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 User requested exit")
                    break
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n✅ Test completed!")

if __name__ == "__main__":
    test_robot_coordinates_in_pipeline() 