"""
Test Pipeline với Robot Coordinates
Script đơn giản để test pipeline với chuyển đổi tọa độ camera sang robot
"""
import cv2
import time
from detection.pipeline import ProcessingPipeline
from detection.camera import CameraInterface
from detection.utils.tensorrt_yolo import YOLOTensorRT
from detection.utils.depth import DepthEstimator


def create_camera():
    """Factory function để tạo camera."""
    camera = CameraInterface(camera_index=0)
    camera.initialize()
    return camera


def create_yolo():
    """Factory function để tạo YOLO model."""
    return YOLOTensorRT("best.engine")


def create_depth():
    """Factory function để tạo depth estimator."""
    return DepthEstimator()


def test_pipeline_robot_coordinates():
    """Test pipeline với robot coordinates."""
    print("🚀 TESTING PIPELINE WITH ROBOT COORDINATES")
    print("="*60)
    
    # Tạo pipeline
    pipeline = ProcessingPipeline(
        camera_factory=create_camera,
        yolo_factory=create_yolo,
        depth_factory=create_depth,
        max_queue_size=3
    )
    
    # Khởi động pipeline
    print("🔄 Đang khởi động pipeline...")
    if not pipeline.start(timeout=60.0):
        print("❌ Không thể khởi động pipeline!")
        for error in pipeline.errors:
            print(f"   Lỗi: {error}")
        return
    
    print("✅ Pipeline đã khởi động thành công!")
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 50:  # Test với 50 frames
            # Lấy kết quả detection mới nhất
            detection_result = pipeline.get_latest_detection()
            
            if detection_result:
                frame, detections = detection_result
                frame_count += 1
                
                # Kiểm tra robot coordinates
                robot_coords = detections.get('robot_coordinates', [])
                
                if robot_coords:
                    print(f"\n📸 Frame {frame_count} - Phát hiện {len(robot_coords)} object(s):")
                    
                    for coord in robot_coords:
                        class_name = coord['class']
                        confidence = coord['confidence']
                        pixel = coord['camera_pixel']
                        robot_pos = coord['robot_coordinates']
                        cam_3d = coord.get('camera_3d')
                        
                        print(f"   🎯 {class_name} (conf: {confidence:.2f})")
                        print(f"      Pixel: ({pixel['x']}, {pixel['y']})")
                        print(f"      Robot: X={robot_pos['x']:8.2f}, Y={robot_pos['y']:8.2f}")
                        if cam_3d:
                            print(f"      Camera 3D: X={cam_3d['X']:8.3f}m, Y={cam_3d['Y']:8.3f}m, Z={cam_3d['Z']:8.3f}m")
                        
                        # Vẽ lên frame để hiển thị
                        center = (pixel['x'], pixel['y'])
                        bbox = coord.get('bbox', [])
                        
                        # Vẽ bounding box
                        if len(bbox) >= 4:
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        
                        # Vẽ center point
                        cv2.circle(frame, center, 8, (0, 0, 255), -1)
                        
                        # Vẽ text thông tin
                        text_lines = [
                            f"{class_name} ({confidence:.2f})",
                            f"Cam3D: X={cam_3d['X']:.2f}",
                            f"Y={cam_3d['Y']:.2f}, Z={cam_3d['Z']:.2f}"
                        ]
                        
                        y_offset = 0
                        for line in text_lines:
                            text_pos = (center[0] + 15, center[1] - 10 + y_offset)
                            cv2.putText(frame, line, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, (255, 255, 0), 1)
                            y_offset += 20
                
                # Hiển thị frame
                cv2.imshow("Pipeline Robot Coordinates Test", frame)
                
                # Thống kê
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    stats = pipeline.get_stats()
                    print(f"\n📊 Stats (Frame {frame_count}):")
                    print(f"   FPS: {fps:.1f}")
                    print(f"   Total frames: {stats['frames']}")
                    print(f"   Total detections: {stats['detections']}")
                    print(f"   Total depths: {stats['depths']}")
            
            # Kiểm tra phím nhấn
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Người dùng yêu cầu thoát")
                break
            elif key == ord(' '):
                print("⏸️  Tạm dừng, nhấn space để tiếp tục...")
                cv2.waitKey(0)
            
            time.sleep(0.1)  # Giảm tốc độ một chút
    
    except KeyboardInterrupt:
        print("\n⚠️  Đã nhận tín hiệu ngắt từ bàn phím")
    
    finally:
        # Dừng pipeline
        print("\n🔄 Đang dừng pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()
        
        # Thống kê cuối
        final_stats = pipeline.get_stats()
        total_time = time.time() - start_time
        
        print(f"\n📈 THỐNG KÊ CUỐI:")
        print(f"   Thời gian chạy: {total_time:.1f}s")
        print(f"   Frames đã test: {frame_count}")
        print(f"   FPS trung bình: {frame_count/total_time:.1f}")
        print(f"   Pipeline stats: {final_stats}")
        
        if pipeline.errors:
            print(f"\n❌ Lỗi trong quá trình chạy:")
            for error in pipeline.errors:
                print(f"   {error}")


if __name__ == "__main__":
    print("🤖 PIPELINE ROBOT COORDINATES TEST")
    print("Nhấn 'q' để thoát, 'space' để tạm dừng")
    print("="*60)
    
    try:
        test_pipeline_robot_coordinates()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc() 