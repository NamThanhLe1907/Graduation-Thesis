"""
Test file để kiểm tra camera brightness với optimized settings đã được fix.
Bây giờ optimized settings sẽ KHÔNG làm camera tối nữa.
"""
import cv2
from detection.camera import CameraInterface
import time

def test_camera_brightness_fixed():
    """Test camera brightness với optimized settings đã fix."""
    
    print("=== TEST CAMERA BRIGHTNESS - OPTIMIZED SETTINGS FIXED ===")
    
    # Test 1: Camera với optimized settings = False (mặc định) 
    print("\n--- TEST 1: Standard Settings ---")
    camera_standard = CameraInterface(use_optimized_settings=False)
    
    try:
        camera_standard.initialize()
        print("📹 Camera khởi tạo với Standard Settings")
        print("Nhấn 'q' để chuyển sang test tiếp theo")
        
        while True:
            frame = camera_standard.get_frame()
            cv2.putText(frame, "Test 1: Standard Settings", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "use_optimized_settings = False", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to continue", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Camera Brightness Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        camera_standard.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Lỗi Test 1: {e}")
        camera_standard.release()
    
    # Test 2: Camera với optimized settings = True (đã fix)
    print("\n--- TEST 2: Optimized Settings (FIXED) ---")
    camera_optimized = CameraInterface(use_optimized_settings=True)
    
    try:
        camera_optimized.initialize()
        print("⚡ Camera khởi tạo với Optimized Settings (đã fix)")
        print("Camera sẽ KHÔNG còn tối như trước nữa!")
        print("Nhấn 'q' để chuyển sang test tiếp theo")
        
        while True:
            frame = camera_optimized.get_frame()
            cv2.putText(frame, "Test 2: Optimized Settings (FIXED)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "use_optimized_settings = True", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Auto exposure ENABLED (0.75)", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to continue", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Camera Brightness Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
    except Exception as e:
        print(f"❌ Lỗi Test 2-3: {e}")
    
    finally:
        camera_optimized.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_brightness_fixed() 