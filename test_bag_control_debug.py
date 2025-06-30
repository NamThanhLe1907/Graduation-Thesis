"""
Test script để debug bag control key mapping issues
"""
import os
import cv2
import numpy as np

def test_environment_variables():
    """Test 1: Kiểm tra environment variables"""
    print("🔍 [TEST 1] Environment Variables:")
    
    enable_plc_raw = os.environ.get('ENABLE_PLC', 'true')
    enable_plc = enable_plc_raw.lower() in ('true', '1', 'yes')
    plc_ip = os.environ.get('PLC_IP', '192.168.0.1')
    
    print(f"   ENABLE_PLC (raw): '{enable_plc_raw}'")
    print(f"   ENABLE_PLC (processed): {enable_plc}")
    print(f"   PLC_IP: {plc_ip}")
    
    return enable_plc, plc_ip

def test_key_detection():
    """Test 2: Kiểm tra key detection"""
    print("\n🔍 [TEST 2] Key Detection Test:")
    print("   Tạo cửa sổ test, nhấn các phím 1, 2, 3, q để test...")
    
    # Tạo dummy image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Key Detection Test", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(test_image, "Press 1, 2, 3 or q to test", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(test_image, "Current: waiting...", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Bag Control Key Test", test_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key != 255:  # Key was pressed
            key_char = chr(key) if 32 <= key <= 126 else f"ASCII:{key}"
            print(f"   ✅ Key detected: {key_char} (code: {key})")
            
            # Update display
            test_image_copy = test_image.copy()
            cv2.putText(test_image_copy, f"Current: {key_char} (code: {key})", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Bag Control Key Test", test_image_copy)
            
            if key == ord('q'):
                print("   ✅ Q detected - exiting test")
                break
            elif key == ord('1'):
                print("   ✅ Key 1 detected - this should work for bag control!")
            elif key == ord('2'):
                print("   ✅ Key 2 detected - this should work for bag control!")
            elif key == ord('3'):
                print("   ✅ Key 3 detected - this should work for bag control!")
    
    cv2.destroyAllWindows()

def test_plc_integration_import():
    """Test 3: Kiểm tra import và tạo PLC integration"""
    print("\n🔍 [TEST 3] PLC Integration Import Test:")
    
    try:
        from region_division_plc_integration import RegionDivisionPLCIntegration
        print("   ✅ RegionDivisionPLCIntegration import successful")
        
        # Try to create instance
        plc_integration = RegionDivisionPLCIntegration(
            plc_ip="192.168.0.1", 
            debug=True
        )
        print("   ✅ RegionDivisionPLCIntegration instance created")
        
        # Test bag control methods
        try:
            plc_integration.set_current_bag_number(1)
            bag_info = plc_integration.get_current_bag_info()
            print(f"   ✅ Bag control methods work: {bag_info['sequence_mapping']}")
            return True
        except Exception as e:
            print(f"   ❌ Bag control methods failed: {e}")
            return False
            
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Instance creation failed: {e}")
        return False

def test_pipeline_creation():
    """Test 4: Kiểm tra tạo pipeline với PLC"""
    print("\n🔍 [TEST 4] Pipeline Creation Test:")
    
    try:
        # Import các factory functions từ camera_standalone
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from camera_standalone import create_camera, create_yolo, create_depth
        from detection import ProcessingPipeline
        
        print("   ✅ Factory functions imported successfully")
        
        # Try to create pipeline
        enable_plc = True
        plc_ip = "192.168.0.1"
        
        pipeline = ProcessingPipeline(
            camera_factory=create_camera,
            yolo_factory=create_yolo,
            depth_factory=create_depth,
            enable_plc=enable_plc,
            plc_ip=plc_ip
        )
        
        print("   ✅ Pipeline created successfully")
        
        # Test get_plc_integration
        plc_integration = pipeline.get_plc_integration()
        print(f"   🔍 Pipeline PLC integration: {plc_integration}")
        
        if plc_integration:
            print("   ✅ Pipeline has PLC integration")
            return True
        else:
            print("   ❌ Pipeline PLC integration is None")
            return False
            
    except Exception as e:
        print(f"   ❌ Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Chạy tất cả các test"""
    print("🚀 BAG CONTROL DEBUG TEST SUITE")
    print("="*50)
    
    # Test 1: Environment variables
    enable_plc, plc_ip = test_environment_variables()
    
    # Test 2: Key detection
    test_key_detection()
    
    # Test 3: PLC integration import
    plc_import_ok = test_plc_integration_import()
    
    # Test 4: Pipeline creation
    pipeline_ok = test_pipeline_creation()
    
    # Summary
    print("\n📊 [SUMMARY] Test Results:")
    print(f"   Environment PLC enabled: {enable_plc}")
    print(f"   PLC Integration import: {'✅' if plc_import_ok else '❌'}")
    print(f"   Pipeline creation: {'✅' if pipeline_ok else '❌'}")
    
    if enable_plc and plc_import_ok and pipeline_ok:
        print("\n🎉 [CONCLUSION] All tests passed!")
        print("   Bag control keys (1, 2, 3) should work in camera_standalone.py")
        print("\n💡 [NEXT STEPS]:")
        print("   1. Run camera_standalone.py")
        print("   2. Wait for pipeline to start")
        print("   3. Focus on the OpenCV window")
        print("   4. Press keys 1, 2, or 3")
        print("   5. Check console output for debug messages")
    else:
        print("\n❌ [CONCLUSION] Some tests failed!")
        print("   Bag control keys may not work properly")
        print("\n🔧 [TROUBLESHOOTING]:")
        if not enable_plc:
            print("   - Set ENABLE_PLC=true environment variable")
        if not plc_import_ok:
            print("   - Check region_division_plc_integration.py exists and works")
        if not pipeline_ok:
            print("   - Check pipeline initialization in camera_standalone.py")

if __name__ == "__main__":
    main() 