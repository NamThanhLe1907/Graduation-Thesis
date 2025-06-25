"""
Script test camera calibration module.
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))

from detection.utils.camera_calibration import CameraCalibration
import numpy as np

def test_camera_calibration():
    """Test các chức năng của camera calibration."""
    print("=== TEST CAMERA CALIBRATION ===\n")
    
    # Test 1: Khởi tạo với file calibration thực
    print("1. Test khởi tạo với file camera_params.npz:")
    calib = CameraCalibration("camera_params.npz")
    print(f"   Calibration loaded: {calib.calibration_loaded}")
    print(f"   Camera matrix:\n{calib.camera_matrix}")
    print(f"   Distortion coefficients: {calib.dist_coeffs.flatten()}")
    
    # Test 2: Chuyển đổi pixel sang 3D
    print("\n2. Test chuyển đổi pixel sang tọa độ 3D:")
    test_cases = [
        # [pixel_x, pixel_y, depth_m]
        [640, 512, 2.0],   # Center của ảnh 1280x1024
        [100, 100, 1.5],   # Góc trên trái
        [1180, 924, 3.0],  # Góc dưới phải
        [400, 300, 1.0],   # Điểm khác
    ]
    
    for pixel_x, pixel_y, depth in test_cases:
        X, Y, Z = calib.pixel_to_3d(pixel_x, pixel_y, depth)
        print(f"   Pixel ({pixel_x:4.0f}, {pixel_y:4.0f}) depth {depth}m -> 3D ({X:6.3f}, {Y:6.3f}, {Z:6.3f})m")
        
        # Test chiếu ngược
        proj_x, proj_y = calib.project_3d_to_pixel(X, Y, Z)
        error_x = abs(proj_x - pixel_x)
        error_y = abs(proj_y - pixel_y)
        print(f"     Chiếu ngược: 3D -> Pixel ({proj_x:6.1f}, {proj_y:6.1f}) | Error: ({error_x:.2f}, {error_y:.2f})")
    
    # Test 3: Ước tính kích thước thực
    print("\n3. Test ước tính kích thước thực từ bounding box:")
    test_bboxes = [
        # [x1, y1, x2, y2, depth]
        [100, 100, 300, 250, 2.0],   # bbox 200x150 pixels tại 2m
        [400, 200, 600, 400, 1.5],   # bbox 200x200 pixels tại 1.5m
        [500, 300, 800, 500, 3.0],   # bbox 300x200 pixels tại 3m
    ]
    
    for x1, y1, x2, y2, depth in test_bboxes:
        bbox = [x1, y1, x2, y2]
        pixel_width = x2 - x1
        pixel_height = y2 - y1
        
        real_size = calib.estimate_real_size(bbox, depth)
        print(f"   BBox {pixel_width}x{pixel_height}px tại {depth}m:")
        print(f"     Kích thước thực: {real_size['width_m']:.3f}m x {real_size['height_m']:.3f}m")
        print(f"     Diện tích: {real_size['area_m2']:.3f}m²")
    
    # Test 4: Thông tin camera
    print("\n4. Thông tin camera:")
    info = calib.get_camera_info()
    print(f"   Focal length: fx={info['focal_length']['fx']:.2f}, fy={info['focal_length']['fy']:.2f}")
    print(f"   Principal point: cx={info['principal_point']['cx']:.2f}, cy={info['principal_point']['cy']:.2f}")
    print(f"   Calibration từ file: {info['calibration_loaded']}")
    
    # Test 5: Test với camera matrix mặc định (khi không có file calibration)
    print("\n5. Test với camera matrix mặc định:")
    calib_default = CameraCalibration("non_existent_file.npz")
    print(f"   Default calibration loaded: {calib_default.calibration_loaded}")
    
    # So sánh kết quả
    test_pixel_x, test_pixel_y, test_depth = 640, 512, 2.0
    X1, Y1, Z1 = calib.pixel_to_3d(test_pixel_x, test_pixel_y, test_depth)
    X2, Y2, Z2 = calib_default.pixel_to_3d(test_pixel_x, test_pixel_y, test_depth)
    
    print(f"   Calibrated result: ({X1:.3f}, {Y1:.3f}, {Z1:.3f})m")
    print(f"   Default result:    ({X2:.3f}, {Y2:.3f}, {Z2:.3f})m")
    print(f"   Difference:        ({abs(X1-X2):.3f}, {abs(Y1-Y2):.3f}, {abs(Z1-Z2):.3f})m")
    
    print("\n=== TEST HOÀN THÀNH ===")

def test_depth_integration():
    """Test tích hợp với DepthEstimator."""
    print("\n=== TEST TÍCH HỢP VỚI DEPTH ESTIMATOR ===\n")
    
    try:
        from detection.utils.depth import DepthEstimator
        
        # Test với camera calibration
        print("1. Khởi tạo DepthEstimator với camera calibration:")
        depth_estimator = DepthEstimator(
            enable=False,  # Tắt để test nhanh
            use_camera_calibration=True,
            camera_calibration_file="camera_params.npz"
        )
        
        has_calibration = depth_estimator.camera_calibration is not None
        print(f"   Camera calibration: {'Có' if has_calibration else 'Không'}")
        
        if has_calibration:
            print(f"   Focal length: fx={depth_estimator.camera_calibration.fx:.2f}, fy={depth_estimator.camera_calibration.fy:.2f}")
            print(f"   Principal point: cx={depth_estimator.camera_calibration.cx:.2f}, cy={depth_estimator.camera_calibration.cy:.2f}")
        
        # Test không có camera calibration
        print("\n2. Khởi tạo DepthEstimator không có camera calibration:")
        depth_estimator_no_calib = DepthEstimator(
            enable=False,  # Tắt để test nhanh
            use_camera_calibration=False
        )
        
        has_calibration_2 = depth_estimator_no_calib.camera_calibration is not None
        print(f"   Camera calibration: {'Có' if has_calibration_2 else 'Không'}")
        
        print("\n   Test thành công!")
        
    except ImportError as e:
        print(f"   Không thể import DepthEstimator: {e}")
    except Exception as e:
        print(f"   Lỗi trong test: {e}")

if __name__ == "__main__":
    test_camera_calibration()
    test_depth_integration()