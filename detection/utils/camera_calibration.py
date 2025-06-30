"""
Module xử lý camera calibration và coordinate transformation.
Sử dụng camera intrinsic matrix và distortion coefficients để cải thiện độ chính xác depth estimation.
"""
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Any
import os


class CameraCalibration:
    """Xử lý camera calibration và coordinate transformation."""
    
    def __init__(self, calib_file: str = "camera_params.npz"):
        """
        Khởi tạo với file calibration.
        
        Args:
            calib_file: Đường dẫn tới file .npz chứa mtx và dist
        """
        # Kiểm tra file calibration
        if not os.path.exists(calib_file):
            # Thử tìm trong thư mục gốc của project
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            calib_file_alt = os.path.join(project_root, os.path.basename(calib_file))
            if os.path.exists(calib_file_alt):
                calib_file = calib_file_alt
            else:
                print(f"[CameraCalibration] CẢNH BÁO: Không tìm thấy file calibration: {calib_file}")
                print(f"[CameraCalibration] Sẽ sử dụng camera matrix mặc định (ước tính)")
                self._use_default_calibration()
                return
        
        try:
            # Load calibration data
            calib_data = np.load(calib_file)
            self.camera_matrix = calib_data['mtx']
            self.dist_coeffs = calib_data['dist']
            
            # Extract camera parameters
            self.fx = self.camera_matrix[0, 0]  # Focal length x
            self.fy = self.camera_matrix[1, 1]  # Focal length y
            self.cx = self.camera_matrix[0, 2]  # Principal point x
            self.cy = self.camera_matrix[1, 2]  # Principal point y
            
            self.calibration_loaded = True
            
            print(f"[CameraCalibration] Camera calibration đã được tải từ: {calib_file}")
            print(f"[CameraCalibration] Camera intrinsics:")
            print(f"  fx: {self.fx:.2f}, fy: {self.fy:.2f}")
            print(f"  cx: {self.cx:.2f}, cy: {self.cy:.2f}")
            print(f"  Distortion coefficients: {self.dist_coeffs.flatten()}")
            
        except Exception as e:
            print(f"[CameraCalibration] Lỗi khi tải calibration: {e}")
            print(f"[CameraCalibration] Sẽ sử dụng camera matrix mặc định")
            self._use_default_calibration()
    
    def _use_default_calibration(self):
        """Sử dụng camera matrix mặc định khi không có file calibration."""
        # Camera matrix mặc định cho camera có độ phân giải 1280x1024
        # Đây là ước tính dựa trên camera thông thường
        self.fx = 1000.0  # Focal length ước tính
        self.fy = 1000.0
        self.cx = 640.0   # Principal point ở giữa ảnh (1280/2)
        self.cy = 512.0   # Principal point ở giữa ảnh (1024/2)
        
        self.camera_matrix = np.array([
            [self.fx, 0,      self.cx],
            [0,       self.fy, self.cy],
            [0,       0,      1      ]
        ], dtype=np.float64)
        
        # Không có distortion correction
        self.dist_coeffs = np.zeros((1, 5), dtype=np.float64)
        
        self.calibration_loaded = False
        
        print(f"[CameraCalibration] Sử dụng camera matrix mặc định:")
        print(f"  fx: {self.fx:.2f}, fy: {self.fy:.2f}")
        print(f"  cx: {self.cx:.2f}, cy: {self.cy:.2f}")
        print(f"  Distortion: None")
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Hiệu chỉnh méo ảnh.
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Ảnh đã được hiệu chỉnh méo (hoặc ảnh gốc nếu không có calibration)
        """
        if not self.calibration_loaded:
            # Không có calibration thực tế, trả về ảnh gốc
            return image
        
        try:
            return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        except Exception as e:
            print(f"[CameraCalibration] Lỗi khi undistort ảnh: {e}")
            return image
    
    def pixel_to_3d(self, pixel_x: float, pixel_y: float, depth: float) -> Tuple[float, float, float]:
        """
        Chuyển đổi từ tọa độ pixel sang tọa độ 3D trong hệ tọa độ camera.
        
        Args:
            pixel_x: Tọa độ x trong ảnh (pixel)
            pixel_y: Tọa độ y trong ảnh (pixel)  
            depth: Độ sâu (mét)
            
        Returns:
            Tuple (X, Y, Z) trong hệ tọa độ camera (mét)
        """
        # Chuyển từ pixel coordinates sang normalized coordinates
        x_norm = (pixel_x - self.cx) / self.fx
        y_norm = (pixel_y - self.cy) / self.fy
        
        # Tính tọa độ 3D trong hệ tọa độ camera
        X = x_norm * depth
        Y = y_norm * depth
        Z = depth
        
        return X, Y, Z
    
    def pixel_to_3d_batch(self, pixel_coords: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi batch các tọa độ pixel sang 3D.
        
        Args:
            pixel_coords: Array shape (N, 2) chứa [pixel_x, pixel_y]
            depths: Array shape (N,) chứa độ sâu
            
        Returns:
            Array shape (N, 3) chứa [X, Y, Z] trong hệ tọa độ camera
        """
        # Chuyển sang normalized coordinates
        x_norm = (pixel_coords[:, 0] - self.cx) / self.fx
        y_norm = (pixel_coords[:, 1] - self.cy) / self.fy
        
        # Tính tọa độ 3D
        X = x_norm * depths
        Y = y_norm * depths
        Z = depths
        
        return np.column_stack([X, Y, Z])
    
    def get_3d_points_from_regions(self, depth_results: List[dict]) -> List[dict]:
        """
        Tính tọa độ 3D cho các vùng được phát hiện.
        
        Args:
            depth_results: Kết quả từ depth estimation với center và depth
            
        Returns:
            Danh sách kết quả với thêm tọa độ 3D
        """
        results_3d = []
        
        for result in depth_results:
            center = result.get('center', [0, 0])
            depth_info = result.get('depth', {})
            
            # Lấy mean depth
            if isinstance(depth_info, dict):
                mean_depth = depth_info.get('mean_depth', 0.0)
                min_depth = depth_info.get('min_depth', 0.0)
                max_depth = depth_info.get('max_depth', 0.0)
            else:
                mean_depth = min_depth = max_depth = 0.0
            
            if mean_depth > 0:
                # Chuyển đổi center pixel sang 3D
                X, Y, Z = self.pixel_to_3d(center[0], center[1], mean_depth)
                
                # Thêm thông tin 3D vào kết quả
                result_3d = result.copy()
                result_3d['position_3d_camera'] = {
                    'X': X,  # Camera coordinate X (mét)
                    'Y': Y,  # Camera coordinate Y (mét) 
                    'Z': Z,  # Camera coordinate Z (mét)
                    'pixel_x': center[0],
                    'pixel_y': center[1],
                    'depth': mean_depth
                }
                
                # Cập nhật position hiện có với tọa độ 3D chính xác hơn
                if 'position' in result_3d:
                    result_3d['position'].update({
                        'x_3d': X,
                        'y_3d': Y,
                        'z_3d': Z
                    })
                
                results_3d.append(result_3d)
            else:
                # Vẫn thêm vào nhưng không có thông tin 3D
                result_3d = result.copy()
                result_3d['position_3d_camera'] = {
                    'X': 0.0, 'Y': 0.0, 'Z': 0.0,
                    'pixel_x': center[0], 'pixel_y': center[1], 'depth': 0.0
                }
                results_3d.append(result_3d)
            
        return results_3d
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin camera calibration.
        
        Returns:
            Dict chứa thông tin camera
        """
        return {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist(),
            'focal_length': {'fx': self.fx, 'fy': self.fy},
            'principal_point': {'cx': self.cx, 'cy': self.cy},
            'calibration_loaded': self.calibration_loaded
        }
    
    def project_3d_to_pixel(self, X: float, Y: float, Z: float) -> Tuple[float, float]:
        """
        Chiếu điểm 3D trong hệ tọa độ camera về pixel coordinates.
        
        Args:
            X, Y, Z: Tọa độ 3D trong hệ tọa độ camera (mét)
            
        Returns:
            Tuple (pixel_x, pixel_y)
        """
        if Z <= 0:
            return 0.0, 0.0
        
        # Chiếu về normalized coordinates
        x_norm = X / Z
        y_norm = Y / Z
        
        # Chuyển về pixel coordinates
        pixel_x = x_norm * self.fx + self.cx
        pixel_y = y_norm * self.fy + self.cy
        
        return pixel_x, pixel_y
    
    def estimate_real_size(self, bbox_pixels: List[float], depth: float) -> Dict[str, float]:
        """
        Ước tính kích thước thực của object từ bounding box và depth.
        
        Args:
            bbox_pixels: [x1, y1, x2, y2] trong pixel
            depth: Độ sâu trung bình (mét)
            
        Returns:
            Dict chứa width_m, height_m (kích thước thực tế tính bằng mét)
        """
        if depth <= 0:
            return {'width_m': 0.0, 'height_m': 0.0}
        
        x1, y1, x2, y2 = bbox_pixels
        
        # Chuyển các góc bounding box sang 3D
        top_left_3d = self.pixel_to_3d(x1, y1, depth)
        bottom_right_3d = self.pixel_to_3d(x2, y2, depth)
        
        # Tính kích thước thực
        width_m = abs(bottom_right_3d[0] - top_left_3d[0])
        height_m = abs(bottom_right_3d[1] - top_left_3d[1])
        
        return {
            'width_m': width_m,
            'height_m': height_m,
            'area_m2': width_m * height_m
        }


# Để test module
if __name__ == "__main__":
    print("=== TEST CAMERA CALIBRATION ===")
    
    # Khởi tạo camera calibration
    calib = CameraCalibration()
    
    # Test chuyển đổi pixel sang 3D
    print("\n--- TEST PIXEL TO 3D ---")
    test_pixels = [
        [640, 512, 2.0],  # Center của ảnh 1280x1024, depth 2m
        [100, 100, 1.5],  # Góc trên trái, depth 1.5m
        [1180, 924, 3.0], # Góc dưới phải, depth 3m
    ]
    
    for pixel_x, pixel_y, depth in test_pixels:
        X, Y, Z = calib.pixel_to_3d(pixel_x, pixel_y, depth)
        print(f"Pixel ({pixel_x}, {pixel_y}) với depth {depth}m -> 3D ({X:.3f}, {Y:.3f}, {Z:.3f})m")
        
        # Test chiếu ngược
        proj_x, proj_y = calib.project_3d_to_pixel(X, Y, Z)
        print(f"  Chiếu ngược: 3D ({X:.3f}, {Y:.3f}, {Z:.3f})m -> Pixel ({proj_x:.1f}, {proj_y:.1f})")
    
    # Test ước tính kích thước thực
    print("\n--- TEST REAL SIZE ESTIMATION ---")
    test_bbox = [100, 100, 300, 250]  # bbox 200x150 pixels
    test_depth = 2.0  # 2 mét
    
    real_size = calib.estimate_real_size(test_bbox, test_depth)
    print(f"BBox {test_bbox} với depth {test_depth}m:")
    print(f"  Kích thước thực: {real_size['width_m']:.3f}m x {real_size['height_m']:.3f}m")
    print(f"  Diện tích: {real_size['area_m2']:.3f}m²")
    
    # Hiển thị thông tin camera
    print("\n--- CAMERA INFO ---")
    info = calib.get_camera_info()
    print(f"Calibration loaded: {info['calibration_loaded']}")
    print(f"Focal length: fx={info['focal_length']['fx']:.2f}, fy={info['focal_length']['fy']:.2f}")
    print(f"Principal point: cx={info['principal_point']['cx']:.2f}, cy={info['principal_point']['cy']:.2f}") 