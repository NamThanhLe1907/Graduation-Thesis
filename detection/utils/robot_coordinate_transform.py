"""
Module chuyển đổi tọa độ từ camera sang robot coordinates.
Sử dụng homography transformation dựa trên 4 điểm calibration.images_pallets/image_11.jpg
"""
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, Union
import json
import os


class RobotCoordinateTransform:
    """Chuyển đổi tọa độ từ camera sang robot coordinates."""
    
    def __init__(self, calibration_points: Optional[List[Dict]] = None, y_offset: float = 0.0, x_offset: float = 40.0):
        """
        Khởi tạo với các điểm calibration.
        
        Args:
            calibration_points: List các dict chứa {'camera': (x, y), 'robot': (x, y)}
                               Nếu None, sẽ sử dụng điểm mặc định được cung cấp
            y_offset: Offset cho tọa độ Y robot (cm). -4.0 để fix lệch +4cm
            x_offset: Offset cho tọa độ X robot (cm). -40.0 để fix lệch +40cm
        """
        # Offset correction để fix lệch tọa độ
        self.y_offset = y_offset  # Y offset (cm)
        self.x_offset = x_offset  # X offset (cm)
        
        if calibration_points is None:
            # Sử dụng 4 điểm calibration được cung cấp
            self.calibration_points = [
                {'camera': (76, 32), 'robot': (312.9774, 17)},
                {'camera': (75, 665), 'robot': (312.9774, 343.7173)},
                {'camera': (1171, 667), 'robot': (-190, 343.7173)},
                {'camera': (1172, 34), 'robot': (-190, 17)}
            ]
        else:
            self.calibration_points = calibration_points
        
        # Tính transformation matrix
        self.transformation_matrix = None
        self.inverse_transformation_matrix = None
        self._compute_transformation_matrix()
        
        print(f"[RobotCoordinateTransform] Khởi tạo với {len(self.calibration_points)} điểm calibration")
        print(f"[RobotCoordinateTransform] X offset correction: {self.x_offset}cm")
        print(f"[RobotCoordinateTransform] Y offset correction: {self.y_offset}cm")
        self._print_calibration_info()
    
    def _compute_transformation_matrix(self):
        """Tính toán transformation matrix từ các điểm calibration."""
        if len(self.calibration_points) < 4:
            raise ValueError("Cần ít nhất 4 điểm để tính homography transformation")
        
        # Chuẩn bị arrays cho cv2.getPerspectiveTransform
        camera_points = np.array([point['camera'] for point in self.calibration_points], dtype=np.float32)
        robot_points = np.array([point['robot'] for point in self.calibration_points], dtype=np.float32)
        
        # Tính homography matrix từ camera sang robot
        self.transformation_matrix = cv2.getPerspectiveTransform(camera_points, robot_points)
        
        # Tính inverse matrix từ robot sang camera
        self.inverse_transformation_matrix = cv2.getPerspectiveTransform(robot_points, camera_points)
        
        print("[RobotCoordinateTransform] Transformation matrix đã được tính toán")
    
    def _print_calibration_info(self):
        """In thông tin calibration points."""
        print("\n=== ĐIỂM CALIBRATION ===")
        for i, point in enumerate(self.calibration_points, 1):
            camera_coord = point['camera']
            robot_coord = point['robot']
            print(f"Điểm {i}: Camera({camera_coord[0]}, {camera_coord[1]}) -> Robot({robot_coord[0]}, {robot_coord[1]})")
        print("=" * 30)
    
    def camera_to_robot(self, camera_x: float, camera_y: float) -> Tuple[float, float]:
        """
        Chuyển đổi từ tọa độ camera sang tọa độ robot.
        
        Args:
            camera_x: Tọa độ x trong camera (pixel)
            camera_y: Tọa độ y trong camera (pixel)
            
        Returns:
            Tuple (robot_x, robot_y) trong hệ tọa độ robot (đã apply X,Y offset correction)
        """
        if self.transformation_matrix is None:
            raise RuntimeError("Transformation matrix chưa được tính toán")
        
        # Tạo homogeneous coordinates
        camera_point = np.array([[camera_x, camera_y]], dtype=np.float32).reshape(1, 1, 2)
        
        # Áp dụng perspective transformation
        robot_point = cv2.perspectiveTransform(camera_point, self.transformation_matrix)
        
        # Apply X và Y offset correction
        robot_x = float(robot_point[0][0][0]) + self.x_offset
        robot_y = float(robot_point[0][0][1]) + self.y_offset
        
        return robot_x, robot_y
    
    def robot_to_camera(self, robot_x: float, robot_y: float) -> Tuple[float, float]:
        """
        Chuyển đổi từ tọa độ robot sang tọa độ camera.
        
        Args:
            robot_x: Tọa độ x trong robot
            robot_y: Tọa độ y trong robot
            
        Returns:
            Tuple (camera_x, camera_y) trong pixel coordinates
        """
        if self.inverse_transformation_matrix is None:
            raise RuntimeError("Inverse transformation matrix chưa được tính toán")
        
        # Apply X và Y offset correction ngược trước khi chuyển đổi
        adjusted_robot_x = robot_x - self.x_offset
        adjusted_robot_y = robot_y - self.y_offset
        
        # Tạo homogeneous coordinates
        robot_point = np.array([[adjusted_robot_x, adjusted_robot_y]], dtype=np.float32).reshape(1, 1, 2)
        
        # Áp dụng inverse perspective transformation
        camera_point = cv2.perspectiveTransform(robot_point, self.inverse_transformation_matrix)
        
        return float(camera_point[0][0][0]), float(camera_point[0][0][1])
    
    def camera_to_robot_batch(self, camera_points: Union[List[Tuple], np.ndarray]) -> np.ndarray:
        """
        Chuyển đổi batch các điểm từ camera sang robot coordinates.
        
        Args:
            camera_points: List hoặc array các điểm [(x1, y1), (x2, y2), ...]
            
        Returns:
            Array shape (N, 2) chứa tọa độ robot (đã apply X,Y offset correction)
        """
        if self.transformation_matrix is None:
            raise RuntimeError("Transformation matrix chưa được tính toán")
        
        # Chuyển về định dạng numpy array
        if isinstance(camera_points, list):
            camera_points = np.array(camera_points, dtype=np.float32)
        
        # Reshape cho cv2.perspectiveTransform
        points_reshaped = camera_points.reshape(-1, 1, 2)
        
        # Áp dụng transformation
        robot_points = cv2.perspectiveTransform(points_reshaped, self.transformation_matrix)
        robot_points = robot_points.reshape(-1, 2)
        
        # Apply X và Y offset correction cho tất cả các điểm
        robot_points[:, 0] += self.x_offset  # X offset
        robot_points[:, 1] += self.y_offset  # Y offset
        
        return robot_points
    
    def transform_detection_results(self, detection_results: List[Dict]) -> List[Dict]:
        """
        Chuyển đổi kết quả detection từ camera coords sang robot coords.
        
        Args:
            detection_results: List các dict chứa thông tin detection
            
        Returns:
            List các dict với tọa độ robot đã được thêm vào
        """
        transformed_results = []
        
        for result in detection_results:
            result_copy = result.copy()
            
            # Chuyển đổi center point nếu có
            if 'center' in result:
                center_cam = result['center']
                center_robot = self.camera_to_robot(center_cam[0], center_cam[1])
                result_copy['center_robot'] = center_robot
            
            # Chuyển đổi bounding box nếu có
            if 'bbox' in result:
                bbox = result['bbox']  # [x1, y1, x2, y2]
                
                # Chuyển đổi 2 góc của bbox
                top_left_robot = self.camera_to_robot(bbox[0], bbox[1])
                bottom_right_robot = self.camera_to_robot(bbox[2], bbox[3])
                
                result_copy['bbox_robot'] = [
                    top_left_robot[0], top_left_robot[1],
                    bottom_right_robot[0], bottom_right_robot[1]
                ]
            
            # Thêm thông tin robot coordinates vào position nếu có
            if 'position' in result_copy:
                if 'center_robot' in result_copy:
                    result_copy['position']['robot_x'] = center_robot[0]
                    result_copy['position']['robot_y'] = center_robot[1]
            else:
                # Tạo position mới với robot coordinates
                if 'center_robot' in result_copy:
                    result_copy['position'] = {
                        'robot_x': center_robot[0],
                        'robot_y': center_robot[1]
                    }
            
            transformed_results.append(result_copy)
        
        return transformed_results
    
    def validate_transformation(self) -> Dict[str, float]:
        """
        Kiểm tra độ chính xác của transformation bằng cách test với các điểm calibration.
        
        Returns:
            Dict chứa thông tin về độ chính xác
        """
        if self.transformation_matrix is None:
            raise RuntimeError("Transformation matrix chưa được tính toán")
        
        errors = []
        print("\n=== KIỂM TRA ĐỘ CHÍNH XÁC TRANSFORMATION ===")
        
        for i, point in enumerate(self.calibration_points, 1):
            camera_coord = point['camera']
            expected_robot = point['robot']
            
            # Chuyển đổi và so sánh
            calculated_robot = self.camera_to_robot(camera_coord[0], camera_coord[1])
            
            # Tính error
            error_x = abs(calculated_robot[0] - expected_robot[0])
            error_y = abs(calculated_robot[1] - expected_robot[1])
            error_total = np.sqrt(error_x**2 + error_y**2)
            
            errors.append(error_total)
            
            print(f"Điểm {i}:")
            print(f"  Camera: {camera_coord}")
            print(f"  Expected Robot: {expected_robot}")
            print(f"  Calculated Robot: ({calculated_robot[0]:.4f}, {calculated_robot[1]:.4f})")
            print(f"  Error: ({error_x:.4f}, {error_y:.4f}) -> Total: {error_total:.4f}")
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"\nTÓM TẮT:")
        print(f"  Mean error: {mean_error:.4f}")
        print(f"  Max error: {max_error:.4f}")
        print("=" * 50)
        
        return {
            'mean_error': mean_error,
            'max_error': max_error,
            'individual_errors': errors
        }
    
    def save_calibration(self, filename: str = "robot_calibration.json"):
        """
        Lưu calibration points và transformation matrix.
        
        Args:
            filename: Tên file để lưu
        """
        data = {
            'calibration_points': self.calibration_points,
            'transformation_matrix': self.transformation_matrix.tolist() if self.transformation_matrix is not None else None,
            'inverse_transformation_matrix': self.inverse_transformation_matrix.tolist() if self.inverse_transformation_matrix is not None else None
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[RobotCoordinateTransform] Calibration đã được lưu vào: {filename}")
    
    @classmethod
    def load_calibration(cls, filename: str = "robot_calibration.json"):
        """
        Tải calibration từ file.
        
        Args:
            filename: Tên file để tải
            
        Returns:
            Instance của RobotCoordinateTransform
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Không tìm thấy file calibration: {filename}")
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Tạo instance với calibration points
        instance = cls(data['calibration_points'])
        
        # Load transformation matrices nếu có
        if data.get('transformation_matrix'):
            instance.transformation_matrix = np.array(data['transformation_matrix'], dtype=np.float32)
        if data.get('inverse_transformation_matrix'):
            instance.inverse_transformation_matrix = np.array(data['inverse_transformation_matrix'], dtype=np.float32)
        
        print(f"[RobotCoordinateTransform] Calibration đã được tải từ: {filename}")
        return instance


# Để test module
if __name__ == "__main__":
    print("=== TEST ROBOT COORDINATE TRANSFORMATION ===")
    
    # Khởi tạo transformer với điểm mặc định
    transformer = RobotCoordinateTransform()
    
    # Kiểm tra độ chính xác
    validation_result = transformer.validate_transformation()
    
    # Test với một số điểm khác
    print("\n=== TEST VỚI CÁC ĐIỂM KHÁC ===")
    test_points = [
        (640, 350),   # Giữa ảnh
        (100, 100),   # Góc trên trái
        (1100, 600),  # Góc dưới phải
        (500, 200),   # Điểm random
    ]
    
    for camera_point in test_points:
        robot_point = transformer.camera_to_robot(camera_point[0], camera_point[1])
        
        # Test chuyển đổi ngược
        camera_back = transformer.robot_to_camera(robot_point[0], robot_point[1])
        
        print(f"Camera {camera_point} -> Robot ({robot_point[0]:.2f}, {robot_point[1]:.2f})")
        print(f"  Ngược lại: Robot ({robot_point[0]:.2f}, {robot_point[1]:.2f}) -> Camera ({camera_back[0]:.1f}, {camera_back[1]:.1f})")
    
    # Test batch transformation
    print("\n=== TEST BATCH TRANSFORMATION ===")
    batch_camera_points = [(76, 32), (640, 350), (1171, 667)]
    batch_robot_points = transformer.camera_to_robot_batch(batch_camera_points)
    
    for i, (cam_pt, robot_pt) in enumerate(zip(batch_camera_points, batch_robot_points)):
        print(f"Batch {i+1}: Camera {cam_pt} -> Robot ({robot_pt[0]:.2f}, {robot_pt[1]:.2f})")
    
    # Test với detection results
    print("\n=== TEST VỚI DETECTION RESULTS ===")
    mock_detection = [
        {
            'class': 'pallet',
            'confidence': 0.95,
            'bbox': [100, 100, 300, 250],
            'center': [200, 175],
            'position': {'x': 200, 'y': 175}
        }
    ]
    
    transformed_detection = transformer.transform_detection_results(mock_detection)
    print("Detection gốc:", mock_detection[0])
    print("Detection đã transform:", transformed_detection[0])
    
    # Lưu calibration
    transformer.save_calibration("robot_calibration_test.json") 