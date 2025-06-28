"""
Hướng dẫn tích hợp Robot Coordinate Transform vào hệ thống chính
Để chuyển đổi tọa độ detection từ camera sang robot coordinates cho PLC
"""

from detection.pipeline import DetectionPipeline
from detection.utils.robot_coordinate_transform import RobotCoordinateTransform
import cv2
import time


class RobotIntegratedPipeline:
    """Pipeline tích hợp chuyển đổi tọa độ robot cho PLC."""
    
    def __init__(self, model_path: str = "best.engine"):
        """
        Khởi tạo pipeline với robot coordinate transformation.
        
        Args:
            model_path: Đường dẫn tới model YOLO
        """
        # Khởi tạo detection pipeline
        self.detection_pipeline = DetectionPipeline(model_path)
        
        # Khởi tạo robot coordinate transformer
        self.robot_transformer = RobotCoordinateTransform()
        
        print("[RobotIntegratedPipeline] Pipeline đã được khởi tạo với robot coordinate transformation")
        
        # Validation transformation
        validation_result = self.robot_transformer.validate_transformation()
        print(f"[RobotIntegratedPipeline] Transformation accuracy - Mean error: {validation_result['mean_error']:.4f}")
    
    def process_frame_for_plc(self, frame):
        """
        Xử lý frame và trả về kết quả với tọa độ robot cho PLC.
        
        Args:
            frame: Frame từ camera
            
        Returns:
            Dict chứa kết quả detection và tọa độ robot
        """
        # 1. Chạy detection pipeline
        detection_results = self.detection_pipeline.process_frame(frame)
        
        # 2. Chuyển đổi sang robot coordinates
        robot_results = self.robot_transformer.transform_detection_results(detection_results)
        
        # 3. Chuẩn bị dữ liệu cho PLC
        plc_data = self._prepare_plc_data(robot_results)
        
        return {
            'detection_results': detection_results,      # Kết quả gốc
            'robot_results': robot_results,              # Kết quả với robot coords
            'plc_data': plc_data,                       # Dữ liệu formatted cho PLC
            'frame_processed': frame                     # Frame đã xử lý
        }
    
    def _prepare_plc_data(self, robot_results):
        """
        Chuẩn bị dữ liệu theo format cho PLC.
        
        Args:
            robot_results: Kết quả đã chuyển đổi robot coordinates
            
        Returns:
            List dữ liệu cho PLC
        """
        plc_data = []
        
        for i, result in enumerate(robot_results, 1):
            if 'center_robot' in result:
                robot_coord = result['center_robot']
                
                plc_item = {
                    'id': i,
                    'x': round(robot_coord[0], 2),          # Tọa độ X robot (mm hoặc m)
                    'y': round(robot_coord[1], 2),          # Tọa độ Y robot (mm hoặc m)
                    'confidence': round(result.get('confidence', 0.0), 2),
                    'class': result.get('class', 'unknown'),
                    'timestamp': time.time()
                }
                
                # Thêm thông tin bounding box robot nếu có
                if 'bbox_robot' in result:
                    bbox_robot = result['bbox_robot']
                    plc_item.update({
                        'bbox_robot_x1': round(bbox_robot[0], 2),
                        'bbox_robot_y1': round(bbox_robot[1], 2),
                        'bbox_robot_x2': round(bbox_robot[2], 2),
                        'bbox_robot_y2': round(bbox_robot[3], 2),
                        'width_robot': round(abs(bbox_robot[2] - bbox_robot[0]), 2),
                        'height_robot': round(abs(bbox_robot[3] - bbox_robot[1]), 2)
                    })
                
                plc_data.append(plc_item)
        
        return plc_data
    
    def run_realtime_with_plc_output(self, camera_index=0, display_results=True):
        """
        Chạy realtime với output cho PLC.
        
        Args:
            camera_index: Index camera
            display_results: Có hiển thị kết quả trên màn hình không
        """
        cap = cv2.VideoCapture(camera_index)
        
        print("[RobotIntegratedPipeline] Bắt đầu xử lý realtime với PLC output")
        print("Nhấn 'q' để thoát, 'p' để in PLC data")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Xử lý frame
                result = self.process_frame_for_plc(frame)
                plc_data = result['plc_data']
                
                # Hiển thị kết quả nếu cần
                if display_results:
                    self._draw_results_on_frame(frame, result)
                    cv2.imshow("Robot Integrated Detection", frame)
                
                # In PLC data nếu có detection
                if plc_data:
                    print(f"\n[PLC DATA] {len(plc_data)} objects detected:")
                    for item in plc_data:
                        print(f"  ID{item['id']}: X={item['x']}, Y={item['y']} ({item['class']}, conf={item['confidence']})")
                
                # Xử lý phím
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    print("\n=== CURRENT PLC DATA ===")
                    if plc_data:
                        for item in plc_data:
                            print(f"PLC Item {item['id']}: {item}")
                    else:
                        print("No objects detected")
                    print("========================")
        
        except KeyboardInterrupt:
            print("\nDừng bởi người dùng")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _draw_results_on_frame(self, frame, result):
        """Vẽ kết quả lên frame."""
        robot_results = result['robot_results']
        
        for res in robot_results:
            # Vẽ bounding box camera
            if 'bbox' in res:
                bbox = res['bbox']
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            
            # Vẽ center point và robot coordinates
            if 'center' in res and 'center_robot' in res:
                center = res['center']
                robot_coord = res['center_robot']
                
                # Vẽ center point
                cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
                
                # Vẽ text với robot coordinates
                text = f"Robot: ({robot_coord[0]:.1f}, {robot_coord[1]:.1f})"
                cv2.putText(frame, text, (int(center[0]) + 10, int(center[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Vẽ class và confidence
                if 'class' in res and 'confidence' in res:
                    class_text = f"{res['class']}: {res['confidence']:.2f}"
                    cv2.putText(frame, class_text, (int(center[0]) + 10, int(center[1]) + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def demo_quick_coordinate_conversion():
    """Demo nhanh chuyển đổi tọa độ."""
    print("=== DEMO CHUYỂN ĐỔI TỌA ĐỘ NHANH ===")
    
    # Khởi tạo transformer
    transformer = RobotCoordinateTransform()
    
    # Test với các tọa độ camera thường gặp
    test_coordinates = [
        (400, 300),   # Center-left của ảnh
        (640, 400),   # Center của ảnh  
        (800, 500),   # Center-right của ảnh
        (200, 200),   # Upper area
        (1000, 600),  # Lower area
    ]
    
    print("Kết quả chuyển đổi:")
    for i, (cam_x, cam_y) in enumerate(test_coordinates, 1):
        robot_x, robot_y = transformer.camera_to_robot(cam_x, cam_y)
        print(f"{i}. Camera({cam_x:4d}, {cam_y:3d}) -> Robot({robot_x:8.2f}, {robot_y:8.2f})")
    
    print("\nSử dụng trong code:")
    print("```python")
    print("transformer = RobotCoordinateTransform()")
    print("robot_x, robot_y = transformer.camera_to_robot(camera_x, camera_y)")
    print("# Gửi robot_x, robot_y tới PLC")
    print("```")


if __name__ == "__main__":
    print("=== ROBOT COORDINATE INTEGRATION DEMO ===")
    
    # Demo chuyển đổi tọa độ nhanh
    demo_quick_coordinate_conversion()
    
    print("\n" + "="*60)
    print("Để chạy pipeline đầy đủ:")
    print("1. Đảm bảo có model 'best.engine'")
    print("2. Kết nối camera")
    print("3. Chạy: pipeline = RobotIntegratedPipeline()")
    print("4. Chạy: pipeline.run_realtime_with_plc_output()")
    
    # Uncomment để chạy pipeline đầy đủ (cần camera và model)
    # try:
    #     pipeline = RobotIntegratedPipeline()
    #     pipeline.run_realtime_with_plc_output(camera_index=0)
    # except Exception as e:
    #     print(f"Lỗi khi chạy pipeline: {e}")
    #     print("Đảm bảo có camera và model YOLO") 