"""
Test Robot Coordinates với ảnh tĩnh
Script để test chuyển đổi tọa độ từ camera sang robot với từng ảnh
"""
import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Import các module cần thiết
from detection.utils.robot_coordinate_transform import RobotCoordinateTransform
from detection.utils.camera_calibration import CameraCalibration
from detection.utils.tensorrt_yolo import YOLOTensorRT


class ImageRobotCoordinateTest:
    """Class test tọa độ robot với ảnh tĩnh."""
    
    def __init__(self, model_path: str = "best.engine"):
        """
        Khởi tạo test pipeline.
        
        Args:
            model_path: Đường dẫn tới model YOLO
        """
        try:
            # Khởi tạo YOLO model
            print("[ImageTest] Đang khởi tạo YOLO model...")
            self.yolo_model = YOLOTensorRT(model_path)
            
            # Khởi tạo robot coordinate transformer
            print("[ImageTest] Đang khởi tạo robot coordinate transformer...")
            self.robot_transformer = RobotCoordinateTransform()
            
            # Khởi tạo camera calibration (cho depth nếu có)
            print("[ImageTest] Đang khởi tạo camera calibration...")
            self.camera_calibration = CameraCalibration()
            
            print("✅ [ImageTest] Khởi tạo thành công!")
            
            # Validation transformation
            validation_result = self.robot_transformer.validate_transformation()
            print(f"📊 [ImageTest] Độ chính xác transformation - Mean error: {validation_result['mean_error']:.4f}")
            
        except Exception as e:
            print(f"❌ [ImageTest] Lỗi khởi tạo: {e}")
            raise
    
    def process_single_image(self, image_path: str, show_result: bool = True, save_result: bool = False):
        """
        Xử lý một ảnh và trả về tọa độ robot.
        
        Args:
            image_path: Đường dẫn tới ảnh
            show_result: Có hiển thị kết quả không
            save_result: Có lưu kết quả không
            
        Returns:
            Dict chứa kết quả detection và tọa độ robot
        """
        print(f"\n{'='*60}")
        print(f"🔍 Đang xử lý: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Load ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Không thể load ảnh: {image_path}")
            return None
        
        print(f"📸 Kích thước ảnh: {image.shape[1]}x{image.shape[0]}")
        
        try:
            # 1. Chạy YOLO detection
            print("🔍 Đang chạy YOLO detection...")
            detection_result = self.yolo_model.detect(image)
            
            if not detection_result or not detection_result.get('bounding_boxes'):
                print("❌ Không phát hiện object nào!")
                if show_result:
                    cv2.imshow(f"Result - {os.path.basename(image_path)}", image)
                    cv2.waitKey(0)
                return {'image_path': image_path, 'detections': [], 'robot_coordinates': []}
            
            bboxes = detection_result.get('bounding_boxes', [])
            scores = detection_result.get('scores', [])
            classes = detection_result.get('classes', [])
            
            print(f"✅ Phát hiện {len(bboxes)} object(s)")
            
            # 2. Chuyển đổi format detection để tương thích
            detection_results = []
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                conf = scores[i] if i < len(scores) else 0.0
                cls_id = classes[i] if i < len(classes) else 0
                
                detection_result = {
                    'bbox': [x1, y1, x2, y2],
                    'center': [center_x, center_y],
                    'confidence': conf,
                    'class': f'class_{int(cls_id)}',  # Tạm thời dùng class_id
                    'class_id': int(cls_id)
                }
                detection_results.append(detection_result)
            
            # 3. Chuyển đổi sang robot coordinates
            print("🔄 Đang chuyển đổi tọa độ robot...")
            robot_results = self.robot_transformer.transform_detection_results(detection_results)
            
            # 4. Tính tọa độ 3D nếu có depth (giả định depth = 2.0m cho test)
            test_depth = 2.0  # Depth giả định cho test
            
            # 5. Chuẩn bị kết quả
            final_results = []
            
            for i, result in enumerate(robot_results, 1):
                if 'center_robot' in result:
                    robot_coord = result['center_robot']
                    center_cam = result['center']
                    
                    # Tính tọa độ 3D với camera calibration
                    X_3d, Y_3d, Z_3d = self.camera_calibration.pixel_to_3d(
                        center_cam[0], center_cam[1], test_depth
                    )
                    
                    object_result = {
                        'id': i,
                        'class': result.get('class', 'unknown'),
                        'confidence': round(result.get('confidence', 0.0), 3),
                        'camera_coordinates': {
                            'pixel_x': int(center_cam[0]),
                            'pixel_y': int(center_cam[1])
                        },
                        'robot_coordinates': {
                            'x': round(robot_coord[0], 2),  # Robot X (mm hoặc m)
                            'y': round(robot_coord[1], 2),  # Robot Y (mm hoặc m)
                            'z': round(test_depth, 2)       # Depth (m)
                        },
                        'camera_3d_coordinates': {
                            'X': round(X_3d, 3),  # Camera coordinate X (m)
                            'Y': round(Y_3d, 3),  # Camera coordinate Y (m)
                            'Z': round(Z_3d, 3)   # Camera coordinate Z (m)
                        },
                        'bbox_camera': result.get('bbox', []),
                        'bbox_robot': result.get('bbox_robot', [])
                    }
                    
                    final_results.append(object_result)
                    
                    # In kết quả
                    print(f"\n📍 Object {i} ({object_result['class']}):")
                    print(f"   Confidence: {object_result['confidence']}")
                    print(f"   Camera pixel: ({object_result['camera_coordinates']['pixel_x']}, {object_result['camera_coordinates']['pixel_y']})")
                    print(f"   ➡️  Robot coordinates: X={object_result['robot_coordinates']['x']}, Y={object_result['robot_coordinates']['y']}, Z={object_result['robot_coordinates']['z']}")
                    print(f"   📐 Camera 3D: X={object_result['camera_3d_coordinates']['X']}, Y={object_result['camera_3d_coordinates']['Y']}, Z={object_result['camera_3d_coordinates']['Z']}")
            
            # 6. Hiển thị kết quả trên ảnh
            if show_result:
                result_image = self._draw_results_on_image(image.copy(), final_results)
                cv2.imshow(f"Result - {os.path.basename(image_path)}", result_image)
                print(f"\n💡 Nhấn phím bất kỳ để tiếp tục, 's' để lưu ảnh, 'q' để thoát...")
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('s'):
                    save_path = f"result_{os.path.basename(image_path)}"
                    cv2.imwrite(save_path, result_image)
                    print(f"💾 Đã lưu ảnh kết quả: {save_path}")
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return None
            
            # 7. Lưu kết quả JSON nếu cần
            if save_result:
                self._save_result_json(image_path, final_results)
            
            return {
                'image_path': image_path,
                'image_shape': image.shape,
                'detections': detection_results,
                'robot_coordinates': final_results,
                'total_objects': len(final_results)
            }
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý ảnh: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _draw_results_on_image(self, image, results):
        """Vẽ kết quả lên ảnh."""
        for result in results:
            # Lấy thông tin
            pixel_coord = (result['camera_coordinates']['pixel_x'], 
                          result['camera_coordinates']['pixel_y'])
            robot_coord = result['robot_coordinates']
            bbox = result.get('bbox_camera', [])
            
            # Vẽ bounding box nếu có
            if bbox and len(bbox) >= 4:
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            
            # Vẽ center point
            cv2.circle(image, pixel_coord, 8, (0, 0, 255), -1)
            cv2.circle(image, pixel_coord, 12, (255, 255, 255), 2)
            
            # Vẽ thông tin text
            y_offset = 0
            texts = [
                f"ID: {result['id']} ({result['class']})",
                f"Conf: {result['confidence']:.2f}",
                f"Robot: X={robot_coord['x']}, Y={robot_coord['y']}",
                f"Z={robot_coord['z']}m"
            ]
            
            for text in texts:
                text_pos = (pixel_coord[0] + 15, pixel_coord[1] - 10 + y_offset)
                
                # Vẽ background cho text
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(image, 
                            (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                            (text_pos[0] + text_size[0] + 2, text_pos[1] + 2),
                            (0, 0, 0), -1)
                
                # Vẽ text
                cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 0), 1)
                y_offset += 20
        
        # Vẽ thông tin tổng quan
        info_text = f"Total objects: {len(results)}"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2)
        
        return image
    
    def _save_result_json(self, image_path, results):
        """Lưu kết quả thành file JSON."""
        result_data = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'total_objects': len(results),
            'objects': results
        }
        
        json_filename = f"result_{os.path.splitext(os.path.basename(image_path))[0]}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Đã lưu kết quả JSON: {json_filename}")


def quick_test_single_image():
    """Test nhanh với 1 ảnh."""
    print("🔍 QUICK TEST - TEST VỚI 1 ẢNH")
    
    # Khởi tạo tester
    tester = ImageRobotCoordinateTest()
    
    # Test với ảnh đầu tiên trong thư mục
    test_image = "test_result/image_1.jpg"
    
    if os.path.exists(test_image):
        print(f"📸 Đang test với: {test_image}")
        result = tester.process_single_image(test_image, show_result=True, save_result=True)
        
        if result and result['robot_coordinates']:
            print(f"\n🎯 KẾT QUẢ TỔNG HỢP:")
            for obj in result['robot_coordinates']:
                robot_coord = obj['robot_coordinates']
                print(f"   {obj['class']} → Robot: X={robot_coord['x']}, Y={robot_coord['y']}, Z={robot_coord['z']}")
    else:
        print(f"❌ Không tìm thấy ảnh test: {test_image}")


def test_multiple_images():
    """Test với nhiều ảnh."""
    print("🔍 TEST VỚI NHIỀU ẢNH")
    
    # Khởi tạo tester
    tester = ImageRobotCoordinateTest()
    
    # Test với 3 ảnh đầu tiên
    test_images = [
        "images_pallets/image_1.jpg",
        "images_pallets/image_2.jpg", 
        "images_pallets/image_3.jpg"
    ]
    
    all_coordinates = []
    
    for i, image_path in enumerate(test_images, 1):
        if os.path.exists(image_path):
            print(f"\n{'='*20} ẢNH {i} {'='*20}")
            result = tester.process_single_image(image_path, show_result=False, save_result=False)
            
            if result and result['robot_coordinates']:
                for obj in result['robot_coordinates']:
                    robot_coord = obj['robot_coordinates']
                    coord_info = {
                        'image': os.path.basename(image_path),
                        'object_id': obj['id'],
                        'class': obj['class'],
                        'confidence': obj['confidence'],
                        'x': robot_coord['x'],
                        'y': robot_coord['y'],
                        'z': robot_coord['z']
                    }
                    all_coordinates.append(coord_info)
                    print(f"🎯 {obj['class']} (ID{obj['id']}) → Robot: X={robot_coord['x']}, Y={robot_coord['y']}, Z={robot_coord['z']}")
    
    # Tóm tắt tất cả tọa độ
    print(f"\n{'='*60}")
    print(f"📊 TÓM TẮT TẤT CẢ TỌA ĐỘ ROBOT:")
    print(f"{'='*60}")
    
    for coord in all_coordinates:
        print(f"{coord['image']:15s} | {coord['class']:8s} | X={coord['x']:8.2f} | Y={coord['y']:8.2f} | Z={coord['z']:6.2f} | Conf={coord['confidence']:.2f}")


if __name__ == "__main__":
    print("🤖 ROBOT COORDINATE TEST VỚI ẢNH TĨNH")
    print("="*50)
    
    print("\n📋 CHỌN KIỂU TEST:")
    print("1. Quick test với 1 ảnh (có hiển thị)")
    print("2. Test với 3 ảnh (chỉ in tọa độ)")
    print("3. Test manual (nhập đường dẫn ảnh)")
    
    choice = input("\n👉 Chọn (1-3): ").strip()
    
    try:
        if choice == "1":
            quick_test_single_image()
        elif choice == "2":
            test_multiple_images()
        elif choice == "3":
            image_path = input("📁 Nhập đường dẫn ảnh: ").strip()
            if os.path.exists(image_path):
                tester = ImageRobotCoordinateTest()
                tester.process_single_image(image_path, show_result=True, save_result=True)
            else:
                print(f"❌ Không tìm thấy ảnh: {image_path}")
        else:
            print("❌ Lựa chọn không hợp lệ")
    
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows() 