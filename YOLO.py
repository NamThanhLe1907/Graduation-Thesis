import cv2
import numpy as np
from ultralytics import YOLO
from Camera_Handler import CameraHandler, Logger
from Module_Division import Module1

from __future__ import annotations
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, List
from dataclasses import dataclass
from src.logger import AppLogger, PerformanceMetrics
from Camera_Handler import ICamera, HDWebCamera, CameraConfig, CameraError
from Module_Division import DivisionProcessor, DivisionConfig, DivisionError

logger = AppLogger.get_logger("yolo")

@dataclass(frozen=True)
class ModelConfig:
    model_path: str = "best.pt"
    confidence: float = 0.9
    iou_threshold: float = 0.7
    input_size: Tuple[int, int] = (640, 640)

@dataclass
class DetectionResult:
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    angles: List[float]
    contours: List[np.ndarray]

class DetectionError(Exception):
    """Base exception cho các lỗi phát hiện đối tượng"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error

class PalletProcessor:
    """Xử lý pipeline phát hiện và phân tích pallet"""
    
    def __init__(
        self,
        camera: ICamera,
        model_config: ModelConfig,
        division_config: DivisionConfig
    ):
        self.camera = camera
        self.model = self._load_model(model_config)
        self.division_processor = DivisionProcessor(division_config)
        self.metrics = PerformanceMetrics()

    def _load_model(self, config: ModelConfig) -> YOLO:
        """Khởi tạo YOLO model từ config"""
        try:
            model = YOLO(config.model_path)
            model.conf = config.confidence
            model.iou = config.iou_threshold
            return model
        except Exception as e:
            logger.error(f"Lỗi khởi tạo model: {str(e)}")
            raise DetectionError("Không thể khởi tạo YOLO model") from e

    def process_frame(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """Xử lý frame qua pipeline phát hiện và phân tích"""
        try:
            start_time = time.perf_counter()
            
            # Phát hiện đối tượng
            results = self.model(frame, verbose=False)
            
            if not results or len(results[0].obb) == 0:
                logger.debug("Không phát hiện đối tượng")
                return None

            # Trích xuất thông tin detection
            obb_data = results[0].obb.data.cpu().numpy()
            contours = self._extract_contours(obb_data)
            angles = self._calculate_angles(contours)
            
            # Tạo kết quả
            det_result = DetectionResult(
                boxes=obb_data[:, :4],
                scores=obb_data[:, 4],
                class_ids=obb_data[:, 6].astype(int),
                angles=angles,
                contours=contours
            )
            
            # Cập nhật metrics
            self.metrics.update(time.perf_counter() - start_time)
            logger.info(f"Xử lý frame thành công: {len(contours)} đối tượng")
            return det_result
            
        except Exception as e:
            logger.error(f"Lỗi xử lý frame: {str(e)}")
            self.metrics.update(0, success=False)
            raise DetectionError("Lỗi xử lý frame") from e

    def _extract_contours(self, obb_data: np.ndarray) -> List[np.ndarray]:
        """Trích xuất contour từ OBB data"""
        return [pred[:8].reshape(-1, 2).astype(np.int32) for pred in obb_data]

    def _calculate_angles(self, contours: List[np.ndarray]) -> List[float]:
        """Tính toán góc xoay từ contour"""
        angles = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            _, (w, h), angle = rect
            angles.append(self._normalize_angle(angle, w, h))
        return angles

    def _normalize_angle(self, angle: float, width: float, height: float) -> float:
        """Chuẩn hóa góc về khoảng [-45, 45] độ"""
        aspect_ratio = max(width, height) / min(width, height)
        if not (1.5 < aspect_ratio < 8.0):
            raise ValueError(f"Tỷ lệ khung hình không hợp lệ: {aspect_ratio:.2f}")
        return angle if abs(angle) < 45 else angle - 90

    def calculate_refined_angles(self, contours):
        """Tính toán góc xoay chính xác sử dụng RANSAC và hình học contour"""
        angles = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            _, _, angle = rect
            angles.append(angle if abs(angle) < 45 else angle - 90)
        return angles

    def validate_angle(self, angle, width, height):
        """Kiểm tra tính hợp lệ của góc dựa trên tỷ lệ W/H"""
        aspect_ratio = max(width, height) / min(width, height)
        return 1.5 < aspect_ratio < 8.0 and -45 < angle < 45

import threading

class AngleCalculator:
    @staticmethod
    def calculate_angle(cnt):
        """Tính toán góc xoay chính xác sử dụng PCA"""
        mean, eigenvectors = cv2.PCACompute(cnt.astype(np.float32), None)
        angle = np.degrees(np.arctan2(eigenvectors[0,1], eigenvectors[0,0]))
        return angle[0]

    @staticmethod
    def is_valid_angle(w, h, angle):
        """Kiểm tra tính hợp lệ của góc dựa trên tỷ lệ W/H"""
        aspect_ratio = max(w, h) / min(w, h)
        return 1.5 < aspect_ratio < 8.0 and -45 < angle < 45

class ProcessingThread(threading.Thread):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True
        self.frame_buffer = None

    def run(self):
        while self.running:
            frame = self.camera.capture_frame()
            if frame is not None:
                self.frame_buffer = frame.copy()

def main():
    # Load mô hình YOLO đã được train cho OBB (Oriented Bounding Boxes)
    model = YOLO('best.pt')
    # Khởi tạo Camera với độ phân giải 1280x1024
    camera = CameraHandler()
    # Module chia pallet không được dùng trong phần demo này
    division_module = Module1()
    row = 1  # Số hàng (không ảnh hưởng tới OBB)

    while True:
        frame = camera.capture_frame()
        if frame is None:
            continue

        # Dự đoán bằng YOLO với ngưỡng confidence 0.90
        results = model(frame, conf=0.90)
        # Sử dụng bản sao của frame gốc để vẽ rotated bounding boxes
        annotated_frame = frame.copy()

        # Xử lý OBB kết hợp RANSAC và hình học contour
        if results and results[0].obb is not None and len(results[0].obb.data) > 0:
            obb_preds = results[0].obb.data.cpu().numpy()
            
            # Tìm contour từ OBB points
            contours = [pred[:8].reshape(-1,2).astype(np.int32) for pred in obb_preds]
            
            # Tính toán góc xoay chính xác bằng phương pháp hình học
            refined_angles = []
            for cnt in contours:
                # Sử dụng PCA để xác định hướng chính
                mean, eigenvectors = cv2.PCACompute(cnt.astype(np.float32), mean=None)
                angle = np.degrees(np.arctan2(eigenvectors[0,1], eigenvectors[0,0]))
                refined_angles.append(angle)
            
            # Lấy toàn bộ điểm OBB và chuyển đổi sang contour
            all_contours = []
            for pred in obb_preds:
                points = pred[:8].reshape(-1, 2)
                all_contours.append(points.astype(np.int32))
            
            # Tính toán góc xoay chính xác bằng hình học
            refined_angles = self.calculate_refined_angles(all_contours)
            
            for idx, (pred, angle) in enumerate(zip(obb_preds, refined_angles)):
                try:
                    center_x, center_y, w, h, _, conf, cls = map(float, pred)
                    center = (int(center_x), int(center_y))
                    
                    # Kiểm tra độ tin cậy góc xoay
                    angle_valid = self.validate_angle(angle, w, h)
                    
                    # Xây dựng rotated rectangle dựa trên OBB
                    rect = (center, (w, h), angle)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.int32(box_points)
                    
                    # Vẽ rotated bounding box lên annotated_frame
                    cv2.polylines(annotated_frame, [box_points], isClosed=True, color=(0, 255, 0), thickness=2)
                    # Vẽ điểm trung tâm
                    cv2.circle(annotated_frame, center, 5, (255, 0, 0), -1)
                    # In thông tin center và góc xoay lên ảnh
                    text = f"Center: {center}, Angle: {angle:.1f} deg"
                    cv2.putText(annotated_frame, text, (center[0]-50, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Log thông tin detection
                    Logger.log(f"Detection - Center: {center}, Angle: {angle:.1f} deg, Class: {int(cls)}", debug=True)
                except Exception as e:
                    Logger.log(f"Error processing OBB detection: {e}", debug=True)
        else:
            Logger.log("No OBB detections", debug=True)

        cv2.imshow("YOLO OBB Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

    # Thêm các phương thức xác thực và tính toán góc
    def calculate_refined_angles(self, contours):
        """Tính toán góc xoay chính xác sử dụng RANSAC và hình học contour"""
        angles = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            _, _, angle = rect
            angles.append(angle if abs(angle) < 45 else angle - 90)
        return angles

    def validate_angle(self, angle, width, height):
        """Kiểm tra tính hợp lệ của góc dựa trên tỷ lệ W/H"""
        aspect_ratio = max(width, height) / min(width, height)
        return 1.5 < aspect_ratio < 8.0 and -45 < angle < 45

if __name__ == "__main__":
    main()
