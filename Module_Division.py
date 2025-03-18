import cv2
import numpy as np
from Camera_Handler import Logger
from typing import List, Tuple, Optional
from dataclasses import dataclass
from src.logger import AppLogger

logger = AppLogger.get_logger("division")

@dataclass(frozen=True)
class DivisionConfig:
    num_sections: int = 3
    aspect_ratio_range: Tuple[float, float] = (1.5, 8.0)
    line_thickness: int = 2
    point_radius: int = 5

class DivisionError(Exception):
    """Base exception cho các lỗi chia điểm"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error

class InvalidInputError(DivisionError):
    """Lỗi dữ liệu đầu vào không hợp lệ"""

class CalculationError(DivisionError):
    """Lỗi trong quá trình tính toán"""

class DivisionStrategy:
    """Interface cho các chiến lược chia điểm"""
    def calculate_points(self, ordered_box: np.ndarray) -> List[Tuple[float, float]]:
        raise NotImplementedError

class HorizontalDivision(DivisionStrategy):
    """Chiến lược chia điểm theo chiều ngang"""
    def __init__(self, config: DivisionConfig):
        self.config = config
        
    def calculate_points(self, ordered_box: np.ndarray) -> List[Tuple[float, float]]:
        try:
            if ordered_box.shape != (4, 2):
                raise InvalidInputError("ordered_box phải có shape (4, 2)")
                
            ratios = np.array([[1/3], [2/3]])
            horizontal_points = ordered_box[0] + (ordered_box[1] - ordered_box[0]) * ratios
            vertical_points = ordered_box[3] + (ordered_box[2] - ordered_box[3]) * ratios
            return np.vstack([horizontal_points, vertical_points]).tolist()
            
        except Exception as e:
            logger.error(f"Lỗi tính toán điểm ngang: {str(e)}")
            raise CalculationError("Lỗi tính toán điểm chia ngang") from e

class VerticalDivision(DivisionStrategy):
    """Chiến lược chia điểm theo chiều dọc"""
    def __init__(self, config: DivisionConfig):
        self.config = config
        
    def calculate_points(self, ordered_box: np.ndarray) -> List[Tuple[float, float]]:
        try:
            if ordered_box.shape != (4, 2):
                raise InvalidInputError("ordered_box phải có shape (4, 2)")
                
            ratios = np.array([[1/3], [2/3]])
            vertical_points = ordered_box[0] + (ordered_box[3] - ordered_box[0]) * ratios
            horizontal_points = ordered_box[1] + (ordered_box[2] - ordered_box[1]) * ratios
            return np.vstack([vertical_points, horizontal_points]).tolist()
            
        except Exception as e:
            logger.error(f"Lỗi tính toán điểm dọc: {str(e)}")
            raise CalculationError("Lỗi tính toán điểm chia dọc") from e

class DivisionProcessor:
    def __init__(self):
        self.num_sections = 3
        
    @staticmethod
    def find_intersection(line1, line2):
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if det == 0:
            return None

        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / det
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / det
        
        return (int(px), int(py)) if (min(x1, x2) <= px <= max(x1, x2) and 
                                      min(y1, y2) <= py <= max(y1, y2) and
                                      min(x3, x4) <= px <= max(x3, x4) and
                                      min(y3, y4) <= py <= max(y3, y4)) else None

    def calculate_division_points(self, ordered_box: np.ndarray, row: int) -> list:
        """
        Tính toán các điểm chia pallet với kiểm tra đầu vào và tối ưu vector hóa
        :param ordered_box: 4 điểm góc pallet dạng numpy array hoặc list
        :param row: Số hàng (integer)
        :return: Danh sách điểm chia dạng numpy array
        """
        try:
            if not isinstance(ordered_box, (np.ndarray, list)):
                raise TypeError("ordered_box phải là numpy array hoặc list")
            if len(ordered_box) != 4:
                raise ValueError("ordered_box phải chứa chính xác 4 điểm")
            if not isinstance(row, int):
                raise TypeError("Tham số row phải là kiểu integer")

            ordered_box = np.array(ordered_box, dtype=np.float32)
            if ordered_box.shape != (4, 2):
                raise ValueError("Mỗi điểm trong ordered_box phải có 2 tọa độ (x,y)")

            ratios = np.array([[1/3], [2/3]])  # Tỷ lệ chia 1/3 và 2/3
            
            if row % 2 == 1:  # Chia ngang cho hàng lẻ
                horizontal_points = ordered_box[0] + (ordered_box[1] - ordered_box[0]) * ratios
                vertical_points = ordered_box[3] + (ordered_box[2] - ordered_box[3]) * ratios
                return np.vstack([horizontal_points, vertical_points]).tolist()
            
            else:  # Chia dọc cho hàng chẵn
                vertical_points = ordered_box[0] + (ordered_box[3] - ordered_box[0]) * ratios
                horizontal_points = ordered_box[1] + (ordered_box[2] - ordered_box[1]) * ratios
                return np.vstack([vertical_points, horizontal_points]).tolist()

        except Exception as e:
            logger.log(f"[MODULE DIVISION ERROR] {str(e)}", debug=True)
            raise ValueError("Không thể tính toán điểm chia") from e
