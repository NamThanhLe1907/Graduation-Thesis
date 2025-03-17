import cv2
import numpy as np
from Camera_Handler import Logger

from typing import List, Tuple, Optional
import cv2
import numpy as np
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
            # Validate input
            if ordered_box.shape != (4, 2):
                raise InvalidInputError("ordered_box phải có shape (4, 2)")
                
            # Tính toán các điểm chia
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
            # Validate input
            if ordered_box.shape != (4, 2):
                raise InvalidInputError("ordered_box phải có shape (4, 2)")
                
            # Tính toán các điểm chia
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
            # Validate input types
            if not isinstance(ordered_box, (np.ndarray, list)):
                raise TypeError("ordered_box phải là numpy array hoặc list")
            if len(ordered_box) != 4:
                raise ValueError("ordered_box phải chứa chính xác 4 điểm")
            if not isinstance(row, int):
                raise TypeError("Tham số row phải là kiểu integer")

            # Convert to numpy array và kiểm tra shape
            ordered_box = np.array(ordered_box, dtype=np.float32)
            if ordered_box.shape != (4, 2):
                raise ValueError("Mỗi điểm trong ordered_box phải có 2 tọa độ (x,y)")

            # Vector hóa tính toán điểm chia
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
            Logger.log(f"[MODULE DIVISION ERROR] {str(e)}", debug=True)
            raise ValueError("Không thể tính toán điểm chia") from e
        """
        Tính toán các điểm chia pallet với kiểm tra đầu vào và tối ưu vector hóa
        :param ordered_box: 4 điểm góc pallet dạng numpy array hoặc list
        :param row: Số hàng (integer)
        :return: Danh sách điểm chia dạng numpy array
        """
        try:
            # Validate input types
            if not isinstance(ordered_box, (np.ndarray, list)):
                raise TypeError("ordered_box phải là numpy array hoặc list")
            if len(ordered_box) != 4:
                raise ValueError("ordered_box phải chứa chính xác 4 điểm")
            if not isinstance(row, int):
                raise TypeError("Tham số row phải là kiểu integer")

            # Convert to numpy array và kiểm tra shape
            ordered_box = np.array(ordered_box, dtype=np.float32)
            if ordered_box.shape != (4, 2):
                raise ValueError("Mỗi điểm trong ordered_box phải có 2 tọa độ (x,y)")

            # Vector hóa tính toán điểm chia
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
            Logger.log(f"[MODULE DIVISION ERROR] {str(e)}", debug=True)
            raise ValueError("Không thể tính toán điểm chia") from e
        """
        Tính toán các điểm chia pallet với kiểm tra đầu vào và tối ưu vector hóa
        :param ordered_box: 4 điểm góc pallet dạng numpy array hoặc list
        :param row: Số hàng (integer)
        :return: Danh sách điểm chia dạng numpy array
        """
        try:
            # Validate input types
            if not isinstance(ordered_box, (np.ndarray, list)):
                raise TypeError("ordered_box phải là numpy array hoặc list")
            if len(ordered_box) != 4:
                raise ValueError("ordered_box phải chứa chính xác 4 điểm")
            if not isinstance(row, int):
                raise TypeError("Tham số row phải là kiểu integer")
            # Kiểm tra đầu vào
            if not isinstance(row, int):
                raise TypeError("Tham số row phải là kiểu integer")
            if not isinstance(ordered_box, (np.ndarray, list)) or len(ordered_box) != 4:
                raise ValueError("ordered_box phải có 4 điểm")
                
            ordered_box = np.array(ordered_box, dtype=np.float32)
            
            # Vector hóa tính toán điểm chia
            if row % 2 == 1:  # Chia ngang cho hàng lẻ
                ratios = np.array([[1/3, 2/3], [1/3, 2/3]])
                horizontal_points = ordered_box[0] + (ordered_box[1] - ordered_box[0]) * ratios
                vertical_points = ordered_box[3] + (ordered_box[2] - ordered_box[3]) * ratios
                return np.vstack([horizontal_points, vertical_points]).tolist()
            
            # Chia dọc cho hàng chẵn
                ratios = np.array([[1/3, 2/3], [1/3, 2/3]])
                vertical_points = ordered_box[0] + (ordered_box[3] - ordered_box[0]) * ratios
                horizontal_points = ordered_box[1] + (ordered_box[2] - ordered_box[1]) * ratios
                return np.vstack([vertical_points, horizontal_points]).tolist()
                
            else:  # Row chẵn - chia dọc
                ratios = np.array([[1/3, 2/3], [1/3, 2/3]])
                vertical_points = ordered_box[0] + (ordered_box[3] - ordered_box[0]) * ratios
                horizontal_points = ordered_box[1] + (ordered_box[2] - ordered_box[1]) * ratios
                return np.vstack([vertical_points, horizontal_points]).tolist()
    
        except Exception as e:
                Logger.log(f"Lỗi khi tính toán điểm chia: {str(e)}", debug=True)
                raise   

        
    # def divide(self, ordered_box, row, pixel_to_robot):
    #     if not isinstance(ordered_box, (np.ndarray, list, tuple)) or len(ordered_box) != 4:
    #         raise ValueError(f"ordered_box phải có 4 phần tử, nhận được {len(ordered_box)}")
        
    #     ordered_box = np.array(ordered_box, dtype=np.float32)
        
    #     A, B, C, D = ordered_box
    #     coordinates = []
    #     alpha_values = np.linspace(1/3, 1, 3)

    #     for alpha in alpha_values:
    #         if row == 1:
    #             start = A + alpha * (D - A)
    #             end = B + alpha * (C - B)
    #         else:
    #             start = A + alpha * (B - A)
    #             end = D + alpha * (C - D)

    #         center = np.mean([start, end], axis=0)
    #         coordinates.append(pixel_to_robot(center) if pixel_to_robot else center)

    #     return coordinates

    def divide(self, ordered_box, row, pixel_to_robot):
        """
        Tính toán tọa độ các điểm chia (dùng cho logic xử lý)
        :param ordered_box: 4 điểm góc của pallet
        :param row: Số hàng
        :param pixel_to_robot: Hàm chuyển đổi tọa độ
        :return: Danh sách tọa độ các điểm chia
        """
        try:
            # Tính toán các điểm chia
            points = self.calculate_division_points(ordered_box, row)
            
            # Chuyển đổi tọa độ nếu cần
            if pixel_to_robot:
                points = [pixel_to_robot(p) for p in points]
                
            return points
            
        except Exception as e:
            Logger.log(f"Lỗi khi chia: {str(e)}", debug=True)
            raise

    # def draw_division_points(self, frame, ordered_box, pixel_to_robot):
    #     if not isinstance(ordered_box, (list, tuple)) or len(ordered_box) != 4:
    #         raise ValueError(f"ordered_box phải có 4 phần tử, nhưng nhận được {len(ordered_box)}: {ordered_box}")

    #     M, N, O, P = ordered_box
    #     intersections = [
    #         (self.find_intersection((M, (M + N)/2), (P, (P + O)/3)), "Bao 1", (0, 0, 255)),
    #         (self.find_intersection((N, (N + O)/2), (O, (O + P)/3)), "Bao 2", (0, 255, 255)),
    #         (self.find_intersection((M, (M + P)/2), (N, (N + O)/3)), "Bao 3", (255, 0, 255)),
    #     ]

    #     for point, label, color in intersections:
    #         if point:
    #             cv2.circle(frame, point, 5, color, -1)
    #             if pixel_to_robot:
    #                 robot_coords = pixel_to_robot(point)
    #                 cv2.putText(frame, f"{label}: {robot_coords}", (point[0]+10, point[1]-10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    def draw_division_points(self, frame, ordered_box, pixel_to_robot, row):
        """
        Vẽ các điểm và đường phân chia lên frame
        :param frame: Ảnh đầu vào
        :param ordered_box: 4 điểm góc của pallet (A,B,C,D)
        :param pixel_to_robot: Hàm chuyển đổi tọa độ
        :param row: Số hàng (lẻ/chẵn)
        """
        try:
            # Tính toán các điểm chia
            points = self.calculate_division_points(ordered_box, row)
            
        # Debug: Kiểm tra giá trị points
            print(f"Calculated division points: {points}")  # Kiểm tra giá trị trả về
            if points is None or len(points) == 0:
                raise ValueError("calculate_division_points() trả về None hoặc rỗng")

        # Kiểm tra từng điểm có hợp lệ không
            for i, point in enumerate(points):
                if point is None or len(point) != 2:
                    raise ValueError(f"Điểm {i} ({point}) không hợp lệ")            
            
            # Màu sắc và độ dày
            line_color = (0, 255, 0)  # Màu xanh lá
            line_div_color = (0,255,0)
            point_color = (0, 0, 255)  # Màu đỏ
            thickness = 2
            
            if row % 2 == 1:  # Row lẻ - chia ngang
                A, B, C, D, M, N, O, P = points 
                cv2.line(frame, tuple(M.astype(int)), tuple(O.astype(int)), line_color, thickness)
                cv2.line(frame, tuple(N.astype(int)), tuple(P.astype(int)), line_color, thickness)
                
                # #AO, MD
                # cv2.line(frame, tuple(A.astype(int)), tuple(O.astype(int)), line_div_color, thickness)
                # cv2.line(frame, tuple(M.astype(int)), tuple(D.astype(int)), line_div_color, thickness)
                # #NC,BP
                # cv2.line(frame, tuple(N.astype(int)), tuple(C.astype(int)), line_div_color, thickness)
                # cv2.line(frame, tuple(B.astype(int)), tuple(P.astype(int)), line_div_color, thickness)                
                # #MP,NO
                # cv2.line(frame, tuple(M.astype(int)), tuple(P.astype(int)), line_div_color, thickness)
                # cv2.line(frame, tuple(N.astype(int)), tuple(O.astype(int)), line_div_color, thickness)  
            else:  # Row chẵn - chia dọc
                G, H, K, I = points
                cv2.line(frame, tuple(G.astype(int)), tuple(H.astype(int)), line_color, thickness)
                cv2.line(frame, tuple(K.astype(int)), tuple(I.astype(int)), line_color, thickness)
                
            # Vẽ các điểm
            for point in points:
                cv2.circle(frame, tuple(point.astype(int)), 5, point_color, -1)
                
        except Exception as e:
            Logger.log(f"Lỗi khi vẽ: {str(e)}", debug=True)
            raise
