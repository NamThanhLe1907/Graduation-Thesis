import numpy as np
import cv2

class Calibration:
    def __init__(self, image_points, robot_points):
        """
        Khởi tạo lớp Calibration với các điểm ảnh và điểm robot.
        :param image_points: Danh sách tọa độ điểm trong không gian ảnh.
        :param robot_points: Danh sách tọa độ điểm trong không gian robot.
        """
        self.image_points = np.array(image_points, dtype=np.float32)
        self.robot_points = np.array(robot_points, dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(self.image_points, self.robot_points)

    def pixel_to_robot(self, pixel_coord):
        """
        Chuyển đổi tọa độ pixel sang tọa độ robot sử dụng ma trận Homography.
        :param pixel_coord: Tọa độ pixel cần chuyển đổi.
        :return: Tọa độ tương ứng trong không gian robot.
        """
        pixel_coord_homogeneous = np.array([pixel_coord[0], pixel_coord[1], 1.0])
        robot_coord_homogeneous = np.dot(self.homography_matrix, pixel_coord_homogeneous)
        robot_coord_homogeneous /= robot_coord_homogeneous[2]
        return robot_coord_homogeneous[:2]

class PalletDetection:
    def __init__(self, division_module=None, min_area=150000, missing_threshold=50):
        self.division_module = division_module if division_module else Module1()
        self.min_area = min_area
        self.missing_threshold = missing_threshold
        self.pallet_list = []

    def find_pallet_in_list(self, center):
        centers = np.array([pallet['center'] for pallet in self.pallet_list])
        if centers.size == 0:
            return -1
        distances = np.linalg.norm(np.array(center) - centers, axis=1)
        min_idx = np.argmin(distances)
        return min_idx if distances[min_idx] < self.missing_threshold else -1

    def detect_pallets(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 70, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_pallets = []
        for contour in contours:
            if len(contour) < 5 or cv2.contourArea(contour) < self.min_area * 0.3:
                continue

            rect = cv2.minAreaRect(contour)
            (w, h), area = rect[1], rect[1][0] * rect[1][1]
            if area < self.min_area:
                continue

            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if 0.95 <= aspect_ratio <= 1.05:
                box = cv2.boxPoints(rect).astype(int)
                center = tuple(np.mean(box, axis=0).astype(int))
                detected_pallets.append({'box': box, 'center': center})
                cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        return detected_pallets, frame, edges

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

    def order_points(self, box):
        centroid = np.mean(box, axis=0)
        angles = np.arctan2(box[:,1]-centroid[1], box[:,0]-centroid[0])
        return box[np.argsort(angles)]

    def divide_pallet_by_row(self, pallet, row=1, pixel_to_robot=None):
        ordered_box = self.order_points(np.array(pallet['box']))
        A, B, C, D = ordered_box

        coordinates = []
        division_lines = []
        alpha_values = np.linspace(1/3, 1, 3)

        for alpha in alpha_values:
            start = A + alpha * (D - A)
            end = B + alpha * (C - B)

            division_line = (tuple(start.astype(int)), tuple(end.astype(int)))
            division_lines.append(division_line)
            center = np.mean([start, end], axis=0)
            coordinates.append(pixel_to_robot(tuple(center)) if pixel_to_robot else tuple(center))

        return coordinates, division_lines

    def process_pallet(self, pallet, row=1, pixel_to_robot=None):
        ordered_box = self.order_points(pallet['box'])
        coordinates = self.division_module.divide(ordered_box, row, pixel_to_robot)
        return coordinates

class DivisionModule:
    def divide(self, ordered_box, row, pixel_to_robot):
        raise NotImplementedError
        
    def draw_division_points(self, frame, ordered_box, pixel_to_robot):
        raise NotImplementedError

class Module1(DivisionModule):
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

    def divide(self, ordered_box, row, pixel_to_robot):
        A, B, C, D = ordered_box
        coordinates = []
        alpha_values = np.linspace(1/3, 1, 3)

        for alpha in alpha_values:
            start = A + alpha * (D - A)
            end = B + alpha * (C - B)

            center = np.mean([start, end], axis=0)
            coordinates.append(pixel_to_robot(center) if pixel_to_robot else center)
            
        return coordinates

    def draw_division_points(self, frame, ordered_box, pixel_to_robot):
        M, N, O, P = ordered_box
        intersections = [
            (self.find_intersection((M, (M + N)/2), (P, (P + O)/3)), "Bao 1", (0, 0, 255)),
            (self.find_intersection((N, (N + O)/2), (O, (O + P)/3)), "Bao 2", (0, 255, 255)),
            (self.find_intersection((M, (M + P)/2), (N, (N + O)/3)), "Bao 3", (255, 0, 255)),
        ]

        for point, label, color in intersections:
            if point:
                cv2.circle(frame, point, 5, color, -1)
                if pixel_to_robot:
                    robot_coords = pixel_to_robot(point)
                    cv2.putText(frame, f"{label}: {robot_coords}", (point[0]+10, point[1]-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
