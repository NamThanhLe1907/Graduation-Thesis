import cv2
import numpy as np
from Camera_Handler import Logger

class PalletDetection:
    def __init__(self, division_module=None, 
                 min_area=10000, load_min_area=10000, 
                 missing_threshold=50, max_missing_frames=20):
        """
        Khởi tạo PalletDetection với khả năng phát hiện cả pallet và load
        :param division_module: Module chia pallet (Module1, Module2, Module3)
        :param min_area: Diện tích tối thiểu của pallet
        :param load_min_area: Diện tích tối thiểu của load
        :param missing_threshold: Ngưỡng khoảng cách tracking
        """
        self.division_module = division_module
        self.min_area = min_area
        self.load_min_area = load_min_area
        self.missing_threshold = missing_threshold
        self.max_missing_frames = max_missing_frames
        self.tracked_objects = []

        self.pallet_color = (0, 255, 0)  # Màu xanh lá
        self.load_color = (255, 0, 0)    # Màu đỏ

    def set_division_module(self, division_module):
        self.division_module = division_module
        Logger.log(f"Đã thiết lập module: {division_module.__class__.__name__}")

    def detect_objects(self, frame):
        """
        Phát hiện đồng thời pallet và load
        :return: (pallets, loads, processed_frame, edges)
        """
        # Tiền xử lý ảnh
        roi_frame = self.filter_roi(frame)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced_gray, (7, 7), 1)
        edges = cv2.Canny(blurred, 50, 100)
        kernel = np.ones((5, 5), np.uint8)
        # edges = cv2.erode(cv2.dilate(edges, kernel, iterations=2), kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        pallets , loads = self._detect_pallets_from_edges(edges)
        # Cập nhật tracking và vẽ kết quả
        processed_frame = roi_frame.copy()
        self._update_tracking(pallets, loads)
        self._draw_objects(processed_frame, pallets, loads)
        
    # Đảm bảo pallets và loads luôn là list
        pallets = pallets if pallets else []
        loads = loads if loads else []
        
        Logger.log(f"Returning: {len(pallets)} pallets, {len(loads)} loads", debug=True)
        
        return pallets, loads, processed_frame, edges

    
    def _detect_pallets_from_edges(self, edges):
        """Phát hiện pallet từ edges (giống code cũ)"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pallets = []
        loads = []
        for cnt in contours:
            if len(cnt) < 5:
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)
            rect = cv2.minAreaRect(cnt)
            center, (width, height), _ = rect
            box = cv2.boxPoints(rect).astype(int)

            if height == 0 or width == 0:
                continue

            aspect_ratio = width / height if width > height else height / width
            if 0.95 <= aspect_ratio <= 1.05 and width * height > self.min_area:
                pallets.append({
                    'type': 'pallet',
                    'box': self.order_points(box),
                    'center': tuple(map(int, center))
                })
                
            if 2.5 <= aspect_ratio <= 3.5 and width*height > self.load_min_area:
                    loads.append({
                        'type': 'load',
                        'box': self.order_points(box),
                        'center': tuple(np.mean(box, axis=0).astype(int))
                    })
        return pallets , loads

    def _update_tracking(self, pallets, loads):
        """Cập nhật trạng thái tracking cho tất cả đối tượng"""
        all_objects = pallets + loads
        for obj in all_objects:
            matched = False
            for tracked in self.tracked_objects:
                distance = np.linalg.norm(np.array(obj['center']) - np.array(tracked['center']))
                if distance < self.missing_threshold:
                    tracked.update(obj)
                    tracked['missing_frames'] = 0
                    matched = True
                    break
            if not matched:
                self.tracked_objects.append({**obj, 'missing_frames': 0})
        
        # Giảm bộ đếm cho các đối tượng không được phát hiện
        for tracked in self.tracked_objects:
            if tracked not in all_objects:
                tracked['missing_frames'] += 1

        # Loại bỏ đối tượng mất tích quá lâu
        self.tracked_objects = [
                                t for t in self.tracked_objects 
                               if t['missing_frames'] < self.max_missing_frames]

    def _draw_objects(self, frame, pallets, loads):
        """Vẽ thông tin lên frame"""
        for obj in pallets + loads:
            color = self.pallet_color if obj['type'] == 'pallet' else self.load_color
            cv2.drawContours(frame, [obj['box']], -1, color, 2)
            cv2.putText(frame, obj['type'], 
                       tuple(np.add(obj['center'], (10, -10))), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Giữ nguyên các hàm cũ cần thiết
    @staticmethod
    def order_points(box):
        centroid = np.mean(box, axis=0)
        angles = np.arctan2(box[:,1]-centroid[1], box[:,0]-centroid[0])
        return box[np.argsort(angles)]
    
    @staticmethod
    def filter_roi(frame):
        # Chuyển đổi ảnh sang HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ngưỡng màu xanh lá cây (băng tải)
        lower_green = np.array([35, 50, 70])  
        upper_green = np.array([85, 255, 255])
        
        # Ngưỡng màu đen (pallet)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 40])

        # Ngưỡng màu trắng (vật thể trên pallet)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])

        # 🟢 **Tọa độ pallet đã xác định trước**
        x1, y1 = 635, 142   # Góc trên trái
        x2, y2 = 1118, 142 # Góc trên phải
        x3, y3 =  1118, 709  # Góc dưới phải
        x4, y4 = 635, 709  # Góc dưới trái
        x5, y5 = 285, 5
        x6, y6 = 506, 9
        x7, y7 = 545, 702
        x8, y8 = 320, 702
        
        
        # 🟢 **Tạo mask từ 4 điểm tọa độ của pallet**
        mask_pallet = np.zeros(frame.shape[:2], dtype=np.uint8)
        points_pallet = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        mask_load = np.zeros(frame.shape[:2], dtype=np.uint8)
        points_load = np.array([[x5, y5], [x6, y6], [x7, y7], [x8, y8]],np.int32)
        
        cv2.fillPoly(mask_pallet, [points_pallet], 255)
        cv2.fillPoly(mask_load, [points_load], 255)
        

        # 🏭 **Tạo mask màu đen để tìm pallet**
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        masked_pallet = cv2.bitwise_and(mask_black, mask_pallet)
        
            # Tạo mask cho vùng xanh lá (băng tải)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        # Áp dụng morphological opening để loại bỏ nhiễu/bóng nhỏ trong vùng xanh
        kernel_small = np.ones((3, 3), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel_small)
        
        masked_load = cv2.bitwise_and(mask_green, mask_load)
        
        # ⚪ **Tạo mask màu trắng để tìm vật thể trên pallet**
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        masked_pallet_white = cv2.bitwise_and(mask_white, mask_pallet)

        # --- Chỉ giữ pallet trắng trong vùng băng tải ---
        masked_conveyor_white = cv2.bitwise_and(mask_white, mask_load)
        # Sau khi có masked_conveyor_white
        # 1) Lọc bằng kênh V
        v_channel = hsv[:, :, 2]
        shadow_mask = cv2.inRange(v_channel,90 , 255)
        masked_conveyor_white = cv2.bitwise_and(masked_conveyor_white, shadow_mask)

        # 2) Morphological Opening với kernel dọc
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 21))
        masked_conveyor_white = cv2.morphologyEx(masked_conveyor_white, cv2.MORPH_OPEN, kernel_vertical)


        final_mask = cv2.bitwise_or(masked_pallet_white, masked_conveyor_white)
        
        # 📏 **Làm sạch nhiễu**
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        
        roi = cv2.bitwise_and(frame, frame, mask=clean_mask)
        # cv2.imshow("Mask White", mask_white)
        # cv2.imshow("Mask green", mask_green)
        # cv2.imshow("Mask Pallet", masked_pallet)
        # cv2.imshow("Mask Load", masked_load)
        # cv2.imshow("Masked Pallet White", masked_pallet_white)
        # cv2.imshow("Masked conveyor white",masked_conveyor_white)
        # cv2.imshow("Final Mask", final_mask)
        # cv2.waitKey(0)
        return roi
        
    
    def divide(self, ordered_box, row, pixel_to_robot):
        if not self.division_module:
            raise ValueError("Chưa chọn module chia!")
        return self.division_module.divide(ordered_box, row, pixel_to_robot)

    def draw_division_points(self, frame, ordered_box, pixel_to_robot, row):
        if not self.division_module:
            raise ValueError("Chưa chọn module chia!")
        self.division_module.draw_division_points(frame, ordered_box, pixel_to_robot, row)