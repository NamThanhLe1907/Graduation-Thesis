import numpy as np

class PostProcessor:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.previous_boxes = None
        # Adaptive geometry constraints
        self.min_area = 0.0001    # 0.01% of image area
        self.max_area = 0.95      # 95% of image area
        self.wh_ratio_range = (0.2, 5.0)  # Extreme width/height ratios

    def filter_by_geometry(self, boxes, img_size):
        """
        Lọc boxes dựa trên các đặc trưng hình học.
        
        Args:
            boxes (np.ndarray): Mảng boxes với shape (N, 5) ở định dạng [xc, yc, w, h, theta]
            img_size (tuple): (width, height) của ảnh.
        
        Returns:
            tuple: (np.ndarray: Các boxes hợp lệ sau khi lọc., list: valid_indices)
        """
        valid_indices = []
        img_area = img_size[0] * img_size[1]
        for i, box in enumerate(boxes):
            xc, yc, w, h, theta = box
            # Tính kích thước thực sau khi xoay
            cos_t = np.abs(np.cos(theta))
            sin_t = np.abs(np.sin(theta))
            effective_w = w * cos_t + h * sin_t
            effective_h = w * sin_t + h * cos_t

            normalized_area = (effective_w * effective_h) / img_area
            wh_ratio = effective_w / effective_h if effective_h > 0 else 0

            # Thiết lập ngưỡng động dựa trên góc xoay
            rotation_factor = np.abs(theta) / np.pi  # từ 0 đến 1
            dynamic_ratio_min = max(0.1, self.wh_ratio_range[0] * (1 - rotation_factor))
            dynamic_ratio_max = min(10.0, self.wh_ratio_range[1] * (1 + rotation_factor))
            
            keep = True
            if normalized_area < self.min_area:
                print(f"Box {i} rejected: NormArea {normalized_area:.4f} < min {self.min_area}")
                keep = False
            elif normalized_area > self.max_area:
                print(f"Box {i} rejected: NormArea {normalized_area:.4f} > max {self.max_area}")
                keep = False
            elif wh_ratio < dynamic_ratio_min:
                print(f"Box {i} rejected: Ratio {wh_ratio:.2f} < min {dynamic_ratio_min:.2f}")
                keep = False
            elif wh_ratio > dynamic_ratio_max:
                print(f"Box {i} rejected: Ratio {wh_ratio:.2f} > max {dynamic_ratio_max:.2f}")
                keep = False
            
            if keep:
                valid_indices.append(i)
                # print(f"Box {i} kept: NormArea {normalized_area:.4f}, Ratio {wh_ratio:.2f}, Theta {np.rad2deg(theta):.1f}°")
        
        return boxes[valid_indices], valid_indices

    def smooth_boxes(self, current_boxes):
        """
        Làm mịn (smoothing) các boxes dùng EMA.
        
        Args:
            current_boxes (np.ndarray): Boxes hiện tại với shape (N,5)
        
        Returns:
            np.ndarray: Boxes đã được làm mịn.
        """
        if self.previous_boxes is None or current_boxes.shape != self.previous_boxes.shape:
            self.previous_boxes = current_boxes.copy()
            return current_boxes
        
        # Giới hạn giá trị trong khoảng [0, 1]
        current_boxes = np.clip(current_boxes, 0, 1)
        self.previous_boxes = np.clip(self.previous_boxes, 0, 1)
        
        smoothed = self.alpha * current_boxes + (1 - self.alpha) * self.previous_boxes
        self.previous_boxes = smoothed.copy()
        return smoothed

    def detect_collisions(self, obb, img_size, valid_indices=None):
        """
        Phát hiện va chạm giữa các box dựa trên thông tin OBB đã có.
        
        Args:
            obb: Đối tượng OBB từ results[0].obb, phải có thuộc tính xyxyxyxy với shape (N,8)
            img_size (tuple): (width, height) của ảnh.
        
        Returns:
            list: Danh sách các cặp index các box có va chạm.
        """
        collisions = []
        polygons = []
        
        # Lấy mảng tọa độ 8 giá trị từ obb và chuyển thành mảng đa giác 4x2
        poly_array = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, "cpu") else obb.xyxyxyxy
        for poly in poly_array:
            # Chuyển flat array (8 giá trị) thành 4 điểm (4x2)
            polygons.append(np.array(poly).reshape(4, 2))
        
        # So sánh từng cặp đa giác sử dụng thuật toán SAT
        # Chỉ kiểm tra các box hợp lệ
        if valid_indices is not None:
            polygons = [polygons[i] for i in valid_indices]
            
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                if self._sat_intersect(polygons[i], polygons[j]):
                    collisions.append((i, j))
        return collisions

    def _sat_intersect(self, poly1, poly2):
        """
        Triển khai thuật toán Separating Axis Theorem (SAT) cho 2 đa giác lồi.
        
        Args:
            poly1, poly2 (np.ndarray): Mảng với shape (4,2) chứa tọa độ các đỉnh.
        
        Returns:
            bool: True nếu có giao nhau, False nếu không.
        """
        for polygon in [poly1, poly2]:
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]
                normal = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
                
                # Chiếu các đỉnh của 2 đa giác lên đường pháp tuyến
                proj1 = poly1 @ normal
                proj2 = poly2 @ normal
                
                if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                    return False
        return True

    def convert_to_xyxy(self, obb):
        """
        Chuyển đổi OBB sang định dạng axis-aligned XYXY.
        
        Args:
            obb: Đối tượng OBB chứa thuộc tính xyxy
        
        Returns:
            np.ndarray: Mảng các box theo định dạng XYXY.
        """
        return obb.xyxy.cpu().numpy() if hasattr(obb.xyxy, "cpu") else obb.xyxy

if __name__ == "__main__":
    from results import OBB  # Import OBB từ YOLO
    processor = PostProcessor(alpha=0.4)
    
    # Tạo test data cho 3 boxes ở định dạng [xc, yc, w, h, theta]
    import numpy as np
    test_data = np.array([
        [0.5, 0.5, 0.2, 0.3, np.deg2rad(45)],
        [0.5, 0.5, 0.2, 0.3, np.deg2rad(-45)],
        [0.1, 0.1, 0.05, 0.05, 0.0]
    ])
    test_obb = OBB(test_data, orig_shape=(640, 480))
    
    # Test filter_by_geometry
    boxes = test_obb.xywhr.cpu().numpy() if hasattr(test_obb.xywhr, "cpu") else test_obb.xywhr
    filtered, valid_indices = processor.filter_by_geometry(boxes, img_size=(640, 480))
    print(f"Kept {len(filtered)}/{len(boxes)} boxes after geometry filtering")
    
    # Test collision detection
    collisions = processor.detect_collisions(test_obb, img_size=(640, 480), valid_indices=valid_indices)
    print(f"Collisions detected: {collisions}")
    
    # Test chuyển đổi sang XYXY
    xyxy_boxes = processor.convert_to_xyxy(test_obb)
    print("Converted XYXY boxes:\n", xyxy_boxes)
