import numpy as np

class PostProcessor:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.previous_boxes = None
        # Adaptive geometry constraints
        self.min_area = 0.0001    # 0.01% of image
        self.max_area = 0.95      # 95% of image
        self.wh_ratio_range = (0.2, 5.0)  # Extreme ratios for rotated objects
        self.collision_threshold = 0.35  # Higher tolerance for dense scenes

    def _get_rotated_corners(self, box, img_size):
        """Get rotated box corners using rotation matrix"""
        xc, yc, w, h, theta = box[:5]
        w *= img_size[0]
        h *= img_size[1]
        xc *= img_size[0]
        yc *= img_size[1]
        
        # Rotation matrix
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Corners relative to center
        corners = np.array([
            [-w/2, -h/2],
            [ w/2, -h/2],
            [ w/2,  h/2],
            [-w/2,  h/2]
        ])
        
        # Rotate and translate
        rot_mat = np.array([[cos_t, -sin_t],
                            [sin_t,  cos_t]])
        rotated = corners @ rot_mat.T + np.array([xc, yc])
        
        return rotated

    def filter_by_geometry(self, boxes, img_size):
        """Lọc boxes dựa trên đặc trưng hình học"""
        valid_indices = []
        for i, box in enumerate(boxes):
            # Get OBB parameters and calculate rotated dimensions
            xc, yc, w, h, theta = box[0:5]
            
            # Calculate rotated bounding box dimensions
            cos_t = np.abs(np.cos(theta))
            sin_t = np.abs(np.sin(theta))
            
            # Calculate actual width/height after rotation
            effective_w = w * cos_t + h * sin_t
            effective_h = w * sin_t + h * cos_t
            
            # Calculate normalized area and aspect ratio
            img_area = img_size[0] * img_size[1]
            normalized_area = (effective_w * effective_h) / img_area
            wh_ratio = effective_w / effective_h if effective_h > 0 else 0
            
            # Dynamic thresholds based on rotation angle
            rotation_factor = np.abs(theta)/np.pi  # 0-1
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
                print(f"Box {i} kept: NormArea {normalized_area:.4f}, Ratio {wh_ratio:.2f}, Theta {np.rad2deg(theta):.1f}°")
        
        return boxes[valid_indices]


    def convert_to_xyxy(self, boxes, img_size):
        """Convert OBB to axis-aligned XYXY encompassing rotated box"""
        xyxy_boxes = []
        for box in boxes:
            corners = self._get_rotated_corners(box, img_size)
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            xyxy_boxes.append([
                np.min(x_coords), 
                np.min(y_coords),
                np.max(x_coords),
                np.max(y_coords)
            ])
        return np.array(xyxy_boxes)

    def smooth_boxes(self, current_boxes):
        """Smooth boxes with dimensional stability checks
        Args:
            current_boxes (np.ndarray): Current frame's boxes in shape (N,5)
        Returns:
            np.ndarray: Smoothed boxes with same shape as input
        """
        if self.previous_boxes is None or \
           current_boxes.shape != self.previous_boxes.shape:
            self.previous_boxes = current_boxes.copy()
            return current_boxes
        
        # Clip values to valid range before smoothing
        current_boxes = np.clip(current_boxes, 0, 1)
        self.previous_boxes = np.clip(self.previous_boxes, 0, 1)
        
        # Vectorized EMA calculation
        smoothed = self.alpha * current_boxes + (1 - self.alpha) * self.previous_boxes
        self.previous_boxes = smoothed.copy()
        
        return smoothed

    def detect_collisions(self, boxes, img_size):
        """Detect OBB collisions using Separating Axis Theorem
        
        Args:
            boxes: List of OBB boxes in xywhr format
            img_size: Tuple of (width, height) for image dimensions
        """
        collisions = []
        polygons = [self._get_rotated_corners(b, img_size) for b in boxes]
        
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if self._sat_intersect(polygons[i], polygons[j]):
                    collisions.append((i, j))
        return collisions

    def _sat_intersect(self, poly1, poly2):
        """SAT implementation for convex polygons"""
        for polygon in [poly1, poly2]:
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i+1) % len(polygon)]
                normal = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
                
                # Project both polygons onto the normal
                proj1 = poly1 @ normal
                proj2 = poly2 @ normal
                
                # Check for overlap
                if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                    return False
        return True

if __name__ == "__main__":
    # Test with rotated boxes
    processor = PostProcessor(alpha=0.4)
    img_size = (640, 480)
    
    # Test collision detection
    boxes = np.array([
        [0.5, 0.5, 0.2, 0.3, np.deg2rad(45)],  # Rotated 45
        [0.5, 0.5, 0.2, 0.3, np.deg2rad(-45)], # Rotated -45
        [0.1, 0.1, 0.05, 0.05, 0]  # Small non-colliding
    ])
    
    print("Collisions:", processor.detect_collisions(boxes, img_size))
    
    # Test EMA with changing box counts
    boxes1 = np.random.rand(3, 5)
    print("Smooth 1:", processor.smooth_boxes(boxes1))
    boxes2 = np.random.rand(2, 5)
    print("Smooth 2 (count change):", processor.smooth_boxes(boxes2))
