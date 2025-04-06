class DivisionModule:
    def __init__(self, pallet_config, dynamic_mode=True):
        """
        Nếu dynamic_mode là True, module sẽ hoạt động theo detected_box (chế độ đặc biệt).
        Nếu dynamic_mode là False, sử dụng pallet_config cố định.
        """
        self.pallet_config = pallet_config
        self.dynamic_mode = dynamic_mode  # True: chế độ đặc biệt, False: chế độ cố định
        self.margin = 1  # 1cm margin
        if not self.dynamic_mode:
            self.LR = pallet_config['LR']
            self.LQ = pallet_config['LQ']
    
    def _calculate_base_points(self):
        """Tính toán tọa độ các điểm cơ sở ABCD theo pallet_config cố định."""
        return {
            'A': (self.margin, self.margin),
            'B': (self.LR - self.margin, self.margin),
            'C': (self.LR - self.margin, self.LQ - self.margin),
            'D': (self.margin, self.LQ - self.margin)
        }
    
    def _apply_margin_to_box(self, detected_box):
        """
        Áp dụng margin vào detected_box.
        Giả sử detected_box là danh sách 4 điểm [A, B, C, D] theo thứ tự:
        A: top-left, B: top-right, C: bottom-right, D: bottom-left.
        Sau khi áp dụng margin:
          - A dịch vào trong theo ( +margin, +margin )
          - B dịch theo ( -margin, +margin )
          - C dịch theo ( -margin, -margin )
          - D dịch theo ( +margin, -margin )
        """
        A, B, C, D = detected_box
        new_A = (A[0] + self.margin, A[1] + self.margin)
        new_B = (B[0] - self.margin, B[1] + self.margin)
        new_C = (C[0] - self.margin, C[1] - self.margin)
        new_D = (D[0] + self.margin, D[1] - self.margin)
        return {'A': new_A, 'B': new_B, 'C': new_C, 'D': new_D}
    
    def _divide_edge(self, start, end, num_points):
        """Chia đều cạnh thành các điểm."""
        dx = (end[0] - start[0]) / (num_points - 1)
        dy = (end[1] - start[1]) / (num_points - 1)
        return [(start[0] + i * dx, start[1] + i * dy) for i in range(num_points)]
    
    def _create_grid(self, ab, dc):
        """Tạo lưới từ các điểm chia của cạnh AB và DC."""
        return [(ab[i], dc[j]) for i in range(len(ab)) for j in range(len(dc))]
    
    def module1_division(self, detected_box=None):
        """
        Chia pallet thành các module.
        Nếu dynamic_mode là True, bắt buộc phải truyền detected_box (danh sách 4 điểm theo thứ tự A, B, C, D).
        Nếu dynamic_mode là False, sử dụng pallet_config cố định.
        """
        if self.dynamic_mode:
            if detected_box is None:
                raise ValueError("Dynamic mode yêu cầu cung cấp detected_box.")
            points = self._apply_margin_to_box(detected_box)
        else:
            points = self._calculate_base_points()
        
        # Giả sử:
        # - Cạnh AB (hoặc A->B) là cạnh trên
        # - Cạnh DC (hoặc D->C) là cạnh dưới
        ab_points = self._divide_edge(points['A'], points['B'], 4)
        dc_points = self._divide_edge(points['D'], points['C'], 4)
        
        return {
            'AB_points': ab_points,
            'DC_points': dc_points,
            'grid': self._create_grid(ab_points, dc_points)
        }


if __name__ == "__main__":
    # Test case cho chế độ đặc biệt (dynamic mode)
    # Giả sử detected_box là: A=(1,1), B=(11,1), C=(11,9), D=(1,9)
    test_detected_box = [(1, 1), (11, 1), (11, 9), (1, 9)]
    config = {'LR': 12, 'LQ': 10}  # pallet_config không được sử dụng trong dynamic mode
    module = DivisionModule(config, dynamic_mode=True)
    divisions = module.module1_division(detected_box=test_detected_box)
    print("Các điểm trên AB:", divisions['AB_points'])
    print("Các điểm trên DC:", divisions['DC_points'])
    print("Lưới chia:", divisions['grid'])
