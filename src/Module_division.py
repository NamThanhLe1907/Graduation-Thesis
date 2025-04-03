class DivisionModule:
    def __init__(self, pallet_config):
        self.LR = pallet_config['LR']
        self.LQ = pallet_config['LQ']
        self.margin = 1  # 1cm margin
        
    def _calculate_base_points(self):
        """Tính toán tọa độ các điểm cơ sở ABCD"""
        return {
            'A': (self.margin, self.margin),
            'B': (self.LR - self.margin, self.margin),
            'C': (self.LR - self.margin, self.LQ - self.margin),
            'D': (self.margin, self.LQ - self.margin)
        }
    
    def _divide_edge(self, start, end, num_points):
        """Chia đều cạnh thành các điểm"""
        dx = (end[0] - start[0]) / (num_points - 1)
        dy = (end[1] - start[1]) / (num_points - 1)
        return [(start[0] + i*dx, start[1] + i*dy) for i in range(num_points)]
    
    def module1_division(self):
        points = self._calculate_base_points()
        ab_points = self._divide_edge(points['A'], points['B'], 4)
        cd_points = self._divide_edge(points['C'], points['D'], 4)
        
        return {
            'AB_points': ab_points,
            'CD_points': cd_points,
            'grid': self._create_grid(ab_points, cd_points)
        }
    
    def _create_grid(self, ab, cd):
        """Tạo lưới từ các điểm chia"""
        return [(ab[i], cd[j]) for i in range(4) for j in range(4)]


if __name__ == "__main__":
    # Test case với kích thước thực
    config = {'LR': 12, 'LQ': 10}
    module = DivisionModule(config)
    divisions = module.module1_division()
    print("Các điểm trên AB:", divisions['AB_points'])
