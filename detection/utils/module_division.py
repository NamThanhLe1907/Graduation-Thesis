"""
Module chia pallet thành các vùng nhỏ hơn để depth estimation.
Đặt giữa YOLO detection và depth estimation trong pipeline.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class ModuleDivision:
    """
    Lớp chia pallet thành các vùng con đơn giản.
    """
    
    def __init__(self):
        """Khởi tạo Module Division."""
        pass
    
    def divide_module_1(self, pallet_bbox: List[float], layer: int = 1) -> List[Dict[str, Any]]:
        """
        Chia pallet theo Module 1:
        - Layer 1: Chia 3 phần bằng nhau theo chiều rộng (3 cột x 1 hàng)
        - Layer 2: Chia 3 phần bằng nhau theo chiều dài (1 cột x 3 hàng)
        
        Args:
            pallet_bbox: [x1, y1, x2, y2] của pallet
            layer: Lớp cần chia (1 hoặc 2)
            
        Returns:
            List[Dict]: Danh sách các vùng con với thông tin tọa độ
            [
                {
                    'region_id': int,           # ID của vùng (1, 2, 3)
                    'bbox': [x1, y1, x2, y2],   # Tọa độ vùng
                    'center': [x, y],           # Tâm vùng (để depth estimation)
                    'layer': int,               # Lớp hiện tại
                    'module': int               # Module ID
                }
            ]
        """
        if layer not in [1, 2]:
            raise ValueError("Layer phải là 1 hoặc 2")
        
        x1, y1, x2, y2 = pallet_bbox
        width = x2 - x1
        height = y2 - y1
        
        regions = []
        
        if layer == 1:
            # Layer 1: Chia 3 phần theo chiều rộng (3 cột x 1 hàng)
            part_width = width / 3
            
            for i in range(3):
                # Tính tọa độ từng vùng
                region_x1 = x1 + i * part_width
                region_y1 = y1
                region_x2 = x1 + (i + 1) * part_width
                region_y2 = y2
                
                # Tính tâm vùng
                center_x = (region_x1 + region_x2) / 2
                center_y = (region_y1 + region_y2) / 2
                
                regions.append({
                    'region_id': i + 1,  # 1, 2, 3
                    'bbox': [region_x1, region_y1, region_x2, region_y2],
                    'center': [center_x, center_y],
                    'layer': layer,
                    'module': 1
                })
        
        elif layer == 2:
            # Layer 2: Chia 3 phần theo chiều dài (1 cột x 3 hàng)
            part_height = height / 3
            
            for i in range(3):
                # Tính tọa độ từng vùng
                region_x1 = x1
                region_y1 = y1 + i * part_height
                region_x2 = x2
                region_y2 = y1 + (i + 1) * part_height
                
                # Tính tâm vùng
                center_x = (region_x1 + region_x2) / 2
                center_y = (region_y1 + region_y2) / 2
                
                regions.append({
                    'region_id': i + 1,  # 1, 2, 3
                    'bbox': [region_x1, region_y1, region_x2, region_y2],
                    'center': [center_x, center_y],
                    'layer': layer,
                    'module': 1
                })
        
        return regions
    
    def process_pallet_detections(self, detection_result: Dict[str, Any], 
                                 layer: int = 1) -> Dict[str, Any]:
        """
        Xử lý kết quả detection từ YOLO và chia thành các vùng nhỏ.
        
        Args:
            detection_result: Kết quả từ YOLO detection
            layer: Lớp cần chia
            
        Returns:
            Dict: Kết quả đã được chia với thông tin các vùng con
            {
                'original_detection': Dict,     # Kết quả YOLO gốc
                'divided_regions': List[Dict],  # Các vùng đã chia
                'total_regions': int,           # Tổng số vùng
                'processing_info': Dict         # Thông tin xử lý
            }
        """
        result = {
            'original_detection': detection_result,
            'divided_regions': [],
            'total_regions': 0,
            'processing_info': {
                'layer': layer,
                'module': 1,
                'success': False,
                'error': None
            }
        }
        
        try:
            # Lấy danh sách bounding boxes từ YOLO
            bounding_boxes = detection_result.get('bounding_boxes', [])
            
            all_regions = []
            
            # Xử lý từng pallet được detect
            for pallet_idx, pallet_bbox in enumerate(bounding_boxes):
                # Chia pallet thành các vùng nhỏ
                regions = self.divide_module_1(pallet_bbox, layer=layer)
                
                # Thêm thông tin pallet_id cho mỗi vùng
                for region in regions:
                    region['pallet_id'] = pallet_idx + 1
                    region['global_region_id'] = len(all_regions) + 1
                
                all_regions.extend(regions)
            
            result['divided_regions'] = all_regions
            result['total_regions'] = len(all_regions)
            result['processing_info']['success'] = True
            
        except Exception as e:
            result['processing_info']['error'] = str(e)
            print(f"Lỗi khi chia pallet: {e}")
        
        return result
    
    def prepare_for_depth_estimation(self, divided_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chuẩn bị dữ liệu cho depth estimation.
        
        Args:
            divided_result: Kết quả từ process_pallet_detections
            
        Returns:
            List[Dict]: Danh sách các vùng cần depth estimation
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'center': [x, y],
                    'region_info': Dict  # Thông tin chi tiết vùng
                }
            ]
        """
        depth_regions = []
        
        try:
            regions = divided_result.get('divided_regions', [])
            
            for region in regions:
                depth_regions.append({
                    'bbox': region['bbox'],
                    'center': region['center'],
                    'region_info': {
                        'region_id': region['region_id'],
                        'pallet_id': region.get('pallet_id', 1),
                        'global_region_id': region.get('global_region_id', 1),
                        'layer': region['layer'],
                        'module': region['module']
                    }
                })
        
        except Exception as e:
            print(f"Lỗi khi chuẩn bị dữ liệu cho depth: {e}")
        
        return depth_regions


# Để test module
if __name__ == "__main__":
    # Test Module Division
    divider = ModuleDivision()
    
    # Pallet mẫu
    test_pallet = [100, 100, 400, 300]  # x1, y1, x2, y2
    
    print("=== TEST MODULE DIVISION ===")
    print(f"Pallet gốc: {test_pallet}")
    print(f"Kích thước: {400-100}x{300-100} = 300x200")
    
    # Test Layer 1
    print("\n--- LAYER 1 (3 phần theo chiều rộng) ---")
    regions_layer1 = divider.divide_module_1(test_pallet, layer=1)
    for region in regions_layer1:
        bbox = region['bbox']
        center = region['center']
        print(f"Vùng {region['region_id']}: bbox={[int(x) for x in bbox]}, center=({center[0]:.1f}, {center[1]:.1f})")
    
    # Test Layer 2
    print("\n--- LAYER 2 (3 phần theo chiều dài) ---")
    regions_layer2 = divider.divide_module_1(test_pallet, layer=2)
    for region in regions_layer2:
        bbox = region['bbox']
        center = region['center']
        print(f"Vùng {region['region_id']}: bbox={[int(x) for x in bbox]}, center=({center[0]:.1f}, {center[1]:.1f})")
    
    # Test với detection result mẫu
    print("\n--- TEST VỚI DETECTION RESULT ---")
    mock_detection = {
        'bounding_boxes': [test_pallet],
        'scores': [0.9],
        'classes': [0]
    }
    
    result = divider.process_pallet_detections(mock_detection, layer=1)
    print(f"Tổng số vùng: {result['total_regions']}")
    print(f"Thành công: {result['processing_info']['success']}")
    
    # Test prepare for depth
    depth_data = divider.prepare_for_depth_estimation(result)
    print(f"\nDữ liệu cho depth estimation: {len(depth_data)} vùng")
    for i, data in enumerate(depth_data):
        info = data['region_info']
        print(f"  Vùng {i+1}: Pallet {info['pallet_id']}, Region {info['region_id']}, Layer {info['layer']}")