"""
Region Manager để quản lý các vùng xử lý cố định trong pipeline
"""
import numpy as np
import cv2
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

class RegionManager:
    """
    Quản lý các regions cố định để xử lý riêng biệt trong pipeline.
    Mỗi region có thể có offset và cấu hình riêng.
    """
    
    def __init__(self, auto_load_offsets: bool = True):
        """
        Khởi tạo RegionManager với các regions mặc định.
        
        Args:
            auto_load_offsets: Có tự động load offset từ file không
        """
        # Định nghĩa các regions dựa trên tọa độ người dùng cung cấp (cập nhật mới)
        self.regions = {
            'loads': {
                'name': 'loads',
                'description': 'Vùng xử lý loads (class 0, 1)',
                'polygon': [(2, 710), (2, 3), (356, 3), (356, 710)],  # điểm 1->2->8->7
                'target_classes': [0.0, 1.0],  # load, load2
                'color': (0, 255, 0),  # Xanh lá
                'offset': {'x': 0, 'y': 0},  # Offset cho robot coordinates - CẦN CHỈNH
                'enabled': True,
                'priority': 1  # Ưu tiên cao nhất cho loads
            },
            'pallets1': {
                'name': 'pallets1', 
                'description': 'Vùng xử lý pallets 1 (class 2)',
                'polygon': [(821, 710), (821, 3), (356, 3), (356, 710)],  # điểm 1->2->8->7 (cùng vùng với loads)
                'target_classes': [0.0,1.0,2.0],  # pallet
                'color': (255, 0, 0),  # Đỏ
                'offset': {'x': 0, 'y': 0},  # Offset cho robot coordinates - ĐÃ CHỈNH
                'enabled': True,
                'priority': 2
            },
            'pallets2': {
                'name': 'pallets2',
                'description': 'Vùng xử lý pallets 2 (class 2)', 
                'polygon': [(821, 710), (821, 3), (1272, 3), (1272, 710)],  # điểm 1->2->6->5
                'target_classes': [0.0,1.0,2.0],  # pallet
                'color': (0, 0, 255),  # Xanh dương
                'offset': {'x': 0, 'y': 0},  # Offset cho robot coordinates - CẦN CHỈNH
                'enabled': True,
                'priority': 3
            }
        }
        
        # Tự động load offset từ file nếu được yêu cầu
        if auto_load_offsets:
            self.auto_load_offsets()
    
    def is_point_in_region(self, point: Tuple[float, float], region_name: str) -> bool:
        """
        Kiểm tra xem một điểm có nằm trong region không.
        
        Args:
            point: Tọa độ điểm (x, y)
            region_name: Tên region để kiểm tra
            
        Returns:
            bool: True nếu điểm nằm trong region
        """
        if region_name not in self.regions:
            return False
        
        region = self.regions[region_name]
        if not region['enabled']:
            return False
        
        polygon = region['polygon']
        x, y = point
        
        # Sử dụng OpenCV để kiểm tra point in polygon
        contour = np.array(polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(contour, (float(x), float(y)), False)
        return result >= 0
    
    def get_region_for_detection(self, detection_center: Tuple[float, float], 
                               detection_class: float) -> Optional[str]:
        """
        Tìm region phù hợp cho một detection dựa trên vị trí và class.
        
        Args:
            detection_center: Tâm của detection (x, y)
            detection_class: Class của detection
            
        Returns:
            str: Tên region hoặc None nếu không thuộc region nào
        """
        # Sắp xếp regions theo priority để ưu tiên region có priority cao hơn
        sorted_regions = sorted(self.regions.items(), 
                              key=lambda x: x[1]['priority'])
        
        for region_name, region_info in sorted_regions:
            # Kiểm tra class có phù hợp không
            if detection_class in region_info['target_classes']:
                # Kiểm tra vị trí có trong region không
                if self.is_point_in_region(detection_center, region_name):
                    return region_name
        
        return None
    
    def filter_detections_by_regions(self, detections: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter detections dựa trên regions và group theo regions.
        
        Args:
            detections: Kết quả detection từ YOLO
            
        Returns:
            Dict: Detections đã được filter và group theo regions
        """
        if 'bounding_boxes' not in detections or not detections['bounding_boxes']:
            return {
                'original': detections,
                'regions': {},
                'unassigned': detections.copy()
            }
        
        bboxes = detections['bounding_boxes']
        classes = detections.get('classes', [])
        scores = detections.get('scores', [])
        corners_list = detections.get('corners', [])
        
        # Khởi tạo kết quả
        result = {
            'original': detections,
            'regions': {},
            'unassigned': {
                'bounding_boxes': [],
                'classes': [],
                'scores': [],
                'corners': []
            }
        }
        
        # Khởi tạo regions
        for region_name in self.regions.keys():
            result['regions'][region_name] = {
                'region_info': self.regions[region_name],
                'bounding_boxes': [],
                'classes': [],
                'scores': [],
                'corners': [],
                'indices': []  # Lưu indices gốc
            }
        
        # Phân loại từng detection
        for i, bbox in enumerate(bboxes):
            # Tính center của bbox
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Lấy class
            detection_class = classes[i] if i < len(classes) else 0.0
            
            # Tìm region phù hợp
            assigned_region = self.get_region_for_detection((center_x, center_y), detection_class)
            
            if assigned_region:
                # Thêm vào region
                region_data = result['regions'][assigned_region]
                region_data['bounding_boxes'].append(bbox)
                region_data['classes'].append(detection_class)
                region_data['indices'].append(i)
                
                if i < len(scores):
                    region_data['scores'].append(scores[i])
                
                if i < len(corners_list):
                    region_data['corners'].append(corners_list[i])
            else:
                # Thêm vào unassigned
                result['unassigned']['bounding_boxes'].append(bbox)
                result['unassigned']['classes'].append(detection_class)
                
                if i < len(scores):
                    result['unassigned']['scores'].append(scores[i])
                
                if i < len(corners_list):
                    result['unassigned']['corners'].append(corners_list[i])
        
        return result
    
    def apply_region_offset(self, coordinates: Dict[str, float], region_name: str) -> Dict[str, float]:
        """
        Áp dụng offset cho tọa độ robot dựa trên region.
        
        Args:
            coordinates: Tọa độ robot {'x': x, 'y': y}
            region_name: Tên region
            
        Returns:
            Dict: Tọa độ đã áp dụng offset
        """
        if region_name not in self.regions:
            return coordinates
        
        region = self.regions[region_name]
        offset = region['offset']
        
        return {
            'x': coordinates['x'] + offset['x'],
            'y': coordinates['y'] + offset['y']
        }
    
    def set_region_offset(self, region_name: str, offset_x: float, offset_y: float):
        """
        Đặt offset cho một region.
        
        Args:
            region_name: Tên region
            offset_x: Offset theo trục X
            offset_y: Offset theo trục Y
        """
        if region_name in self.regions:
            self.regions[region_name]['offset'] = {'x': offset_x, 'y': offset_y}
            print(f"[RegionManager] Đã đặt offset cho {region_name}: X={offset_x}, Y={offset_y}")
    
    def enable_region(self, region_name: str, enabled: bool = True):
        """
        Bật/tắt một region.
        
        Args:
            region_name: Tên region
            enabled: True để bật, False để tắt
        """
        if region_name in self.regions:
            self.regions[region_name]['enabled'] = enabled
            status = "bật" if enabled else "tắt"
            print(f"[RegionManager] Đã {status} region {region_name}")
    
    def draw_regions(self, image: np.ndarray, show_labels: bool = True) -> np.ndarray:
        """
        Vẽ các regions lên ảnh.
        
        Args:
            image: Ảnh để vẽ
            show_labels: Có hiển thị labels không
            
        Returns:
            np.ndarray: Ảnh đã vẽ regions
        """
        result_image = image.copy()
        
        for region_name, region_info in self.regions.items():
            if not region_info['enabled']:
                continue
            
            polygon = region_info['polygon']
            color = region_info['color']
            
            # Vẽ polygon
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result_image, [pts], True, color, 2)
            
            # Tô màu nhạt bên trong (alpha blending)
            if len(polygon) >= 3:
                overlay = result_image.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(result_image, 0.8, overlay, 0.2, 0, result_image)
            
            # Vẽ label nếu cần
            if show_labels:
                # Tính center của polygon để đặt text
                center_x = int(np.mean([p[0] for p in polygon]))
                center_y = int(np.mean([p[1] for p in polygon]))
                
                # Vẽ background cho text
                text = f"{region_info['name']}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(result_image,
                            (center_x - text_size[0]//2 - 5, center_y - text_size[1]//2 - 5),
                            (center_x + text_size[0]//2 + 5, center_y + text_size[1]//2 + 5),
                            (0, 0, 0), -1)
                
                # Vẽ text
                cv2.putText(result_image, text,
                          (center_x - text_size[0]//2, center_y + text_size[1]//2),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image
    
    def get_region_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin tất cả regions.
        
        Returns:
            Dict: Thông tin regions
        """
        return {
            'total_regions': len(self.regions),
            'enabled_regions': len([r for r in self.regions.values() if r['enabled']]),
            'regions': self.regions
        }
    
    def update_region_polygon(self, region_name: str, new_polygon: List[Tuple[int, int]]):
        """
        Cập nhật polygon cho một region.
        
        Args:
            region_name: Tên region
            new_polygon: Polygon mới
        """
        if region_name in self.regions:
            self.regions[region_name]['polygon'] = new_polygon
            print(f"[RegionManager] Đã cập nhật polygon cho {region_name}")
    
    def load_offsets_from_file(self, file_path: str = "region_offsets.json") -> bool:
        """
        Load offset từ file JSON.
        
        Args:
            file_path: Đường dẫn tới file JSON
            
        Returns:
            bool: True nếu load thành công
        """
        try:
            if not os.path.exists(file_path):
                print(f"[RegionManager] File {file_path} không tồn tại, sử dụng offset mặc định")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'regions' not in data:
                print(f"[RegionManager] Format file không hợp lệ, thiếu key 'regions'")
                return False
            
            regions_data = data['regions']
            loaded_count = 0
            
            for region_name in self.regions.keys():
                if region_name in regions_data:
                    region_offset = regions_data[region_name]
                    offset_x = region_offset.get('offset_x', 0.0)
                    offset_y = region_offset.get('offset_y', 0.0)
                    
                    self.regions[region_name]['offset'] = {
                        'x': float(offset_x),
                        'y': float(offset_y)
                    }
                    loaded_count += 1
                    # Chỉ log khi load lần đầu, tránh spam logs
                    if not hasattr(self, '_offsets_logged'):
                        print(f"[RegionManager] Loaded offset cho {region_name}: X={offset_x}, Y={offset_y}")
            
            # Log tổng kết và đánh dấu đã load
            if not hasattr(self, '_offsets_logged'):
                print(f"[RegionManager] Đã load {loaded_count}/{len(self.regions)} offset từ {file_path}")
                self._offsets_logged = True
            return True
            
        except Exception as e:
            print(f"[RegionManager] Lỗi khi load offset từ {file_path}: {e}")
            return False
    
    def save_offsets_to_file(self, file_path: str = "region_offsets.json") -> bool:
        """
        Save offset hiện tại ra file JSON.
        
        Args:
            file_path: Đường dẫn tới file JSON
            
        Returns:
            bool: True nếu save thành công
        """
        try:
            # Tạo data structure để save
            data = {
                "description": "Cấu hình offset cho các regions - Auto generated",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "regions": {},
                "instructions": {
                    "how_to_use": "Chỉnh sửa offset_x và offset_y cho từng region theo nhu cầu",
                    "coordinate_system": "X+ hướng phải, Y+ hướng xuống", 
                    "units": "pixels hoặc meters tùy theo hệ thống robot coordinates"
                }
            }
            
            # Thêm thông tin offset cho từng region
            for region_name, region_info in self.regions.items():
                offset = region_info['offset']
                data['regions'][region_name] = {
                    "offset_x": offset['x'],
                    "offset_y": offset['y'],
                    "note": f"Offset cho region {region_name} ({region_info['description']})"
                }
            
            # Ghi ra file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"[RegionManager] Đã lưu offset ra {file_path}")
            return True
            
        except Exception as e:
            print(f"[RegionManager] Lỗi khi lưu offset ra {file_path}: {e}")
            return False
    
    def auto_load_offsets(self):
        """
        Tự động load offset từ file nếu có.
        Được gọi khi khởi tạo RegionManager.
        """
        default_files = ["region_offsets.json", "config/region_offsets.json"]
        
        for file_path in default_files:
            if os.path.exists(file_path):
                # Chỉ log lần đầu tìm thấy file
                if not hasattr(self, '_file_found_logged'):
                    print(f"[RegionManager] Tìm thấy file offset: {file_path}")
                    self._file_found_logged = True
                if self.load_offsets_from_file(file_path):
                    return True
        
        print(f"[RegionManager] Không tìm thấy file offset, sử dụng giá trị mặc định")
        return False 