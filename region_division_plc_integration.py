"""
Integration Module cho Region Division và PLC Communication
Chia pallets thành regions và gửi tọa độ Px, Py của các regions vào PLC thông qua DB26.

Chức năng:
1. Phát hiện pallets bằng YOLO
2. Chia pallets thành 3 regions sử dụng Module Division
3. Truyền tọa độ robot (Px, Py) của các regions vào PLC DB26
4. Monitoring và logging
"""
import cv2
import time
import os
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from plc_communication import DB26Communication
from detection import (YOLOTensorRT, ModuleDivision)
from detection.utils.robot_coordinate_transform import RobotCoordinateTransform
from detection.utils.region_manager import RegionManager

class RegionDivisionPLCIntegration:
    """
    Class chính tích hợp Region Division với PLC Communication
    """
    
    def __init__(self, 
                 plc_ip: str = "192.168.0.1", 
                 plc_rack: int = 0, 
                 plc_slot: int = 1,
                 debug: bool = True):
        """
        Args:
            plc_ip: IP address của PLC
            plc_rack: Rack number của PLC
            plc_slot: Slot number của PLC
            debug: Bật chế độ debug
        """
        self.debug = debug
        
        # Khởi tạo các component
        self.module_divider = ModuleDivision(debug=debug)
        self.robot_transformer = RobotCoordinateTransform()
        self.region_manager = RegionManager(auto_load_offsets=True)
        self.plc_comm = DB26Communication(plc_ip, plc_rack, plc_slot)
        
        # DB26 Memory Layout cho Region Coordinates (Theo yêu cầu mới):
        # loads: DB26.0 (Px), DB26.4 (Py) 
        # pallets1: DB26.12 (Px), DB26.16 (Py)
        # pallets2: DB26.24 (Px), DB26.28 (Py)
        self.db26_offsets = {
            'loads': {'px': 0, 'py': 4},
            'pallets1': {'px': 12, 'py': 16}, 
            'pallets2': {'px': 24, 'py': 28}
        }
        
        # Tracking data
        self.last_regions_data = []
        self.plc_connected = False
        
        # ⭐ BAG PALLET TRACKING SYSTEM ⭐
        # Theo dõi riêng biệt từng region để dễ trích xuất
        self.bag_pallet_1 = 1  # Default tracking cho pallets1
        self.bag_pallet_2 = 1  # Default tracking cho pallets2  
        self.current_region_data = {
            'loads': None,      # Lưu data region loads
            'pallets1': None,   # Lưu data region pallets1
            'pallets2': None    # Lưu data region pallets2
        }
        
        # ⭐ BAG SEQUENCE SETTINGS ⭐
        self.current_bag_number = 1  # Bag hiện tại đang đặt (1, 2, hoặc 3)
        self.bag_to_region_mapping = {
            1: 1,  # bao 1 → region_id 1 (R1) 
            2: 3,  # bao 2 → region_id 3 (R3)
            3: 2   # bao 3 → region_id 2 (R2, center, cuối cùng)
        }
        
        if self.debug:
            print(f"[RegionPLC] Khởi tạo với PLC: {plc_ip}:{plc_rack}:{plc_slot}")
            print(f"[RegionPLC] DB26 Memory Layout (Updated):")
            for region_name, offsets in self.db26_offsets.items():
                print(f"  {region_name}: Px=DB26.{offsets['px']}, Py=DB26.{offsets['py']}")
    
    def connect_plc(self) -> bool:
        """
        Kết nối đến PLC
        
        Returns:
            bool: True nếu kết nối thành công
        """
        if self.debug:
            print(f"[RegionPLC] Đang kết nối PLC...")
        
        self.plc_connected = self.plc_comm.connect()
        
        if self.plc_connected:
            if self.debug:
                print(f"[RegionPLC] ✅ Kết nối PLC thành công!")
        else:
            if self.debug:
                print(f"[RegionPLC] ❌ Kết nối PLC thất bại!")
        
        return self.plc_connected
    
    def disconnect_plc(self):
        """Ngắt kết nối PLC"""
        if self.plc_connected:
            self.plc_comm.disconnect()
            self.plc_connected = False
            if self.debug:
                print(f"[RegionPLC] PLC đã ngắt kết nối")
    
    def process_detection_and_divide_regions(self, detections: Dict[str, Any], layer: int = 1) -> List[Dict]:
        """
        Xử lý detection và chia pallets thành regions
        
        Args:
            detections: Kết quả detection từ YOLO
            layer: Layer để chia (1 hoặc 2)
            
        Returns:
            List[Dict]: Danh sách regions với robot coordinates
        """
        if self.debug:
            print(f"\n[RegionPLC] Bắt đầu xử lý detection và chia regions (Layer {layer})")
        
        # Chỉ xử lý pallets (class 2.0)
        pallet_classes = [2.0]
        
        # Chia pallets thành regions sử dụng Module Division
        divided_result = self.module_divider.process_pallet_detections(
            detections, 
            layer=layer, 
            target_classes=pallet_classes
        )
        
        # Chuẩn bị dữ liệu regions
        regions_data = self.module_divider.prepare_for_depth_estimation(divided_result)
        
        if self.debug:
            print(f"[RegionPLC] Đã chia được {len(regions_data)} regions")
        
        # ⭐ RESET CURRENT REGION DATA ⭐
        self.current_region_data = {
            'loads': None,
            'pallets1': None,
            'pallets2': None
        }
        
        # Chuyển đổi pixel coordinates sang robot coordinates cho từng region
        regions_with_robot_coords = []
        
        for i, region_data in enumerate(regions_data):
            center_pixel = region_data['center']
            region_info = region_data['region_info']
            
            # Chuyển đổi pixel center sang robot coordinates  
            robot_x, robot_y = self.robot_transformer.camera_to_robot(
                center_pixel[0], center_pixel[1]
            )
            
            # ⭐ ÁP DỤNG OFFSET TỪ REGIONMANAGER ⭐
            # Xác định region name để áp dụng offset phù hợp
            region_name = self._determine_region_name(center_pixel, region_info)
            
            if region_name:
                # Áp dụng offset từ RegionManager
                robot_coords_with_offset = self.region_manager.apply_region_offset(
                    {'x': robot_x, 'y': robot_y}, 
                    region_name
                )
                robot_x = robot_coords_with_offset['x']
                robot_y = robot_coords_with_offset['y']
                
                if self.debug:
                    offset = self.region_manager.regions[region_name]['offset']
                    print(f"    Áp dụng offset {region_name}: X+{offset['x']}, Y+{offset['y']}")
            
            # Tạo region data với robot coordinates
            region_with_coords = {
                'region_id': region_info['region_id'],
                'pallet_id': region_info['pallet_id'], 
                'global_region_id': region_info.get('global_region_id', i+1),
                'layer': layer,
                'pixel_center': center_pixel,
                'robot_coordinates': {
                    'px': round(robot_x, 2),
                    'py': round(robot_y, 2)
                },
                'bbox': region_data['bbox'],
                'corners': region_data.get('corners', []),
                'applied_region_offset': region_name  # Lưu tên region để debug
            }
            
            # ⭐ IMPROVED BAG PALLET TRACKING LOGIC ⭐
            if region_name == 'pallets1':
                # Có P{pallet_id}R{region_id} ở pallets1 → update bag_pallet_1
                self.bag_pallet_1 = region_info['pallet_id']
                # ⭐ STRATEGY: Sử dụng center region (R2) làm representative ⭐
                if region_info['region_id'] == 2:  # R2 là center region
                    self.current_region_data['pallets1'] = region_with_coords
                    if self.debug:
                        print(f"    📦 bag_pallet_1 = {self.bag_pallet_1} (Representative: P{region_info['pallet_id']}R{region_info['region_id']} ở pallets1)")
                elif not self.current_region_data['pallets1']:  # Fallback nếu chưa có R2
                    self.current_region_data['pallets1'] = region_with_coords
                    if self.debug:
                        print(f"    📦 bag_pallet_1 = {self.bag_pallet_1} (Fallback: P{region_info['pallet_id']}R{region_info['region_id']} ở pallets1)")
                    
            elif region_name == 'pallets2':
                # Có P{pallet_id}R{region_id} ở pallets2 → update bag_pallet_2 
                self.bag_pallet_2 = region_info['pallet_id']
                # ⭐ STRATEGY: Sử dụng center region (R2) làm representative ⭐
                if region_info['region_id'] == 2:  # R2 là center region
                    self.current_region_data['pallets2'] = region_with_coords
                    if self.debug:
                        print(f"    📦 bag_pallet_2 = {self.bag_pallet_2} (Representative: P{region_info['pallet_id']}R{region_info['region_id']} ở pallets2)")
                elif not self.current_region_data['pallets2']:  # Fallback nếu chưa có R2
                    self.current_region_data['pallets2'] = region_with_coords
                    if self.debug:
                        print(f"    📦 bag_pallet_2 = {self.bag_pallet_2} (Fallback: P{region_info['pallet_id']}R{region_info['region_id']} ở pallets2)")
                    
            elif region_name == 'loads':
                # Update loads region data - sử dụng bất kỳ region nào ở loads workspace
                if not self.current_region_data['loads']:  # Chỉ update nếu chưa có
                    self.current_region_data['loads'] = region_with_coords
                    if self.debug:
                        print(f"    📦 loads region updated (P{region_info['pallet_id']}R{region_info['region_id']} ở loads)")
            
            regions_with_robot_coords.append(region_with_coords)
            
            if self.debug:
                print(f"  Region {region_info['region_id']} (Pallet {region_info['pallet_id']}): "
                      f"Pixel({center_pixel[0]:.1f}, {center_pixel[1]:.1f}) → "
                      f"Robot(Px={robot_x:.2f}, Py={robot_y:.2f})")
        
        self.last_regions_data = regions_with_robot_coords
        return regions_with_robot_coords
    
    def _determine_region_name(self, center_pixel: Tuple[float, float], region_info: Dict) -> Optional[str]:
        """
        Xác định tên region trong RegionManager dựa trên vị trí pixel và thông tin region
        
        Args:
            center_pixel: Tọa độ pixel center của region
            region_info: Thông tin region từ module division
            
        Returns:
            str: Tên region trong RegionManager hoặc None
        """
        # ⭐ IMPROVED LOGIC: Dựa trên pallet_id và vị trí pixel ⭐
        pallet_id = region_info.get('pallet_id', 0)
        region_id = region_info.get('region_id', 0)
        
        if self.debug:
            print(f"    🔍 Mapping P{pallet_id}R{region_id} at pixel({center_pixel[0]:.1f}, {center_pixel[1]:.1f})")
        
        # ⭐ CHIẾN LƯỢC 1: Dựa trên vị trí pixel (PRIMARY) ⭐
        # Sử dụng RegionManager để xác định region dựa trên vị trí
        for region_name in self.region_manager.regions.keys():
            if self.region_manager.is_point_in_region(center_pixel, region_name):
                if self.debug:
                    print(f"    ✅ Pixel mapping: P{pallet_id}R{region_id} → {region_name}")
                return region_name
        
        # ⭐ CHIẾN LƯỢC 2: Dựa trên pallet position heuristic (FALLBACK) ⭐
        # Nếu không tìm thấy pixel mapping, dùng heuristic dựa trên vị trí X
        x_position = center_pixel[0]
        
        # Giả định layout: loads ở giữa, pallets1 ở trái, pallets2 ở phải
        # (Có thể cần điều chỉnh theo camera setup thực tế)
        if x_position < 400:  # Vùng trái
            mapped_region = 'pallets1'
        elif x_position > 800:  # Vùng phải
            mapped_region = 'pallets2'
        else:  # Vùng giữa
            mapped_region = 'loads'
        
        if self.debug:
            print(f"    🔄 Heuristic mapping: P{pallet_id}R{region_id} (X={x_position:.0f}) → {mapped_region}")
        
        return mapped_region
    
    def send_regions_to_plc(self, regions_data: List[Dict] = None) -> bool:
        """
        Gửi tọa độ robot của các regions vào PLC DB26 theo BAG PALLET TRACKING
        
        Args:
            regions_data: DEPRECATED - Không sử dụng, chỉ để backward compatibility
            
        Returns:
            bool: True nếu gửi thành công tất cả regions
        """
        if not self.plc_connected:
            if self.debug:
                print(f"[RegionPLC] ❌ PLC chưa kết nối, bỏ qua gửi dữ liệu")
            return False
        
        # ⭐ SỬ DỤNG BAG PALLET TRACKING THAY VÌ regions_data ⭐
        regions_to_send = []
        
        # Chỉ gửi các region có dữ liệu thực tế
        for region_name, region_data in self.current_region_data.items():
            if region_data is not None:
                regions_to_send.append((region_name, region_data))
        
        if not regions_to_send:
            if self.debug:
                print(f"[RegionPLC] Không có region data để gửi (current_region_data trống)")
            return False
        
        if self.debug:
            print(f"[RegionPLC] Đang gửi {len(regions_to_send)} regions vào PLC theo BAG PALLET TRACKING...")
            print(f"  📦 bag_pallet_1={self.bag_pallet_1}, bag_pallet_2={self.bag_pallet_2}")
        
        success_count = 0
        total_writes = 0
        
        # Gửi tọa độ theo từng region riêng biệt
        for region_name, region_data in regions_to_send:
            if region_name in self.db26_offsets:
                offsets = self.db26_offsets[region_name]
                robot_coords = region_data['robot_coordinates']
                region_info = region_data
                
                px = robot_coords['px']
                py = robot_coords['py']
                
                # Ghi Px
                px_success = self.plc_comm.write_db26_real(offsets['px'], px)
                total_writes += 1
                if px_success:
                    success_count += 1
                
                # Ghi Py
                py_success = self.plc_comm.write_db26_real(offsets['py'], py)
                total_writes += 1
                if py_success:
                    success_count += 1
                
                if self.debug:
                    px_status = "✅" if px_success else "❌"
                    py_status = "✅" if py_success else "❌"
                    pallet_id = region_info.get('pallet_id', '?')
                    region_id = region_info.get('region_id', '?')
                    
                    print(f"  [{region_name}] P{pallet_id}R{region_id}: {px_status} Px={px:7.2f} (DB26.{offsets['px']}), "
                          f"{py_status} Py={py:7.2f} (DB26.{offsets['py']})")
            else:
                if self.debug:
                    print(f"  ⚠️  Bỏ qua region không có offset: {region_name}")
        
        # Thành công nếu tất cả writes đều OK
        all_success = success_count == total_writes
        
        if self.debug:
            if all_success:
                print(f"[RegionPLC] ✅ Đã gửi thành công tất cả {total_writes} giá trị vào PLC")
            else:
                print(f"[RegionPLC] ❌ Chỉ gửi thành công {success_count}/{total_writes} giá trị")
        
        return all_success
    
    def read_regions_from_plc(self) -> Dict[str, Dict]:
        """
        Đọc lại tọa độ regions từ PLC để verify
        
        Returns:
            Dict: Tọa độ đọc được từ PLC
        """
        if not self.plc_connected:
            return {}
        
        plc_data = {}
        
        for region_name, offsets in self.db26_offsets.items():
            px = self.plc_comm.read_db26_real(offsets['px'])
            py = self.plc_comm.read_db26_real(offsets['py'])
            
            plc_data[region_name] = {
                'px': px,
                'py': py,
                'px_offset': offsets['px'],
                'py_offset': offsets['py']
            }
        
        return plc_data
    
    def process_detection_and_send_to_plc(self, detections: Dict[str, Any], layer: int = 1) -> Tuple[List[Dict], bool]:
        """
        Xử lý detection hoàn chỉnh: chia regions và gửi vào PLC theo BAG PALLET TRACKING
        ⭐ FIXED: Sử dụng robot coordinates từ pipeline thay vì tự tính ⭐
        
        Args:
            detections: Kết quả detection từ YOLO (phải có robot_coordinates từ pipeline)
            layer: Layer để chia
            
        Returns:
            Tuple[List[Dict], bool]: (regions_data, send_success)
        """
        # ⭐ STEP 1: Lấy robot coordinates từ pipeline (ĐÚNG) ⭐
        pipeline_robot_coords = detections.get('robot_coordinates', [])
        if not pipeline_robot_coords:
            if self.debug:
                print(f"[RegionPLC] ❌ Không có robot_coordinates từ pipeline!")
            return [], False
        
        # Bước 2: Chia regions sử dụng Module Division (chỉ để có region structure)
        regions_data = self.process_detection_and_divide_regions(detections, layer)
        
        # ⭐ STEP 3: MAP pipeline robot coords với regions ⭐ 
        # Thay vì sử dụng robot coords từ regions_data (SAI), 
        # sử dụng robot coords từ pipeline (ĐÚNG)
        self._map_pipeline_coords_to_regions(pipeline_robot_coords, regions_data)
        
        # Bước 4: Gửi vào PLC theo bag pallet tracking
        send_success = self.send_regions_to_plc()
        
        return regions_data, send_success
    
    def _map_pipeline_coords_to_regions(self, pipeline_robot_coords: List[Dict], regions_data: List[Dict]):
        """
        ⭐ MAP robot coordinates từ pipeline vào region system ⭐
        
        Args:
            pipeline_robot_coords: Robot coordinates từ pipeline (ĐÚNG)
            regions_data: Regions từ module division
        """
        if self.debug:
            print(f"[RegionPLC] 🔗 Mapping {len(pipeline_robot_coords)} pipeline coords với {len(regions_data)} regions...")
        
        # ⭐ RESET CURRENT REGION DATA ⭐
        self.current_region_data = {
            'loads': None,
            'pallets1': None,
            'pallets2': None
        }
        
        # ⭐ STRATEGY: Sử dụng trực tiếp pipeline robot coordinates với region assignment ⭐
        for coord in pipeline_robot_coords:
            assigned_region = coord.get('assigned_region')  # Region từ pipeline 
            robot_pos = coord['robot_coordinates']  # Robot coordinates ĐÚNG từ pipeline
            
            if assigned_region and assigned_region in self.current_region_data:
                # ⭐ DETERMINE REGION_ID BY BAG POSITION ⭐
                target_region_id = self.bag_to_region_mapping.get(self.current_bag_number, 1)
                
                # Tạo region data với coordinates ĐÚNG từ pipeline
                region_with_coords = {
                    'region_id': target_region_id,  # ⭐ FIXED: Map theo bag number thay vì hard-code R2 ⭐
                    'pallet_id': 1 if assigned_region == 'pallets1' else (2 if assigned_region == 'pallets2' else 1),
                    'robot_coordinates': {
                        'px': robot_pos['x'],  # ⭐ SỬ DỤNG PIPELINE COORDS (ĐÚNG) ⭐
                        'py': robot_pos['y']   # ⭐ SỬ DỤNG PIPELINE COORDS (ĐÚNG) ⭐
                    },
                    'pixel_center': [coord['camera_pixel']['x'], coord['camera_pixel']['y']],
                    'class': coord['class'],
                    'bag_number': self.current_bag_number,  # ⭐ NEW: Track bag number ⭐
                    'sequence_mapping': f"bao {self.current_bag_number} → R{target_region_id}"  # ⭐ DEBUG INFO ⭐
                }
                
                # Update BAG PALLET TRACKING theo region
                if assigned_region == 'pallets1':
                    self.bag_pallet_1 = region_with_coords['pallet_id']
                    self.current_region_data['pallets1'] = region_with_coords
                    if self.debug:
                        print(f"    📦 bag_pallet_1 = {self.bag_pallet_1} (Pipeline: {coord['class']} ở pallets1)")
                        print(f"    🎯 BAG MAPPING: {region_with_coords['sequence_mapping']}")
                elif assigned_region == 'pallets2':
                    self.bag_pallet_2 = region_with_coords['pallet_id'] 
                    self.current_region_data['pallets2'] = region_with_coords
                    if self.debug:
                        print(f"    📦 bag_pallet_2 = {self.bag_pallet_2} (Pipeline: {coord['class']} ở pallets2)")
                        print(f"    🎯 BAG MAPPING: {region_with_coords['sequence_mapping']}")
                elif assigned_region == 'loads':
                    self.current_region_data['loads'] = region_with_coords
                    if self.debug:
                        print(f"    📦 loads region updated (Pipeline: {coord['class']} ở loads)")
                        print(f"    🎯 BAG MAPPING: {region_with_coords['sequence_mapping']}")
                
                if self.debug:
                    print(f"    ✅ Mapped {coord['class']}: [{assigned_region}] P{region_with_coords['pallet_id']}R{region_with_coords['region_id']} → Px={robot_pos['x']:.2f}, Py={robot_pos['y']:.2f}")
            else:
                if self.debug:
                    print(f"    ⚠️ Skipped {coord['class']}: No region assignment")
    
    def set_current_bag_number(self, bag_number: int):
        """
        ⭐ NEW: Set current bag number để map đúng region theo sequence ⭐
        
        Args:
            bag_number: Bag number (1, 2, hoặc 3)
        """
        if bag_number in self.bag_to_region_mapping:
            old_bag = self.current_bag_number
            self.current_bag_number = bag_number
            target_region = self.bag_to_region_mapping[bag_number]
            
            if self.debug:
                print(f"🎯 [BAG CONTROL] Switched: bao {old_bag} → bao {bag_number} (maps to R{target_region})")
                print(f"   Sequence mapping: {self.bag_to_region_mapping}")
        else:
            if self.debug:
                print(f"❌ [BAG CONTROL] Invalid bag number: {bag_number}. Valid: {list(self.bag_to_region_mapping.keys())}")
    
    def get_current_bag_info(self) -> Dict[str, Any]:
        """
        ⭐ NEW: Get current bag information ⭐
        
        Returns:
            Dict: Current bag info và mapping
        """
        target_region = self.bag_to_region_mapping.get(self.current_bag_number, 1)
        return {
            'current_bag_number': self.current_bag_number,
            'target_region_id': target_region,
            'sequence_mapping': f"bao {self.current_bag_number} → R{target_region}",
            'all_mappings': self.bag_to_region_mapping.copy()
        }
    
    def get_bag_pallet_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của bag pallet tracking
        
        Returns:
            Dict: Thông tin bag pallet status
        """
        status = {
            'bag_pallet_1': self.bag_pallet_1,
            'bag_pallet_2': self.bag_pallet_2,
            'current_regions': {},
            'active_regions_count': 0,
            'current_bag_info': self.get_current_bag_info()  # ⭐ NEW: Add bag info ⭐
        }
        
        for region_name, region_data in self.current_region_data.items():
            if region_data is not None:
                status['current_regions'][region_name] = {
                    'pallet_id': region_data.get('pallet_id'),
                    'region_id': region_data.get('region_id'), 
                    'robot_coords': region_data.get('robot_coordinates'),
                    'pixel_center': region_data.get('pixel_center'),
                    'bag_number': region_data.get('bag_number'),  # ⭐ NEW: Add bag info ⭐
                    'sequence_mapping': region_data.get('sequence_mapping')  # ⭐ NEW: Add mapping info ⭐
                }
                status['active_regions_count'] += 1
            else:
                status['current_regions'][region_name] = None
        
        return status
    
    # ⭐ ENHANCED VISUALIZATION WITH COMPLETED REGIONS ⭐
    def create_visualization(self, image: np.ndarray, regions_data: List[Dict] = None, 
                           completed_regions: List[str] = None) -> np.ndarray:
        """
        ⭐ ENHANCED: Tạo visualization cho regions với robot coordinates + completed status ⭐
        
        Args:
            image: Ảnh gốc
            regions_data: Danh sách regions với robot coordinates (optional)
            completed_regions: List tên regions đã hoàn thành (e.g. ['P1R1', 'P1R2'])
            
        Returns:
            np.ndarray: Ảnh đã vẽ visualization
        """
        result_image = image.copy()
        
        # Sử dụng last_regions_data nếu không có regions_data
        if regions_data is None:
            regions_data = self.last_regions_data
        
        if not regions_data:
            # Vẽ status cơ bản nếu không có regions
            cv2.putText(result_image, "Waiting for regions...", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return result_image
        
        # ⭐ ENHANCED: Màu sắc cho regions với completed status ⭐
        def get_region_color(region_data, completed_regions):
            """Get color based on completion status"""
            if completed_regions:
                region_id = region_data.get('region_id', 0)
                pallet_id = region_data.get('pallet_id', 0)
                region_name = f"P{pallet_id}R{region_id}"
                
                if region_name in completed_regions:
                    return (0, 255, 0)  # 🟢 Xanh lá - Hoàn thành
                else:
                    return (0, 255, 255)  # 🟡 Vàng - Đang chờ/Đang xử lý
            else:
                # Default colors khi không có completed info
                return [(0, 255, 0), (255, 0, 0), (0, 0, 255)][region_data.get('region_id', 1) % 3]
        
        # Vẽ từng region
        for i, region_data in enumerate(regions_data):
            color = get_region_color(region_data, completed_regions)
            
            # Vẽ region boundary
            if 'corners' in region_data and region_data['corners']:
                corners = region_data['corners']
                pts = np.array(corners, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(result_image, [pts], True, color, 3)
            else:
                # Fallback: vẽ bbox
                bbox = region_data['bbox']
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
            
            # Vẽ center point
            center = region_data['pixel_center']
            cv2.circle(result_image, (int(center[0]), int(center[1])), 8, color, -1)
            
            # ⭐ ENHANCED: Hiển thị completion status ⭐
            robot_coords = region_data['robot_coordinates']
            region_id = region_data['region_id']
            pallet_id = region_data['pallet_id']
            applied_offset = region_data.get('applied_region_offset', 'unknown')
            
            # Check completion status
            region_name = f"P{pallet_id}R{region_id}"
            is_completed = completed_regions and region_name in completed_regions
            status_icon = "✅" if is_completed else "⏳"
            
            text_lines = [
                f"{status_icon} [{applied_offset}]",    # Status + Region name
                f"P{pallet_id}R{region_id}",            # Pallet/Region ID
                f"Px:{robot_coords['px']:.1f}",         # Robot X coordinate
                f"Py:{robot_coords['py']:.1f}"          # Robot Y coordinate
            ]
            
            # Vẽ từng dòng text
            for j, text in enumerate(text_lines):
                text_y = int(center[1]) - 40 + j * 20
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background cho text với màu tương ứng completion status
                bg_color = (0, 100, 0) if is_completed else (0, 0, 0)  # Xanh đậm nếu completed
                cv2.rectangle(result_image,
                            (int(center[0]) - text_size[0]//2 - 2, text_y - text_size[1] - 2),
                            (int(center[0]) + text_size[0]//2 + 2, text_y + 2),
                            bg_color, -1)
                
                # Text color
                text_color = (255, 255, 255) if not is_completed else (200, 255, 200)  # Xanh nhạt nếu completed
                cv2.putText(result_image, text,
                          (int(center[0]) - text_size[0]//2, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # ⭐ VẼ BAG PALLET STATUS VÀ COMPLETION STATISTICS ⭐
        status_y = 30
        status_text = [
            f"PLC: {'🟢 Connected' if self.plc_connected else '🔴 Disconnected'}",
            f"BAG PALLET TRACKING:",
            f"bag_pallet_1 = {self.bag_pallet_1}",
            f"bag_pallet_2 = {self.bag_pallet_2}",
            f"Regions: {len(regions_data)} | Active: {sum(1 for x in self.current_region_data.values() if x is not None)}"
        ]
        
        # ⭐ THÊM COMPLETION STATISTICS ⭐
        if completed_regions:
            total_expected = len(regions_data)  # Tổng regions dự kiến
            completed_count = len(completed_regions)
            progress_text = f"Progress: {completed_count}/{total_expected} ✅"
            status_text.append(progress_text)
            
            # Hiển thị completed regions
            if completed_count > 0:
                completed_str = ", ".join(completed_regions)
                status_text.append(f"Completed: {completed_str}")
        
        # Vẽ status text
        for i, text in enumerate(status_text):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background cho text
            cv2.rectangle(result_image,
                        (10 - 2, status_y + i * 20 - text_size[1] - 2),
                        (10 + text_size[0] + 2, status_y + i * 20 + 2),
                        (0, 0, 0), -1)
            
            # Chọn màu text
            if "🟢" in text or "✅" in text:
                color = (0, 255, 0)  # Xanh lá
            elif "🔴" in text or "Progress:" in text:
                color = (255, 255, 0)  # Vàng
            else:
                color = (255, 255, 255)  # Trắng
            
            cv2.putText(result_image, text,
                      (10, status_y + i * 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_image

def demo_single_image_with_plc():
    """
    Demo với một ảnh đơn lẻ: chia regions và gửi vào PLC
    """
    print("=== DEMO REGION DIVISION + PLC INTEGRATION ===")
    print("Chia pallets thành regions và gửi tọa độ vào PLC DB26")
    print()
    
    # Cấu hình PLC
    plc_ip = input("Nhập IP PLC (mặc định: 192.168.0.1): ").strip()
    if not plc_ip:
        plc_ip = "192.168.0.1"
    
    # Chọn layer
    layer_choice = input("Chọn layer (1 hoặc 2, mặc định 1): ").strip()
    try:
        layer = int(layer_choice) if layer_choice else 1
    except ValueError:
        layer = 1
    
    # Chọn ảnh
    pallets_folder = "images_pallets2"
    if os.path.exists(pallets_folder):
        image_files = [f for f in os.listdir(pallets_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()
        
        if image_files:
            print("\nẢnh có sẵn:")
            for i, img_file in enumerate(image_files, 1):
                print(f"  {i}. {img_file}")
            
            choice = input(f"\nChọn ảnh (1-{len(image_files)}): ").strip()
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(image_files):
                    image_path = os.path.join(pallets_folder, image_files[choice_num - 1])
                else:
                    image_path = os.path.join(pallets_folder, image_files[0])
            except ValueError:
                image_path = os.path.join(pallets_folder, image_files[0])
        else:
            print("Không tìm thấy ảnh trong folder!")
            return
    else:
        print(f"Không tìm thấy folder {pallets_folder}!")
        return
    
    # Đọc ảnh
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    print(f"\nĐang xử lý ảnh: {image_path}")
    print(f"Layer: {layer}")
    
    # Khởi tạo system
    print(f"\nKhởi tạo system...")
    engine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "utils", "detection", "best.engine")
    
    yolo_model = YOLOTensorRT(engine_path=engine_path, conf=0.5)
    region_plc = RegionDivisionPLCIntegration(plc_ip=plc_ip, debug=True)
    
    # Kết nối PLC
    if not region_plc.connect_plc():
        print("❌ Không thể kết nối PLC! Tiếp tục demo mà không gửi dữ liệu...")
    
    try:
        # Thực hiện YOLO detection
        print(f"\nThực hiện YOLO detection...")
        start_time = time.time()
        detections = yolo_model.detect(frame)
        yolo_time = time.time() - start_time
        
        print(f"YOLO time: {yolo_time*1000:.2f} ms")
        print(f"Đã phát hiện {len(detections.get('bounding_boxes', []))} objects")
        
        if len(detections.get('bounding_boxes', [])) == 0:
            print("Không phát hiện object nào!")
            return
        
        # Xử lý và gửi vào PLC
        print(f"\nXử lý regions và gửi vào PLC...")
        regions_data, send_success = region_plc.process_detection_and_send_to_plc(detections, layer)
        
        # Đọc lại từ PLC để verify
        if region_plc.plc_connected:
            print(f"\nĐọc lại từ PLC để verify...")
            time.sleep(0.2)  # Đợi một chút
            plc_data = region_plc.read_regions_from_plc()
            
            print(f"Dữ liệu trong PLC:")
            for region_name, data in plc_data.items():
                print(f"  {region_name}: Px={data['px']:7.2f} (DB26.{data['px_offset']}), "
                      f"Py={data['py']:7.2f} (DB26.{data['py_offset']})")
        
        # Tạo visualization
        print(f"\nTạo visualization...")
        vis_image = region_plc.create_visualization(frame, regions_data)
        
        # Hiển thị kết quả
        cv2.imshow("YOLO Detection", detections["annotated_frame"])
        cv2.imshow("Region Division + PLC", vis_image)
        
        print(f"\nNhấn phím bất kỳ để tiếp tục...")
        cv2.waitKey(0)
        
        # Lưu kết quả
        save_choice = input("\nBạn có muốn lưu kết quả? (y/n): ").lower()
        if save_choice in ['y', 'yes']:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"region_plc_{base_name}_layer{layer}.jpg"
            cv2.imwrite(output_path, vis_image)
            print(f"Đã lưu: {output_path}")
        
        cv2.destroyAllWindows()
        
    finally:
        # Cleanup
        region_plc.disconnect_plc()

if __name__ == "__main__":
    demo_single_image_with_plc() 