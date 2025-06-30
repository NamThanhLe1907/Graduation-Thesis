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
        # bag_number: DB26.40 (INT - 2 bytes)
        self.db26_offsets = {
            'loads': {'px': 0, 'py': 4},
            'pallets1': {'px': 12, 'py': 16}, 
            'pallets2': {'px': 24, 'py': 28},
            'bag_number': {'offset': 40}  # Offset 40 for bag number (INT)
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
                if region_name == 'bag_number':
                    print(f"  {region_name}: DB26.{offsets['offset']} (INT)")
                else:
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
        
        # ⭐ ENHANCED DEBUG: Check input detections ⭐
        if self.debug:
            print(f"[RegionPLC] Input: {len(detections.get('classes', []))} detections, {len(detections.get('corners', []))} corners")
        
        # Chỉ xử lý pallets (class 2.0)
        pallet_classes = [2.0]
        
        # Chia pallets thành regions sử dụng Module Division
        if self.debug:
            print(f"[RegionPLC] Calling ModuleDivision...")
        divided_result = self.module_divider.process_pallet_detections(
            detections, 
            layer=layer, 
            target_classes=pallet_classes
        )
        
        # ⭐ ENHANCED DEBUG: Check divided_result ⭐
        if self.debug:
            success = divided_result.get('processing_info', {}).get('success', False)
            total_regions = divided_result.get('total_regions', 0)
            print(f"[RegionPLC] ModuleDivision result: success={success}, total_regions={total_regions}")
        
        # Chuẩn bị dữ liệu regions
        if self.debug:
            print(f"[RegionPLC] Calling prepare_for_depth_estimation...")
        regions_data = self.module_divider.prepare_for_depth_estimation(divided_result)
        
        # ⭐ ENHANCED DEBUG: Check regions_data ⭐
        if self.debug:
            print(f"[RegionPLC] Final regions_data: {len(regions_data)} regions")
            for i, region_data in enumerate(regions_data):
                region_info = region_data.get('region_info', {})
                pallet_id = region_info.get('pallet_id', 'MISSING')
                region_id = region_info.get('region_id', 'MISSING')
                center = region_data.get('center', [0, 0])
                print(f"  Final Region {i}: P{pallet_id}R{region_id} center=({center[0]:.1f}, {center[1]:.1f})")
        
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
        ⭐ ENHANCED: Sử dụng depth results trực tiếp để lấy x,y,z coordinates ⭐
        Workflow: Depth Results (x,y,z) → Robot Transformation → PLC
        
        Args:
            detections: Kết quả detection từ YOLO (bao gồm cả depth results từ pipeline)
            layer: Layer để chia
            
        Returns:
            Tuple[List[Dict], bool]: (regions_data, send_success)
        """
        if self.debug:
            print(f"\n[RegionPLC] 🎯 NEW WORKFLOW: Using depth results directly")
        
        # ⭐ STEP 1: Lấy depth results từ pipeline (CHÍNH XÁC) ⭐
        # Access depth results từ detection pipeline thay vì tự tạo
        depth_results = detections.get('depth_results')
        pallet_regions = detections.get('pallet_regions', [])
        
        if not depth_results and not pallet_regions:
            if self.debug:
                print(f"[RegionPLC] ❌ Không có depth_results hoặc pallet_regions từ pipeline!")
            return [], False
        
        if self.debug:
            print(f"[RegionPLC] 📊 Input data: depth_results={len(depth_results) if depth_results else 0}, pallet_regions={len(pallet_regions)}")
        
        # ⭐ STEP 2: Sử dụng depth results nếu có, fallback về pallet_regions ⭐
        source_data = depth_results if depth_results else pallet_regions
        data_source = "depth_results" if depth_results else "pallet_regions"
        
        if self.debug:
            print(f"[RegionPLC] 🔄 Using {data_source} with {len(source_data)} regions")
        
        # ⭐ STEP 3: Process depth results theo bag mapping ⭐
        self._detections_context = detections  # ⭐ Store for fallback access ⭐
        self._process_depth_results_to_plc(source_data, data_source)
        
        # ⭐ STEP 4: Gửi vào PLC theo bag pallet tracking ⭐
        send_success = self.send_regions_to_plc()
        
        return source_data, send_success
    
    def _process_depth_results_to_plc(self, source_data: List[Dict], data_source: str):
        """
        ⭐ NEW: Process depth results theo bag mapping ⭐
        
        Args:
            source_data: Depth results or pallet regions
            data_source: Source of the data (depth_results or pallet_regions)
        """
        if self.debug:
            print(f"[RegionPLC] 📋 {data_source} data:")
            for i, region_data in enumerate(source_data):
                region_info = region_data.get('region_info', {})
                pallet_id = region_info.get('pallet_id', 'MISSING')
                region_id = region_info.get('region_id', 'MISSING')
                print(f"  {i}: P{pallet_id}R{region_id}")
        
        # ⭐ RESET CURRENT REGION DATA ⭐
        self.current_region_data = {
            'loads': None,
            'pallets1': None,
            'pallets2': None
        }
        
        # ⭐ ENHANCED: BAG POSITION MAPPING WITH REGION LOOKUP ⭐
        # Map bag number to actual region coordinates from source_data
        target_region_id = self.bag_to_region_mapping.get(self.current_bag_number, 1)
        
        if self.debug:
            print(f"[RegionPLC] 🎯 Current bag {self.current_bag_number} → Target region R{target_region_id}")
            print(f"[RegionPLC] 📊 Bag mapping: {self.bag_to_region_mapping}")
        
        # ⭐ STEP 1: Populate pallets1/pallets2 từ depth results (chính xác) ⭐
        self._populate_pallet_regions_from_source_data(source_data, target_region_id, data_source)
        
        # ⭐ STEP 2: Populate loads từ source_data hoặc robot_coordinates ⭐
        # Depth results thường chỉ có pallet regions, loads có thể ở robot_coordinates
        loads_found = self._populate_loads_from_source_data(source_data, data_source)
        
        # ⭐ FALLBACK: Nếu không tìm thấy loads trong depth results, thử robot_coordinates ⭐
        if not loads_found and hasattr(self, '_detections_context'):
            robot_coordinates = self._detections_context.get('robot_coordinates', [])
            if robot_coordinates:
                if self.debug:
                    print(f"[RegionPLC] 🔄 Fallback: Using robot_coordinates for loads...")
                self._populate_loads_from_robot_coordinates(robot_coordinates)
        
        # ⭐ DEBUG: Show final region data ⭐
        if self.debug:
            print(f"\n[RegionPLC] 📊 Final region data:")
            for region_name, region_data in self.current_region_data.items():
                if region_data:
                    px = region_data['robot_coordinates']['px']
                    py = region_data['robot_coordinates']['py']
                    source = region_data.get('coordinate_source', 'unknown')
                    print(f"  {region_name}: Px={px:.2f}, Py={py:.2f} ({source})")
                else:
                    print(f"  {region_name}: EMPTY")
    
    def _populate_pallet_regions_from_source_data(self, source_data: List[Dict], target_region_id: int, data_source: str):
        """
        ⭐ NEW: Populate pallets1/pallets2 từ source_data (ĐÚNG) ⭐
        
        Args:
            source_data: Depth results or pallet regions
            target_region_id: Target region ID based on bag mapping
            data_source: Source of the data (depth_results or pallet_regions)
        """
        # ⭐ FORCE DEBUG để tìm hiểu vấn đề ⭐
        print(f"\n[RegionPLC] 🏭 POPULATING PALLET REGIONS from source_data...")
        print(f"[RegionPLC] 🎯 Looking for P1R{target_region_id} and P2R{target_region_id}...")
        print(f"[RegionPLC] 📋 source_data count: {len(source_data)}")
        
        # Find P1R{target_region_id} for pallets1
        for i, region_data in enumerate(source_data):
            region_info = region_data.get('region_info', {})
            pallet_id = region_info.get('pallet_id')
            region_id = region_info.get('region_id')
            
            print(f"  Region {i}: P{pallet_id}R{region_id}")
            
            if pallet_id == 1 and region_id == target_region_id:
                # Found P1R{target_region_id} for pallets1
                
                # ⭐ USE DEPTH COORDINATES thay vì pixel coordinates ⭐
                robot_x, robot_y = self._extract_robot_coordinates_from_region(region_data, data_source)
                
                self.current_region_data['pallets1'] = {
                    'region_id': target_region_id,
                    'pallet_id': 1,
                    'robot_coordinates': {'px': robot_x, 'py': robot_y},
                    'pixel_center': region_data.get('center', [0, 0]),
                    'class': 'pallet',
                    'bag_number': self.current_bag_number,
                    'sequence_mapping': f"bao {self.current_bag_number} → P1R{target_region_id}",
                    'coordinate_source': f'{data_source}_P1R{target_region_id}',
                    'depth_info': region_data.get('position', {}),  # ⭐ Store depth info ⭐
                    'camera_3d': region_data.get('position_3d_camera', {})  # ⭐ Store 3D camera coords ⭐
                }
                
                self.bag_pallet_1 = 1
                
                print(f"    ✅ FOUND P1R{target_region_id} for pallets1: robot=({robot_x:.2f}, {robot_y:.2f}) from {data_source}")
            
            elif pallet_id == 2 and region_id == target_region_id:
                # Found P2R{target_region_id} for pallets2
                
                # ⭐ USE DEPTH COORDINATES thay vì pixel coordinates ⭐
                robot_x, robot_y = self._extract_robot_coordinates_from_region(region_data, data_source)
                
                self.current_region_data['pallets2'] = {
                    'region_id': target_region_id,
                    'pallet_id': 2,
                    'robot_coordinates': {'px': robot_x, 'py': robot_y},
                    'pixel_center': region_data.get('center', [0, 0]),
                    'class': 'pallet',
                    'bag_number': self.current_bag_number,
                    'sequence_mapping': f"bao {self.current_bag_number} → P2R{target_region_id}",
                    'coordinate_source': f'{data_source}_P2R{target_region_id}',
                    'depth_info': region_data.get('position', {}),  # ⭐ Store depth info ⭐
                    'camera_3d': region_data.get('position_3d_camera', {})  # ⭐ Store 3D camera coords ⭐
                }
                
                self.bag_pallet_2 = 2
                
                print(f"    ✅ FOUND P2R{target_region_id} for pallets2: robot=({robot_x:.2f}, {robot_y:.2f}) from {data_source}")
        
        # Check results
        pallets1_status = "✅ FOUND" if self.current_region_data['pallets1'] else "❌ NOT FOUND"
        pallets2_status = "✅ FOUND" if self.current_region_data['pallets2'] else "❌ NOT FOUND"
        print(f"[RegionPLC] 📊 Pallet regions status: pallets1={pallets1_status}, pallets2={pallets2_status}")
        
        # ⭐ DEBUG: Show final pallet coordinates ⭐
        if self.current_region_data['pallets1']:
            px1 = self.current_region_data['pallets1']['robot_coordinates']['px']
            py1 = self.current_region_data['pallets1']['robot_coordinates']['py']
            print(f"[RegionPLC] 🎯 pallets1 final: P1R{target_region_id} → Px={px1:.2f}, Py={py1:.2f}")
        
        if self.current_region_data['pallets2']:
            px2 = self.current_region_data['pallets2']['robot_coordinates']['px']
            py2 = self.current_region_data['pallets2']['robot_coordinates']['py']
            print(f"[RegionPLC] 🎯 pallets2 final: P2R{target_region_id} → Px={px2:.2f}, Py={py2:.2f}")
    
    def _populate_loads_from_source_data(self, source_data: List[Dict], data_source: str) -> bool:
        """
        ⭐ NEW: Populate loads từ source_data với IMPROVED detection logic ⭐
        
        Args:
            source_data: Depth results or pallet regions
            data_source: Source of the data (depth_results or pallet_regions)
            
        Returns:
            bool: True if loads was found and populated
        """
        print(f"\n[RegionPLC] 📦 POPULATING LOADS REGION from {data_source}...")
        print(f"[RegionPLC] 📋 source_data count: {len(source_data)}")
        
        # ⭐ IMPROVED LOAD DETECTION LOGIC ⭐
        for i, region_data in enumerate(source_data):
            region_info = region_data.get('region_info', {})
            object_class = region_info.get('object_class', '')
            pallet_id = region_info.get('pallet_id', 0)
            
            # Convert numeric class to string for comparison
            if isinstance(object_class, (int, float)):
                object_class = str(object_class)
            
            print(f"  Region {i}: object_class='{object_class}', pallet_id={pallet_id}")
            
            # ⭐ LOGIC 1: Non-pallet objects (pallet_id = 0) ⭐
            if pallet_id == 0 and object_class in ['load', 'load2', '0.0', '1.0', 0.0, 1.0]:
                # ⭐ NEW: Thêm region filtering để đảm bảo load không nằm trong pallets region ⭐
                center = region_data.get('center', [0, 0])
                is_in_loads_region = self.region_manager.is_point_in_region((center[0], center[1]), 'loads')
                is_in_pallets1 = self.region_manager.is_point_in_region((center[0], center[1]), 'pallets1')
                is_in_pallets2 = self.region_manager.is_point_in_region((center[0], center[1]), 'pallets2')
                
                # ⭐ CHỈ NHẬN LOAD NẾU: pallet_id=0 VÀ KHÔNG nằm trong pallets1/pallets2 ⭐
                if not is_in_pallets1 and not is_in_pallets2:
                    # Extract coordinates với robot transform + region offsets
                    robot_x, robot_y = self._extract_robot_coordinates_from_region(region_data, data_source)
                    
                    self.current_region_data['loads'] = {
                        'region_id': 1,  # Default loads region ID
                        'pallet_id': 1,  # Default loads pallet ID for PLC
                        'robot_coordinates': {'px': robot_x, 'py': robot_y},
                        'pixel_center': center,
                        'class': object_class,
                        'bag_number': self.current_bag_number,
                        'sequence_mapping': f"loads → {object_class} detection (filtered)",
                        'coordinate_source': f'{data_source}_load_filtered_{object_class}',
                        'depth_info': region_data.get('position', {}),
                        'depth_value': region_data.get('position', {}).get('z', 0.0)  # Depth từ depth module
                    }
                    
                    region_info = "loads" if is_in_loads_region else "general"
                    print(f"    ✅ FOUND {object_class} (pallet_id=0) in {region_info} area: robot=({robot_x:.2f}, {robot_y:.2f})")
                    return True
                else:
                    # ⭐ LOG: Load với pallet_id=0 nhưng nằm trong pallet region (đã được robot pick) ⭐
                    pallet_region = "pallets1" if is_in_pallets1 else "pallets2" 
                    print(f"    🚫 SKIPPED {object_class} (pallet_id=0) in {pallet_region} (robot already processed)")
            
            # ⭐ LOGIC 2: Load objects được assigned vào loads region thông qua RegionManager ⭐
            elif object_class in ['load', 'load2', '0.0', '1.0', 0.0, 1.0]:
                # Kiểm tra xem object có nằm trong loads region không
                center = region_data.get('center', [0, 0])
                is_in_loads_region = self.region_manager.is_point_in_region((center[0], center[1]), 'loads')
                
                # ⭐ NEW: Kiểm tra load KHÔNG nằm trong pallets1 để tránh nhận nhầm load đã được robot pick ⭐
                is_in_pallets1 = self.region_manager.is_point_in_region((center[0], center[1]), 'pallets1')
                is_in_pallets2 = self.region_manager.is_point_in_region((center[0], center[1]), 'pallets2')
                
                # ⭐ CHỈ NHẬN LOAD NẾU: nằm trong loads region VÀ KHÔNG nằm trong pallets1/pallets2 ⭐
                if is_in_loads_region and not is_in_pallets1 and not is_in_pallets2:
                    # Extract coordinates với robot transform + region offsets
                    robot_x, robot_y = self._extract_robot_coordinates_from_region(region_data, data_source)
                    
                    self.current_region_data['loads'] = {
                        'region_id': 1,  # Default loads region ID
                        'pallet_id': 1,  # Default loads pallet ID for PLC
                        'robot_coordinates': {'px': robot_x, 'py': robot_y},
                        'pixel_center': center,
                        'class': object_class,
                        'bag_number': self.current_bag_number,
                        'sequence_mapping': f"loads → {object_class} in loads region only",
                        'coordinate_source': f'{data_source}_loads_region_filtered_{object_class}',
                        'depth_info': region_data.get('position', {}),
                        'depth_value': region_data.get('position', {}).get('z', 0.0)  # Depth từ depth module
                    }
                    
                    print(f"    ✅ FOUND {object_class} in loads region (excluded from pallets): robot=({robot_x:.2f}, {robot_y:.2f})")
                    return True
                elif is_in_pallets1 or is_in_pallets2:
                    # ⭐ LOG: Load đã được robot pick vào pallet region ⭐
                    pallet_region = "pallets1" if is_in_pallets1 else "pallets2"
                    print(f"    🚫 SKIPPED {object_class} in {pallet_region} (already processed by robot)")
                elif not is_in_loads_region:
                    # ⭐ LOG: Load không nằm trong loads region ⭐
                    print(f"    ⚠️ SKIPPED {object_class} outside loads region")
                else:
                    print(f"    ❓ SKIPPED {object_class} (unknown filtering reason)")
        
        print(f"    ❌ NO LOAD DETECTION found for loads region")
        return False
    
    def _populate_loads_from_robot_coordinates(self, robot_coordinates: List[Dict]):
        """
        ⭐ NEW: Populate loads từ robot_coordinates (fallback) ⭐
        
        Args:
            robot_coordinates: Robot coordinates từ pipeline
        """
        print(f"\n[RegionPLC] 📦 POPULATING LOADS from robot_coordinates...")
        print(f"[RegionPLC] 📋 robot_coordinates count: {len(robot_coordinates)}")
        
        # Tìm load detection từ robot_coordinates
        load_classes = ['load', 'load2']
        for i, coord in enumerate(robot_coordinates):
            coord_class = coord.get('class', '')
            print(f"  Coord {i}: {coord_class}")
            
            if coord_class in load_classes:
                # ⭐ NEW: Kiểm tra vị trí pixel để đảm bảo không nằm trong pallets region ⭐
                camera_pixel = coord.get('camera_pixel', {})
                center = (camera_pixel.get('x', 0), camera_pixel.get('y', 0))
                
                is_in_loads_region = self.region_manager.is_point_in_region(center, 'loads')
                is_in_pallets1 = self.region_manager.is_point_in_region(center, 'pallets1')
                is_in_pallets2 = self.region_manager.is_point_in_region(center, 'pallets2')
                
                # ⭐ CHỈ NHẬN LOAD NẾU: KHÔNG nằm trong pallets1/pallets2 ⭐
                if not is_in_pallets1 and not is_in_pallets2:
                    robot_pos = coord['robot_coordinates']
                    
                    self.current_region_data['loads'] = {
                        'region_id': 1,  # Default loads region ID
                        'pallet_id': 1,  # Default loads pallet ID
                        'robot_coordinates': {
                            'px': robot_pos['x'],
                            'py': robot_pos['y']
                        },
                        'pixel_center': [center[0], center[1]],
                        'class': coord_class,
                        'bag_number': self.current_bag_number,
                        'sequence_mapping': f"loads → {coord_class} detection (filtered)",
                        'coordinate_source': f'robot_coordinates_load_filtered_{coord_class}'
                    }
                    
                    region_info = "loads" if is_in_loads_region else "general"
                    print(f"    ✅ FOUND {coord_class} in {region_info} area: robot=({robot_pos['x']:.2f}, {robot_pos['y']:.2f})")
                    return  # Exit after finding first valid load
                else:
                    # ⭐ LOG: Load nằm trong pallet region (đã được robot pick) ⭐
                    pallet_region = "pallets1" if is_in_pallets1 else "pallets2"
                    robot_pos = coord['robot_coordinates']
                    print(f"    🚫 SKIPPED {coord_class} in {pallet_region} (robot already processed): robot=({robot_pos['x']:.2f}, {robot_pos['y']:.2f})")
        
        print(f"    ❌ NO VALID LOAD DETECTION found in robot_coordinates (excluding pallets regions)")
    
    def set_current_bag_number(self, bag_number: int, send_to_plc: bool = True):
        """
        ⭐ NEW: Set current bag number và gửi vào PLC ⭐
        
        Args:
            bag_number: Bag number (1, 2, hoặc 3)
            send_to_plc: Có gửi bag number vào PLC không
        """
        if bag_number in self.bag_to_region_mapping:
            old_bag = self.current_bag_number
            self.current_bag_number = bag_number
            target_region = self.bag_to_region_mapping[bag_number]
            
            # Gửi bag number vào PLC nếu được yêu cầu
            if send_to_plc:
                self.send_bag_number_to_plc(bag_number)
            
            if self.debug:
                print(f"🎯 [BAG CONTROL] Switched: bao {old_bag} → bao {bag_number} (maps to R{target_region})")
                print(f"   Sequence mapping: {self.bag_to_region_mapping}")
        else:
            if self.debug:
                print(f"❌ [BAG CONTROL] Invalid bag number: {bag_number}. Valid: {list(self.bag_to_region_mapping.keys())}")
    
    def send_bag_number_to_plc(self, bag_number: int) -> bool:
        """
        ⭐ NEW: Gửi bag number vào PLC tại DB26.40 ⭐
        
        Args:
            bag_number: Bag number (1, 2, hoặc 3)
            
        Returns:
            bool: True nếu gửi thành công
        """
        if not self.plc_connected:
            if self.debug:
                print(f"[RegionPLC] ⚠️ PLC chưa kết nối, không thể gửi bag number")
            return False
        
        try:
            offset = self.db26_offsets['bag_number']['offset']
            success = self.plc_comm.write_db26_int(offset, bag_number)
            
            if self.debug:
                if success:
                    print(f"[RegionPLC] ✅ Đã gửi bag_number={bag_number} vào DB26.{offset}")
                else:
                    print(f"[RegionPLC] ❌ Lỗi gửi bag_number={bag_number} vào DB26.{offset}")
            
            return success
            
        except Exception as e:
            if self.debug:
                print(f"[RegionPLC] ❌ Exception khi gửi bag_number: {e}")
            return False
    
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

    def create_bag_tracking_visualization(self, frame, regions_data: List[Dict]):
        """
        ⭐ NEW: Create bag tracking visualization để hiển thị bag status ⭐
        
        Args:
            frame: Frame gốc
            regions_data: Regions từ module division
            
        Returns:
            np.ndarray: Frame với bag tracking info
        """
        viz_frame = frame.copy()
        
        # ⭐ DRAW CURRENT BAG INFO ⭐
        bag_info = self.get_current_bag_info()
        current_mapping = bag_info['sequence_mapping']
        target_region_id = self.bag_to_region_mapping.get(self.current_bag_number, 1)
        
        # Background cho header
        cv2.rectangle(viz_frame, (10, 10), (600, 100), (0, 0, 0), -1)
        cv2.rectangle(viz_frame, (10, 10), (600, 100), (255, 255, 255), 2)
        
        # Header text
        cv2.putText(viz_frame, "BAG PALLET TRACKING", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(viz_frame, current_mapping, (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(viz_frame, f"Target: P1R{target_region_id}, P2R{target_region_id}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 255, 128), 1)
        
        # ⭐ HIGHLIGHT TARGET REGIONS ⭐
        if regions_data:
            for i, region_data in enumerate(regions_data):
                region_info = region_data.get('region_info', {})
                pallet_id = region_info.get('pallet_id')
                region_id = region_info.get('region_id')
                
                # Check if this is target region for current bag
                is_target = region_id == target_region_id
                
                if 'center' in region_data:
                    center = region_data['center']
                    center_x, center_y = int(center[0]), int(center[1])
                    
                    # Color based on target status
                    if is_target:
                        color = (0, 255, 255)  # Yellow for target regions
                        thickness = 4
                        label = f"P{pallet_id}R{region_id} [TARGET]"
                    else:
                        color = (128, 128, 128)  # Gray for non-target
                        thickness = 2
                        label = f"P{pallet_id}R{region_id}"
                    
                    # Draw region boundary
                    if 'corners' in region_data and region_data['corners']:
                        corners = region_data['corners']
                        pts = np.array(corners, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(viz_frame, [pts], True, color, thickness)
                    elif 'bbox' in region_data:
                        bbox = region_data['bbox']
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw center point
                    cv2.circle(viz_frame, (center_x, center_y), 8, color, -1)
                    
                    # Draw label with background
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(viz_frame,
                                (center_x - text_size[0]//2 - 5, center_y - 30 - text_size[1] - 5),
                                (center_x + text_size[0]//2 + 5, center_y - 30 + 5),
                                (0, 0, 0), -1)
                    cv2.putText(viz_frame, label,
                              (center_x - text_size[0]//2, center_y - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # ⭐ DRAW CURRENT REGION DATA STATUS ⭐
        y_offset = 120
        line_height = 25
        
        status_lines = [
            "=== CURRENT REGION DATA ===",
            f"bag_pallet_1 = {self.bag_pallet_1}",
            f"bag_pallet_2 = {self.bag_pallet_2}",
        ]
        
        # Add region status
        for region_name in ['loads', 'pallets1', 'pallets2']:
            region_data = self.current_region_data.get(region_name)
            if region_data:
                px = region_data['robot_coordinates']['px']
                py = region_data['robot_coordinates']['py']
                source = region_data.get('coordinate_source', 'unknown')
                status_lines.append(f"{region_name}: Px={px:.2f}, Py={py:.2f} ({source})")
            else:
                status_lines.append(f"{region_name}: EMPTY")
        
        # Draw status with background
        for i, line in enumerate(status_lines):
            y_pos = y_offset + i * line_height
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background
            cv2.rectangle(viz_frame,
                        (10 - 2, y_pos - text_size[1] - 2),
                        (10 + text_size[0] + 2, y_pos + 2),
                        (0, 0, 0), -1)
            
            # Text color
            if i == 0:  # Header
                color = (0, 255, 255)  # Yellow
            elif "TARGET" in line:
                color = (0, 255, 255)  # Yellow for target
            elif "EMPTY" in line:
                color = (0, 0, 255)    # Red for empty
            else:
                color = (255, 255, 255)  # White
            
            cv2.putText(viz_frame, line, (10, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # ⭐ DRAW BAG CONTROLS ⭐
        controls_y = y_offset + len(status_lines) * line_height + 20
        control_lines = [
            "=== BAG CONTROLS ===",
            "Press '1': Set bao 1 → R1",
            "Press '2': Set bao 2 → R3", 
            "Press '3': Set bao 3 → R2",
            "Press 'n': Send to PLC",
        ]
        
        for i, line in enumerate(control_lines):
            y_pos = controls_y + i * 20
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            # Background
            cv2.rectangle(viz_frame,
                        (10 - 2, y_pos - text_size[1] - 2),
                        (10 + text_size[0] + 2, y_pos + 2),
                        (0, 0, 0), -1)
            
            # Text color
            color = (255, 255, 0) if i == 0 else (128, 255, 128)  # Yellow header, green controls
            
            cv2.putText(viz_frame, line, (10, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return viz_frame

    def _extract_robot_coordinates_from_region(self, region_data: Dict, data_source: str) -> Tuple[float, float]:
        """
        ⭐ Extract robot coordinates from region data với ĐÚNG robot transform + region offsets ⭐
        
        Args:
            region_data: Region data from source_data
            data_source: Source of the data (depth_results or pallet_regions)
            
        Returns:
            Tuple[float, float]: Robot coordinates (x, y) đã áp dụng region offsets
        """
        if self.debug:
            print(f"    [EXTRACT] Processing {data_source} data...")
        
        # ⭐ STRATEGY 1: Use depth position + robot transformation + region offsets ⭐
        if 'position' in region_data:
            position = region_data['position']
            pixel_x = position.get('x', 0.0)  # Pixel X từ depth results
            pixel_y = position.get('y', 0.0)  # Pixel Y từ depth results
            depth_z = position.get('z', 2.0)  # Depth Z từ depth module (không dùng cho tọa độ)
            
            # Transform pixel coordinates to robot coordinates
            raw_robot_x, raw_robot_y = self.robot_transformer.camera_to_robot(pixel_x, pixel_y)
            
            # ⭐ XÁC ĐỊNH REGION VÀ ÁP DỤNG OFFSET ⭐
            region_name = self._determine_region_for_coordinates((pixel_x, pixel_y), region_data)
            if region_name and region_name in self.region_manager.regions:
                offset = self.region_manager.regions[region_name]['offset']
                final_robot_x = raw_robot_x + offset['x']
                final_robot_y = raw_robot_y + offset['y']
                
                if self.debug:
                    print(f"      Using position + robot_transform + {region_name} offset:")
                    print(f"        pixel=({pixel_x:.1f}, {pixel_y:.1f}), depth={depth_z:.3f}m")
                    print(f"        raw_robot=({raw_robot_x:.2f}, {raw_robot_y:.2f})")
                    print(f"        offset=({offset['x']:.1f}, {offset['y']:.1f})")
                    print(f"        final_robot=({final_robot_x:.2f}, {final_robot_y:.2f})")
            else:
                # Không có region offset, dùng raw robot coordinates
                final_robot_x, final_robot_y = raw_robot_x, raw_robot_y
                if self.debug:
                    print(f"      Using position + robot_transform (no region offset):")
                    print(f"        pixel=({pixel_x:.1f}, {pixel_y:.1f}) → robot=({final_robot_x:.2f}, {final_robot_y:.2f})")
                
            return final_robot_x, final_robot_y
        
        # ⭐ STRATEGY 2: Fallback to pixel center coordinates + region offsets ⭐
        else:
            center_pixel = region_data.get('center', [0, 0])
            pixel_x, pixel_y = center_pixel[0], center_pixel[1]
            
            # Transform pixel coordinates to robot coordinates
            raw_robot_x, raw_robot_y = self.robot_transformer.camera_to_robot(pixel_x, pixel_y)
            
            # ⭐ XÁC ĐỊNH REGION VÀ ÁP DỤNG OFFSET ⭐
            region_name = self._determine_region_for_coordinates((pixel_x, pixel_y), region_data)
            if region_name and region_name in self.region_manager.regions:
                offset = self.region_manager.regions[region_name]['offset']
                final_robot_x = raw_robot_x + offset['x']
                final_robot_y = raw_robot_y + offset['y']
                
                if self.debug:
                    print(f"      Fallback center + robot_transform + {region_name} offset:")
                    print(f"        pixel=({pixel_x:.1f}, {pixel_y:.1f})")
                    print(f"        raw_robot=({raw_robot_x:.2f}, {raw_robot_y:.2f})")
                    print(f"        offset=({offset['x']:.1f}, {offset['y']:.1f})")
                    print(f"        final_robot=({final_robot_x:.2f}, {final_robot_y:.2f})")
            else:
                # Không có region offset, dùng raw robot coordinates
                final_robot_x, final_robot_y = raw_robot_x, raw_robot_y
                if self.debug:
                    print(f"      Fallback center + robot_transform (no region offset):")
                    print(f"        pixel=({pixel_x:.1f}, {pixel_y:.1f}) → robot=({final_robot_x:.2f}, {final_robot_y:.2f})")
                
            return final_robot_x, final_robot_y
    
    def _determine_region_for_coordinates(self, pixel_coords: Tuple[float, float], region_data: Dict) -> Optional[str]:
        """
        ⭐ Xác định region name để áp dụng offset dựa trên pixel coordinates ⭐
        
        Args:
            pixel_coords: Pixel coordinates (x, y)
            region_data: Region data from depth results
            
        Returns:
            str: Region name hoặc None
        """
        # Cố gắng sử dụng RegionManager để xác định region
        for region_name in self.region_manager.regions.keys():
            if self.region_manager.is_point_in_region(pixel_coords, region_name):
                return region_name
        
        # Fallback: Heuristic dựa trên class và position
        region_info = region_data.get('region_info', {})
        object_class = region_info.get('object_class', '')
        pallet_id = region_info.get('pallet_id', 0)
        
        # Load classes → loads region
        if object_class in ['load', 'load2', '0.0', '1.0'] or pallet_id == 0:
            return 'loads'
        
        # Pallet classes → dựa trên pallet_id
        elif pallet_id == 1:
            return 'pallets1'
        elif pallet_id == 2:
            return 'pallets2'
        
        return None

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