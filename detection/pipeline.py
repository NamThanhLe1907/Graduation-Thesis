"""
Pipeline xử lý đa luồng cho camera, phát hiện đối tượng và ước tính độ sâu.
"""
import multiprocessing as mp
import time
import traceback
import threading
import queue
from typing import Any, Callable,  Optional
import numpy as np
import cv2
import logging
import os
from pathlib import Path
import sys

# Add the main directory to the path to import region_division_plc_integration
current_dir = Path(__file__).parent
main_dir = current_dir.parent
sys.path.insert(0, str(main_dir))

# ⭐ IMPORT REGION DIVISION PLC INTEGRATION ⭐
try:
    from region_division_plc_integration import RegionDivisionPLCIntegration
    REGION_PLC_AVAILABLE = True
except ImportError as e:
    print(f"[Pipeline] Warning: Could not import RegionDivisionPLCIntegration: {e}")
    REGION_PLC_AVAILABLE = False

# ---------------------- CLASS QUẢN LÝ QUEUE ĐƠN GIẢN ----------------------
class QueueManager:
    """Quản lý queue với cơ chế lấy giá trị mới nhất."""
    
    def __init__(self, maxsize: int = 10):
        """
        Khởi tạo Queue Manager.
        
        Args:
            maxsize: Kích thước tối đa của queue
        """
        self._queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
    
    def put(self, item: Any) -> None:
        """
        Thêm item vào queue, nếu đầy thì bỏ các item cũ.
        
        Args:
            item: Đối tượng cần thêm
        """
        with self._lock:
            try:
                # Nếu queue đầy, xóa item cũ nhất
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                # Thêm item mới
                self._queue.put_nowait(item)
            except queue.Full:
                pass  # Bỏ qua nếu không thể thêm
    
    def get(self, timeout: float = 0.1) -> Optional[Any]:
        """
        Lấy item mới nhất từ queue.
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            Any: Item mới nhất hoặc None nếu không có
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest(self, timeout: float = 0.1) -> Optional[Any]:
        """
        Lấy item mới nhất từ queue (bỏ qua tất cả các item cũ).
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            Any: Item mới nhất hoặc None nếu không có
        """
        with self._lock:
            # Thử lấy tất cả các item hiện có
            latest_item = None
            try:
                while True:
                    item = self._queue.get_nowait()
                    latest_item = item
            except queue.Empty:
                pass
            return latest_item


# ---------------------- WORKER PROCESSES ----------------------

def _capture_frames_worker(
    camera_factory: Callable[[], Any],
    frame_queue: mp.Queue,
    run_event: mp.Event,
    frame_counter: mp.Value,
    ready_event: mp.Event,
    error_queue: mp.Queue,
    sleep: float = 0.01,
) -> None:
    """
    Worker chạy trong process con - chụp frame từ camera.
    
    Args:
        camera_factory: Hàm factory tạo camera
        frame_queue: Queue để đưa frame vào
        run_event: Event để báo hiệu process nên tiếp tục chạy
        frame_counter: Bộ đếm frame đã xử lý
        ready_event: Event báo hiệu process đã sẵn sàng
        error_queue: Queue để báo lỗi
        sleep: Thời gian sleep giữa các lần lấy frame nếu không có frame mới
    """
    try:
        print(f"[Camera Process {mp.current_process().pid}] Đã khởi động")
        
        try:
            camera = camera_factory()
            print(f"[Camera Process {mp.current_process().pid}] Camera đã khởi tạo thành công: {type(camera).__name__}")
            
            # Đảm bảo ready_event được set
            if not ready_event.is_set():
                ready_event.set()
                print(f"[Camera Process {mp.current_process().pid}] Đã đặt ready_event")
            
            frame_count = 0
            
            while run_event.is_set():
                try:
                    frame = camera.get_frame()
                    if frame is not None:
                        frame_queue.put(frame)
                        with frame_counter.get_lock():
                            frame_counter.value += 1
                        frame_count += 1
                        if frame_count % 10 == 0:
                            print(f"[Camera Process {mp.current_process().pid}] Đã xử lý {frame_count} frames")
                    else:
                        time.sleep(sleep)
                except Exception as e:
                    error_msg = f"[Camera Process {mp.current_process().pid}] Lỗi khi xử lý frame: {e}"
                    print(error_msg)
                    error_queue.put(error_msg)
                    traceback.print_exc()
                    time.sleep(sleep)
        except Exception as e:
            error_msg = f"[Camera Process {mp.current_process().pid}] Lỗi khởi tạo camera: {e}"
            print(error_msg)
            error_queue.put(error_msg)
            traceback.print_exc()
    except Exception as e:
        error_msg = f"[Camera Process {mp.current_process().pid}] Lỗi khởi tạo process: {e}"
        print(error_msg)
        try:
            error_queue.put(error_msg)
        except:
            pass
        traceback.print_exc()
    finally:
        # Đảm bảo ready_event được set dù có lỗi
        if not ready_event.is_set():
            ready_event.set()
            # print(f"[Camera Process {mp.current_process().pid}] Đã đặt ready_event (finally)")


def _yolo_detection_worker(
    yolo_factory: Callable[[], Any],
    frame_queue: mp.Queue,
    detection_queue: mp.Queue,
    depth_info_queue: mp.Queue,  # Queue để gửi thông tin tới depth process
    run_event: mp.Event,
    detection_counter: mp.Value,
    ready_event: mp.Event,
    error_queue: mp.Queue,
    sleep: float = 0.01,
) -> None:
    """
    Worker chạy trong process con - phát hiện đối tượng bằng YOLO.
    
    Args:
        yolo_factory: Hàm factory tạo YOLO model
        frame_queue: Queue để lấy frame đầu vào
        detection_queue: Queue để đưa kết quả detection ra
        depth_info_queue: Queue để gửi thông tin tới depth process
        run_event: Event để báo hiệu process nên tiếp tục chạy
        detection_counter: Bộ đếm số lượng detection đã xử lý
        ready_event: Event báo hiệu process đã sẵn sàng
        error_queue: Queue để báo lỗi
        sleep: Thời gian sleep nếu không có frame mới
    """
    try:
        print(f"[YOLO Process {mp.current_process().pid}] Đã khởi động")
        
        try:
            # Import module division inside worker to avoid multiprocessing issues
            import importlib
            import detection.utils.module_division
            importlib.reload(detection.utils.module_division)  # Force reload để tránh cache
            
            from detection import ModuleDivision
            
            # Import Robot Coordinate Transform
            from detection.utils.robot_coordinate_transform import RobotCoordinateTransform
            from detection.utils.camera_calibration import CameraCalibration
            
            # Import RegionManager for region-based processing
            from detection.utils.region_manager import RegionManager
            
            # ⭐ THÊM: Import PLC Communication cho Region Division ⭐
            try:
                from plc_communication import DB26Communication
                plc_available = True
                print(f"[YOLO Process] PLC Communication available")
            except ImportError as e:
                print(f"[YOLO Process] Warning: PLC Communication not available: {e}")
                plc_available = False
            
            # Import Theta4WithModuleDivision class
            try:
                import sys
                import os
                # Thêm thư mục gốc vào sys.path để import theta4_with_module_division
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                from theta4_with_module_division import Theta4WithModuleDivision
                theta4_available = True
                # print(f"[YOLO Process] Theta4WithModuleDivision imported successfully")
            except ImportError as e:
                # print(f"[YOLO Process] Warning: Could not import Theta4WithModuleDivision: {e}")
                theta4_available = False
            
            yolo_model = yolo_factory()
            print(f"[YOLO Process {mp.current_process().pid}] Model đã khởi tạo thành công: {type(yolo_model).__name__}")
            divider = ModuleDivision(debug=True)
            print(f"[YOLO Process] ModuleDivision đã được tạo với debug enabled")
            
            # Khởi tạo Robot Coordinate Transform và Camera Calibration
            robot_transformer = RobotCoordinateTransform()
            camera_calibration = CameraCalibration()
            print(f"[YOLO Process] Robot transformer và camera calibration đã được khởi tạo")
            
            # Khởi tạo RegionManager và load offset
            region_manager = RegionManager(auto_load_offsets=True)
            print(f"[YOLO Process] RegionManager đã được khởi tạo với {len(region_manager.regions)} regions")
            
            # Hiển thị offset đã load
            for region_name, region_info in region_manager.regions.items():
                offset = region_info['offset']
                if offset['x'] != 0 or offset['y'] != 0:
                    print(f"[YOLO Process] Region {region_name} offset: X={offset['x']}, Y={offset['y']}")
                else:
                    print(f"[YOLO Process] Region {region_name}: No offset (X=0, Y=0)")

            # ⭐ THÊM: Khởi tạo PLC Communication cho Region Division ⭐
            plc_comm = None
            enable_plc_regions = os.environ.get('ENABLE_PLC_REGIONS', 'true').lower() in ('true', '1', 'yes')
            # ⭐ FORCE ENABLE PLC cho testing với PLC thật ⭐ 
            enable_plc_regions = True
            plc_available = True
            if enable_plc_regions and plc_available:
                try:
                    plc_ip = os.environ.get('PLC_IP', '192.168.0.1')
                    plc_rack = int(os.environ.get('PLC_RACK', '0'))
                    plc_slot = int(os.environ.get('PLC_SLOT', '1'))
                    
                    plc_comm = DB26Communication(plc_ip, plc_rack, plc_slot)
                    
                    if plc_comm.connect():
                        print(f"[YOLO Process] ✅ PLC kết nối thành công: {plc_ip}:{plc_rack}:{plc_slot}")
                        print(f"[YOLO Process] DB26 Layout: Region1(0,4), Region2(8,12), Region3(16,20)")
                    else:
                        print(f"[YOLO Process] ❌ PLC kết nối thất bại, tiếp tục mà không gửi PLC")
                        plc_comm = None
                except Exception as e:
                    print(f"[YOLO Process] Warning: PLC initialization failed: {e}")
                    plc_comm = None
            else:
                print(f"[YOLO Process] PLC Regions disabled (ENABLE_PLC_REGIONS=false)")
            
            # Khởi tạo Theta4 calculator nếu có thể
            theta4_calculator = None
            if theta4_available:
                try:
                    theta4_calculator = Theta4WithModuleDivision(debug=False)  # Tắt debug để tránh spam log
                    print(f"[YOLO Process] Theta4Calculator đã được khởi tạo thành công")
                except Exception as e:
                    print(f"[YOLO Process] Warning: Could not initialize Theta4Calculator: {e}")
                    theta4_calculator = None
            
            ready_event.set()
            
            detection_count = 0
            
            while run_event.is_set():
                try:
                    frame = frame_queue.get(timeout=1.0)
                    if frame is not None:
                        # Phát hiện đối tượng với YOLO
                        detections = yolo_model.detect(frame)

                        # ⭐ SỬ DỤNG REGION MANAGER ĐỂ FILTER DETECTIONS ⭐
                        region_filtered = region_manager.filter_detections_by_regions(detections)
                        
                        # In thông tin regions
                        # region_counts = {name: len(data['bounding_boxes']) for name, data in region_filtered['regions'].items()}
                        # print(f"[REGION DEBUG] Detections by regions: {region_counts}")
                        # print(f"[REGION DEBUG] Unassigned: {len(region_filtered['unassigned']['bounding_boxes'])}")

                        # LOGIC MỚI: Tách xử lý pallet và non-pallet
                        pallet_classes = [2.0]  # Chỉ class 2.0 (pallet) mới chia regions
                        # print(f"[PIPELINE DEBUG] Tách xử lý: pallet_classes = {pallet_classes}")
                        
                        # 1. XỬ LÝ PALLET (class 2.0): Chia thành regions nhỏ
                        divided_result = divider.process_pallet_detections(detections, layer=1, target_classes=pallet_classes)
                        pallet_depth_regions = divider.prepare_for_depth_estimation(divided_result)
                        
                        # ⭐ DEBUG: Trace sequential logic input (mỗi 10 frames để dễ thấy) ⭐
                        if detection_counter.value % 10 == 0:
                            print(f"[SEQUENCE DEBUG] Sequential logic input:")
                            print(f"  pallet_depth_regions count: {len(pallet_depth_regions)}")
                            for i, region in enumerate(pallet_depth_regions):
                                region_info = region.get('region_info', {})
                                bbox = region.get('bbox', [])
                                has_corners = 'corners' in region
                                print(f"    Region {i}: P{region_info.get('pallet_id')}R{region_info.get('region_id')} bbox={[int(x) for x in bbox]} corners={has_corners}")
                            print(f"  divided_result success: {divided_result.get('processing_info', {}).get('success', False)}")
                            print(f"  divided_result total_regions: {divided_result.get('total_regions', 0)}")
                            print(f"  divided_result error: {divided_result.get('processing_info', {}).get('error', 'None')}")

                        # ⭐ SEQUENTIAL REGION SENDING WITH Z COORDINATES (PLAN IMPLEMENTATION) ⭐ 
                        if plc_comm and len(pallet_depth_regions) > 0:
                            try:
                                # ⭐ IMPORT REGION SEQUENCER ⭐
                                from detection.utils.region_sequencer import RegionSequencer
                                
                                # ⭐ NEW DB26 Layout with Z coordinates (12 bytes per region) ⭐
                                # Region 1: DB26.0 (X), DB26.4 (Y), DB26.8 (Z)    [12 bytes]
                                # Region 2: DB26.12 (X), DB26.16 (Y), DB26.20 (Z)  [12 bytes] 
                                # Region 3: DB26.24 (X), DB26.28 (Y), DB26.32 (Z)  [12 bytes]
                                db26_offsets_xyz = [
                                    {'px': 0, 'py': 4, 'pz': 8},       # Region 1
                                    {'px': 12, 'py': 16, 'pz': 20},    # Region 2  
                                    {'px': 24, 'py': 28, 'pz': 32}     # Region 3
                                ]
                                
                                # ⭐ INITIALIZE SEQUENCER ⭐
                                # Sử dụng biến global thay vì self vì đây là worker function
                                if '_region_sequencer' not in globals():
                                    globals()['_region_sequencer'] = RegionSequencer(sequence=[1, 3, 2])
                                
                                # ⭐ DETECT LOAD CLASSES FOR TARGET PALLET MAPPING ⭐
                                load_classes = [0.0, 1.0]  # class 0.0 (load) → Pallets1, class 1.0 (load2) → Pallets2
                                has_loads = any(cls in load_classes for cls in detections.get('classes', []))
                                
                                # ⭐ DEBUG: Hiển thị classes để check mapping (mỗi 10 frames để dễ thấy) ⭐
                                if detection_counter.value % 10 == 0:
                                    detected_classes = detections.get('classes', [])
                                    print(f"[SEQUENCE DEBUG][Frame {detection_counter.value}] Detected classes: {detected_classes}")
                                    print(f"[SEQUENCE DEBUG][Frame {detection_counter.value}] Target load_classes: {load_classes}")
                                    print(f"[SEQUENCE DEBUG][Frame {detection_counter.value}] has_loads: {has_loads}")
                                
                                if has_loads and len(pallet_depth_regions) > 0:
                                    # ⭐ GET TARGET PALLET (assume Pallet1 for now, can be enhanced later) ⭐
                                    target_pallet_id = 1
                                    pallet_regions = [r for r in pallet_depth_regions 
                                                    if r.get('region_info', {}).get('pallet_id') == target_pallet_id]
                                    
                                    if len(pallet_regions) > 0:
                                        # ⭐ ADD TO SEQUENCER QUEUE ⭐
                                        region_sequencer = globals()['_region_sequencer']
                                        if region_sequencer.is_queue_empty():
                                            region_sequencer.add_pallet_to_queue(pallet_regions, target_pallet_id)
                                        
                                        # ⭐ GET NEXT REGION IN SEQUENCE ⭐
                                        next_region = region_sequencer.get_next_region()
                                        
                                        if next_region:
                                            center_pixel = next_region['center']
                                            region_info = next_region['region_info']
                                            region_id = region_info['region_id']
                                            
                                            # ⭐ CHUYỂN ĐỔI PIXEL SANG ROBOT COORDINATES ⭐
                                            robot_x, robot_y = robot_transformer.camera_to_robot(
                                                center_pixel[0], center_pixel[1]
                                            )
                                            
                                            # ⭐ EXTRACT Z COORDINATE FROM DEPTH MODEL ⭐
                                            # TODO: Integrate depth results từ depth process để có Z coordinate thực tế
                                            depth_results = None  # Temporary fix - depth results không available trong context này
                                            robot_z = 2.0  # Default Z fallback (2 meters) cho sequential sending
                                            if depth_results and 'robot_coordinate_results' in depth_results:
                                                robot_coords = depth_results['robot_coordinate_results']
                                                robot_z = robot_coords.get('Z', 2.0)  # Use Z from depth or default 2.0m
                                            elif hasattr(camera_calibration, 'pixel_to_3d'):
                                                try:
                                                    # Fallback: sử dụng camera calibration để ước tính Z
                                                    test_depth = 2.0  # Assumed depth
                                                    X_3d, Y_3d, Z_3d = camera_calibration.pixel_to_3d(
                                                        center_pixel[0], center_pixel[1], test_depth
                                                    )
                                                    robot_z = Z_3d
                                                except Exception:
                                                    robot_z = 2.0  # Final fallback
                                            
                                            # ⭐ SEND SINGLE REGION WITH X,Y,Z TO PLC ⭐
                                            # Map region_id to DB26 offset index
                                            offset_index = region_id - 1  # region_id 1,2,3 -> index 0,1,2
                                            if 0 <= offset_index < len(db26_offsets_xyz):
                                                offsets = db26_offsets_xyz[offset_index]
                                                
                                                # Write X, Y, Z coordinates
                                                px_success = plc_comm.write_db26_real(offsets['px'], robot_x)
                                                py_success = plc_comm.write_db26_real(offsets['py'], robot_y)
                                                pz_success = plc_comm.write_db26_real(offsets['pz'], robot_z)
                                                
                                                if px_success and py_success and pz_success:
                                                    # ⭐ NEW CONSOLE OUTPUT WITH Z (mỗi 10 frames để dễ thấy) ⭐
                                                    if detection_counter.value % 10 == 0:
                                                        print(f"[SEQUENCE][Frame {detection_counter.value}] P{target_pallet_id}R{region_id} → DB26.{offsets['px']}/{offsets['py']}/{offsets['pz']}: Px={robot_x:.2f}, Py={robot_y:.2f}, Pz={robot_z:.2f}")
                                                        
                                                        # ⭐ SHOW SEQUENCE STATUS ⭐
                                                        status = region_sequencer.get_queue_status()
                                                        print(f"[SEQUENCE][Frame {detection_counter.value}] Progress: {status['progress']}, Status: {status['status']}")
                                                        
                                                        # ⭐ SHOW DEPTH INFO ⭐
                                                        # TODO: Integrate với depth process để có depth info thực tế
                                                        print(f"[SEQUENCE][Frame {detection_counter.value}] Using fallback Z={robot_z:.2f}m (depth integration needed)")
                                                    
                                                    # ⭐ AUTO-COMPLETE FOR DEMO (can be manual in real implementation) ⭐
                                                    # TODO: In real implementation, wait for robot completion signal
                                                    # region_sequencer.mark_region_completed()
                                                    
                                                else:
                                                    print(f"[SEQUENCE] ❌ Failed to send P{target_pallet_id}R{region_id}")
                                            else:
                                                print(f"[SEQUENCE] ❌ Invalid region_id {region_id} for offset mapping")
                                        else:
                                            print(f"[SEQUENCE] ⏳ No more regions in queue")
                                    else:
                                        print(f"[SEQUENCE] ❌ No regions found for target pallet {target_pallet_id}")
                                else:
                                    if not has_loads:
                                        print(f"[SEQUENCE] No loads detected (classes {load_classes}), skipping PLC sending")
                                    else:
                                        print(f"[SEQUENCE] No pallet regions available")
                                    
                            except Exception as e:
                                print(f"[SEQUENCE] Error in sequential sending: {e}")
                                # Traceback đã được import ở đầu file
                                traceback.print_exc()
                        
                        # 2. XỬ LÝ NON-PALLET (class khác): Không chia regions, chỉ lấy depth cho toàn bộ bbox
                        non_pallet_depth_regions = []
                        if 'classes' in detections and detections['classes']:
                            classes = detections['classes']
                            bboxes = detections.get('bounding_boxes', [])
                            corners_list = detections.get('corners', [])
                            
                            for i, cls in enumerate(classes):
                                if cls not in pallet_classes:  # Không phải pallet
                                    # Tạo depth region cho toàn bộ object (không chia nhỏ)
                                    region_info = {
                                        'region_id': 1,  # Chỉ có 1 region cho toàn bộ object
                                        'pallet_id': 0,  # Không phải pallet
                                        'global_region_id': len(pallet_depth_regions) + len(non_pallet_depth_regions) + 1,
                                        'layer': 0,  # Không có layer
                                        'module': 0,  # Không có module
                                        'object_class': cls  # Lưu class gốc
                                    }
                                    
                                    if i < len(bboxes):
                                        bbox = bboxes[i]
                                        center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                                        
                                        depth_region = {
                                            'bbox': bbox,
                                            'center': center,
                                            'region_info': region_info
                                        }
                                        
                                        # Thêm corners nếu có
                                        if i < len(corners_list):
                                            depth_region['corners'] = corners_list[i]
                                        
                                        non_pallet_depth_regions.append(depth_region)
                        
                        # print(f"[PIPELINE DEBUG] Non-pallet regions: {len(non_pallet_depth_regions)}")
                        
                        # 3. MERGE CẢ 2 LOẠI REGIONS
                        depth_regions = pallet_depth_regions + non_pallet_depth_regions
                        # print(f"[PIPELINE DEBUG] Total depth regions: {len(depth_regions)}")

                        # ⭐ THÊM THETA4 CALCULATION TẠI ĐÂY ⭐
                        theta4_result = None
                        if theta4_calculator is not None:
                            try:
                                # Chỉ tính theta4 nếu có loads (class 0, 1) và có regions
                                has_loads = any(cls in [0.0, 1.0] for cls in detections.get('classes', []))
                                has_regions = len(pallet_depth_regions) > 0
                                
                                if has_loads and has_regions:
                                    # print(f"[PIPELINE DEBUG] Tính toán Theta4...")
                                    theta4_result = theta4_calculator.process_full_pipeline(detections, layer=1)
                                    # print(f"[PIPELINE DEBUG] Theta4 completed: {theta4_result['summary']['successful_theta4']} successful calculations")
                                else:
                                    # print(f"[PIPELINE DEBUG] Bỏ qua Theta4: has_loads={has_loads}, has_regions={has_regions}")
                                    pass
                            except Exception as e:
                                # print(f"[YOLO Process] Warning: Theta4 calculation failed: {e}")
                                theta4_result = None

                        # ⭐ THÊM ROBOT COORDINATE TRANSFORMATION VỚI REGION OFFSET ⭐
                        robot_coordinates = []
                        if 'classes' in detections and detections['classes']:
                            classes = detections['classes']
                            bboxes = detections.get('bounding_boxes', [])
                            scores = detections.get('scores', [])
                            
                            for i, cls in enumerate(classes):
                                try:
                                    # Chỉ xử lý pallet (2), load (0), load2 (1)
                                    class_names = {0: 'load', 1: 'load2', 2: 'pallet'}
                                    if cls not in class_names:
                                        continue
                                    
                                    if i < len(bboxes):
                                        bbox = bboxes[i]
                                        center_x = (bbox[0] + bbox[2]) / 2
                                        center_y = (bbox[1] + bbox[3]) / 2
                                        confidence = scores[i] if i < len(scores) else 0.0
                                        
                                        # ⭐ TÌM REGION CHO DETECTION ⭐
                                        assigned_region = region_manager.get_region_for_detection((center_x, center_y), cls)
                                        
                                        # Chuyển đổi từ camera pixel sang robot coordinates
                                        robot_x, robot_y = robot_transformer.camera_to_robot(center_x, center_y)
                                        
                                        # ⭐ ÁP DỤNG OFFSET THEO REGION ⭐
                                        robot_coords_raw = {'x': robot_x, 'y': robot_y}
                                        if assigned_region:
                                            robot_coords_final = region_manager.apply_region_offset(robot_coords_raw, assigned_region)
                                        else:
                                            robot_coords_final = robot_coords_raw
                                        
                                        # Tính thêm tọa độ 3D camera để so sánh (sử dụng camera calibration)
                                        camera_3d = None
                                        try:
                                            test_depth = 2.0  # Depth giả định
                                            X_3d, Y_3d, Z_3d = camera_calibration.pixel_to_3d(
                                                center_x, center_y, test_depth
                                            )
                                            camera_3d = {
                                                'X': round(X_3d, 3),
                                                'Y': round(Y_3d, 3), 
                                                'Z': round(Z_3d, 3)
                                            }
                                        except Exception as e:
                                            # print(f"[YOLO Process] Camera 3D calculation failed: {e}")
                                            pass
                                        
                                        coord_info = {
                                            'class': class_names[cls],
                                            'class_id': int(cls),
                                            'confidence': round(confidence, 3),
                                            'camera_pixel': {
                                                'x': int(center_x),
                                                'y': int(center_y)
                                            },
                                            'robot_coordinates': {
                                                'x': round(robot_coords_final['x'], 2),
                                                'y': round(robot_coords_final['y'], 2)
                                            },
                                            'robot_coordinates_raw': {
                                                'x': round(robot_x, 2),
                                                'y': round(robot_y, 2)
                                            },
                                            'assigned_region': assigned_region,  # ⭐ THÊM THÔNG TIN REGION ⭐
                                            'camera_3d': camera_3d,  # Thêm để so sánh
                                            'bbox': [int(x) for x in bbox]
                                        }
                                        
                                        robot_coordinates.append(coord_info)
                                        
                                        # In ra console - LOG ROBOT COORDINATES VỚI REGION INFO (mỗi 50 frames)
                                        if detection_counter.value % 50 == 0:
                                            region_str = f"[{assigned_region}]" if assigned_region else "[UNASSIGNED]"
                                            print(f"[ROBOT COORDS][Frame {detection_counter.value}] {region_str} {class_names[cls]}: Pixel({int(center_x)},{int(center_y)}) → Robot(X={robot_coords_final['x']:.2f}, Y={robot_coords_final['y']:.2f})")
                                            if assigned_region and robot_coords_raw != robot_coords_final:
                                                print(f"                Raw Robot(X={robot_x:.2f}, Y={robot_y:.2f}) + Offset → Final(X={robot_coords_final['x']:.2f}, Y={robot_coords_final['y']:.2f})")
                                            if camera_3d:
                                                print(f"                Camera3D: X={camera_3d['X']:.3f}, Y={camera_3d['Y']:.3f}, Z={camera_3d['Z']:.3f}")
                                        
                                except Exception as e:
                                    # print(f"[YOLO Process] Error processing robot coordinate for class {cls}: {e}")
                                    pass

                        # Thêm theta4 info, robot coordinates và region info vào detections để truyền ra ngoài
                        detections_with_theta4 = detections.copy()
                        detections_with_theta4['theta4_result'] = theta4_result
                        detections_with_theta4['divided_result'] = divided_result  # Cũng truyền module division result
                        detections_with_theta4['robot_coordinates'] = robot_coordinates  # ⭐ THÊM ROBOT COORDINATES
                        detections_with_theta4['region_filtered'] = region_filtered  # ⭐ THÊM REGION INFORMATION
                        detections_with_theta4['pallet_regions'] = pallet_depth_regions  # ⭐ THÊM PALLET REGIONS INFORMATION
                        
                        # ⭐ DEBUG: Confirm pallet_regions được gửi ra ngoài (mỗi 10 frames để dễ thấy) ⭐
                        if detection_counter.value % 10 == 0:
                            print(f"[SEQUENCE DEBUG] Sending to main process:")
                            print(f"  pallet_regions in output: {len(pallet_depth_regions)} regions")
                            print(f"  detections_with_theta4 keys: {list(detections_with_theta4.keys())}")
                        
                        # ⭐ THÊM SEQUENCER STATUS CHO KEYBOARD CONTROLS ⭐
                        if '_region_sequencer' in globals():
                            region_sequencer = globals()['_region_sequencer']
                            sequencer_status = region_sequencer.get_queue_status()
                            detections_with_theta4['sequencer_status'] = sequencer_status
                            detections_with_theta4['sequencer_available'] = True
                            
                            # ⭐ DEBUG: Confirm sequencer status được thêm (mỗi 10 frames) ⭐
                            if detection_counter.value % 10 == 0:
                                print(f"[SEQUENCE DEBUG] Adding sequencer status:")
                                print(f"  sequencer_available: True")
                                print(f"  sequencer_status: {sequencer_status['status']}")
                                print(f"  current_pallet: {sequencer_status.get('current_pallet')}")
                                print(f"  progress: {sequencer_status['progress']}")
                        else:
                            detections_with_theta4['sequencer_status'] = None
                            detections_with_theta4['sequencer_available'] = False
                            
                            # ⭐ DEBUG: No sequencer found ⭐
                            if detection_counter.value % 10 == 0:
                                print(f"[SEQUENCE DEBUG] No sequencer found in globals()")
                                print(f"  sequencer_available: False")

                        # Gửi kết quả detection (bao gồm theta4) ra ngoài
                        detection_queue.put((frame, detections_with_theta4))
                        
                        # Gửi thông tin cần thiết cho depth process (non-blocking)
                        depth_info = {
                            'frame': frame,
                            'regions': depth_regions,
                            'divided_result': divided_result,
                        }
                        
                        # Không chặn YOLO process nếu depth_info_queue đầy
                        try:
                            # Kiểm tra xem queue có đầy không
                            if depth_info_queue.full():
                                # Nếu đầy, bỏ qua việc đưa vào depth queue
                                # Điều này cho phép YOLO tiếp tục hoạt động
                                # print(f"[YOLO Process] Depth queue đầy, bỏ qua xử lý depth cho frame này")
                                pass
                            else:
                                # Sử dụng put_nowait thay vì put để tránh chặn
                                depth_info_queue.put_nowait(depth_info)
                        except Exception as e:
                            # Bỏ qua lỗi khi queue đầy
                            # print(f"[YOLO Process] Không thể đưa vào depth queue: {str(e)}")
                            pass
                        
                        with detection_counter.get_lock():
                            detection_counter.value += 1
                        detection_count += 1
                        
                        # if detection_count % 10 == 0:
                        #     print(f"[YOLO Process {mp.current_process().pid}] Đã xử lý {detection_count} detections")
                    else:
                        time.sleep(sleep)
                except Exception as e:
                    error_msg = f"[YOLO Process {mp.current_process().pid}] Lỗi khi xử lý detection: {e}"
                    print(error_msg)
                    error_queue.put(error_msg)
                    traceback.print_exc()
                    time.sleep(sleep)
        except Exception as e:
            error_msg = f"[YOLO Process {mp.current_process().pid}] Lỗi khởi tạo model: {e}"
            print(error_msg)
            error_queue.put(error_msg)
            traceback.print_exc()
        finally:
            # ⭐ CLEANUP PLC CONNECTION ⭐
            if 'plc_comm' in locals() and plc_comm:
                try:
                    plc_comm.disconnect()
                    print(f"[YOLO Process] PLC disconnected")
                except:
                    pass
    except Exception as e:
        error_msg = f"[YOLO Process {mp.current_process().pid}] Lỗi khởi tạo process: {e}"
        print(error_msg)
        try:
            error_queue.put(error_msg)
        except:
            pass
        traceback.print_exc()
    finally:
        ready_event.set()


def _depth_estimation_worker(
    depth_factory: Callable[[], Any],
    depth_info_queue: mp.Queue,
    depth_result_queue: mp.Queue,
    run_event: mp.Event,
    depth_counter: mp.Value,
    ready_event: mp.Event,
    error_queue: mp.Queue,
    sleep: float = 0.01,
) -> None:
    """
    Worker chạy trong process con - ước tính độ sâu.
    
    Args:
        depth_factory: Hàm factory tạo Depth model
        depth_info_queue: Queue để lấy thông tin từ YOLO process
        depth_result_queue: Queue để đưa kết quả độ sâu ra
        run_event: Event để báo hiệu process nên tiếp tục chạy
        depth_counter: Bộ đếm số lượng depth đã xử lý
        ready_event: Event báo hiệu process đã sẵn sàng
        error_queue: Queue để báo lỗi
        sleep: Thời gian sleep nếu không có dữ liệu mới
    """
    try:
        # print(f"[Depth Process {mp.current_process().pid}] Đã khởi động")
        import threading
        import queue
        
        # Queue nội bộ trong process để tách việc nhận dữ liệu và xử lý depth
        internal_queue = queue.Queue(maxsize=5)
        
        # Flag để kiểm soát thread
        thread_running = threading.Event()
        thread_running.set()
        
        # Thread riêng để xử lý depth (tác vụ nặng)
        def depth_processing_thread():
            nonlocal depth_counter, depth_count
            # print(f"[Depth Thread] Đã khởi động thread xử lý độ sâu")
            
            while thread_running.is_set():
                try:
                    # Lấy từ queue nội bộ (non-blocking)
                    try:
                        depth_task = internal_queue.get(timeout=0.5)
                        frame = depth_task['frame']
                        regions = depth_task.get('regions', [])
                        divided_result = depth_task.get('divided_result', {})
                        
                        # Xử lý depth cho từng region
                        depth_results = []
                        for region in regions:
                            bbox = region['bbox']
                            region_info = region['region_info']
                            
                            # Ước tính độ sâu cho bbox này (sử dụng phiên bản có 3D nếu có camera calibration)
                            if hasattr(depth_model, 'camera_calibration') and depth_model.camera_calibration is not None:
                                region_depth = depth_model.estimate_depth_with_3d(frame, [bbox])
                            else:
                                region_depth = depth_model.estimate_depth(frame, [bbox])
                            
                            # DEBUG: In thông tin chi tiết về depth
                            # if region_depth and len(region_depth) > 0:
                            #     depth_info = region_depth[0]
                            #     print(f"[DEPTH DEBUG] Region {region_info.get('region_id', '?')} (Pallet {region_info.get('pallet_id', '?')}):")
                            #     if isinstance(depth_info, dict):
                            #         print(f"  - Mean depth: {depth_info.get('mean_depth', 0.0):.3f}m")
                            #         print(f"  - Min depth: {depth_info.get('min_depth', 0.0):.3f}m") 
                            #         print(f"  - Max depth: {depth_info.get('max_depth', 0.0):.3f}m")
                            #         print(f"  - Std depth: {depth_info.get('std_depth', 0.0):.3f}m")
                            #         if 'center_3d' in depth_info:
                            #             pos_3d = depth_info['center_3d']
                            #             print(f"  - 3D position: X={pos_3d.get('X', 0):.3f}m, Y={pos_3d.get('Y', 0):.3f}m, Z={pos_3d.get('Z', 0):.3f}m")
                            #     else:
                            #         print(f"  - Depth value: {depth_info}")
                            
                            # Tạo kết quả chi tiết cho region
                            if region_depth and len(region_depth) > 0:
                                depth_info = region_depth[0]  # Lấy kết quả đầu tiên
                                result = {
                                    'region_info': region_info,
                                    'bbox': bbox,
                                    'center': region['center'],
                                    'depth': depth_info,
                                    'position': {
                                        'x': region['center'][0],
                                        'y': region['center'][1], 
                                        'z': depth_info.get('mean_depth', 0.0) if isinstance(depth_info, dict) else 0.0
                                    }
                                }
                                
                                # Thêm thông tin 3D nếu có camera calibration
                                if hasattr(depth_model, 'camera_calibration') and depth_model.camera_calibration is not None:
                                    if 'center_3d' in depth_info:
                                        result['position_3d_camera'] = depth_info['center_3d']
                                    if 'real_size' in depth_info:
                                        result['real_size'] = depth_info['real_size']
                                
                                # Thêm corners nếu có (để vẽ rotated boxes)
                                if 'corners' in region:
                                    result['corners'] = region['corners']
                                
                                # Thêm corners gốc của pallet nếu có
                                if 'original_corners' in region:
                                    result['original_corners'] = region['original_corners']
                                
                                depth_results.append(result)
                        
                        # Gửi kết quả ra
                        depth_result_queue.put((frame, depth_results))
                        
                        with depth_counter.get_lock():
                            depth_counter.value += 1
                        depth_count += 1
                        
                        # if depth_count % 10 == 0:
                        #     print(f"[Depth Thread] Đã xử lý {depth_count} depth estimates cho {len(depth_results)} regions")
                    except queue.Empty:
                        time.sleep(0.01)
                        continue
                        
                except Exception as e:
                    error_msg = f"[Depth Thread] Lỗi khi xử lý depth: {str(e)}"
                    print(error_msg)
                    try:
                        error_queue.put(error_msg)
                    except:
                        pass
                    traceback.print_exc()
                    time.sleep(sleep)
        
        try:
            # Khởi tạo model
            depth_model = depth_factory()
            # print(f"[Depth Process {mp.current_process().pid}] Model đã khởi tạo thành công: {type(depth_model).__name__}")
            
            # Báo hiệu đã sẵn sàng
            ready_event.set()
            
            depth_count = 0
            
            # Khởi động thread xử lý độ sâu
            depth_thread = threading.Thread(target=depth_processing_thread, daemon=True)
            depth_thread.start()
            
            # Vòng lặp chính trong process - chỉ nhận dữ liệu từ queue và chuyển vào queue nội bộ
            while run_event.is_set():
                try:
                    # Lấy dữ liệu từ queue liên process (có thể block nhưng chỉ trong thời gian ngắn)
                    try:
                        depth_info = depth_info_queue.get(timeout=0.1)
                        if depth_info is not None:
                            # Chỉ đưa vào queue nội bộ nếu nó không đầy (tránh tích tụ quá nhiều frame cũ)
                            if not internal_queue.full():
                                internal_queue.put(depth_info)
                            else:
                                # Nếu queue đầy, loại bỏ item cũ nhất và thêm item mới
                                try:
                                    internal_queue.get_nowait()  # Loại bỏ item cũ nhất
                                except queue.Empty:
                                    pass
                                internal_queue.put_nowait(depth_info)
                    except mp.queues.Empty:
                        time.sleep(sleep)
                        continue
                        
                except Exception as e:
                    error_msg = f"[Depth Process {mp.current_process().pid}] Lỗi khi nhận dữ liệu: {str(e)}"
                    print(error_msg)
                    error_queue.put(error_msg)
                    traceback.print_exc()
                    time.sleep(sleep)
                    
        except Exception as e:
            error_msg = f"[Depth Process {mp.current_process().pid}] Lỗi khởi tạo model: {str(e)}"
            print(error_msg)
            error_queue.put(error_msg)
            traceback.print_exc()
            
        finally:
            # Dừng thread xử lý
            thread_running.clear()
            if 'depth_thread' in locals() and depth_thread.is_alive():
                depth_thread.join(timeout=1.0)
                
    except Exception as e:
        error_msg = f"[Depth Process {mp.current_process().pid}] Lỗi khởi tạo process: {str(e)}"
        print(error_msg)
        try:
            error_queue.put(error_msg)
        except:
            pass
        traceback.print_exc()
    finally:
        ready_event.set()


# ---------------------- MAIN CLASS ----------------------

class ProcessingPipeline:
    """Pipeline xử lý đa luồng với Camera, YOLO và Depth Estimation."""
    
    def __init__(
        self,
        camera_factory: Callable[[], Any],
        yolo_factory: Callable[[], Any],
        depth_factory: Callable[[], Any],
        max_queue_size: int =  3,
        enable_plc: bool = True,
        plc_ip: str = "192.168.0.1"
    ):
        """
        Khởi tạo pipeline xử lý đa luồng.
        
        Args:
            camera_factory: Hàm factory tạo camera
            yolo_factory: Hàm factory tạo YOLO model
            depth_factory: Hàm factory tạo Depth model
            max_queue_size: Kích thước tối đa của mỗi queue
        """
        # Đảm bảo sử dụng spawn method cho Windows
        if mp.get_start_method(allow_none=True) != 'spawn':
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
        
        # Lưu các factory function
        self._camera_factory = camera_factory
        self._yolo_factory = yolo_factory
        self._depth_factory = depth_factory
        
        # Tạo các queue
        self._frame_queue = mp.Queue(maxsize=max_queue_size)
        self._detection_queue = mp.Queue(maxsize=max_queue_size)
        self._depth_info_queue = mp.Queue(maxsize=max_queue_size)
        self._depth_result_queue = mp.Queue(maxsize=max_queue_size)
        self._error_queue = mp.Queue()
        
        # Tạo các bộ đếm
        self.frame_counter = mp.Value('i', 0)
        self.detection_counter = mp.Value('i', 0)
        self.depth_counter = mp.Value('i', 0)
        
        # Tạo các event
        self._run_event = mp.Event()
        self._camera_ready_event = mp.Event()
        self._yolo_ready_event = mp.Event()
        self._depth_ready_event = mp.Event()
        
        # Các process
        self._camera_process = None
        self._yolo_process = None
        self._depth_process = None
        
        # Danh sách lỗi
        self._errors = []
        
        # Các QueueManager cho việc lấy kết quả
        self.detection_manager = QueueManager(maxsize=max_queue_size)
        self.depth_manager = QueueManager(maxsize=max_queue_size)
        
        # ⭐ PLC INTEGRATION ⭐
        self.enable_plc = enable_plc
        self.plc_ip = plc_ip
        self.plc_integration = None
        
        if self.enable_plc and REGION_PLC_AVAILABLE:
            try:
                print(f"[Pipeline] 🔧 Initializing PLC Integration (IP: {plc_ip})...")
                self.plc_integration = RegionDivisionPLCIntegration(
                    plc_ip=plc_ip, 
                    debug=True
                )
                print(f"[Pipeline] ✅ PLC Integration initialized successfully!")
                print(f"[Pipeline] 📋 PLC DB26 Layout: loads=0/4, pallets1=12/16, pallets2=24/28")
            except Exception as e:
                print(f"[Pipeline] ❌ Failed to initialize PLC Integration: {e}")
                import traceback
                traceback.print_exc()
                self.plc_integration = None
        elif self.enable_plc:
            print(f"[Pipeline] ❌ PLC Integration requested but RegionDivisionPLCIntegration not available")
            print(f"[Pipeline] 💡 Check if region_division_plc_integration.py exists and imports correctly")
        else:
            print(f"[Pipeline] ℹ️ PLC Integration disabled")
    
    def start(self, timeout: float = 30.0) -> bool:
        """
        Khởi động tất cả các process.
        
        Args:
            timeout: Thời gian tối đa chờ các process khởi động (giây)
            
        Returns:
            bool: True nếu tất cả process đã sẵn sàng, False nếu không
        """
        # Đặt run event
        self._run_event.set()
        
        # Clear các ready event
        self._camera_ready_event.clear()
        self._yolo_ready_event.clear() 
        self._depth_ready_event.clear()
        
        # Khởi động Camera Process
        self._camera_process = mp.Process(
            target=_capture_frames_worker,
            args=(
                self._camera_factory,
                self._frame_queue,
                self._run_event,
                self.frame_counter,
                self._camera_ready_event,
                self._error_queue,
            ),
            daemon=True,
        )
        self._camera_process.start()
        # print(f"Camera Process đã khởi động với PID: {self._camera_process.pid}")
        
        # Khởi động YOLO Process
        self._yolo_process = mp.Process(
            target=_yolo_detection_worker,
            args=(
                self._yolo_factory,
                self._frame_queue,
                self._detection_queue,
                self._depth_info_queue,
                self._run_event,
                self.detection_counter,
                self._yolo_ready_event,
                self._error_queue,
            ),
            daemon=True,
        )
        self._yolo_process.start()
        # print(f"YOLO Process đã khởi động với PID: {self._yolo_process.pid}")
        
        # Khởi động Depth Process
        self._depth_process = mp.Process(
            target=_depth_estimation_worker,
            args=(
                self._depth_factory,
                self._depth_info_queue,
                self._depth_result_queue,
                self._run_event,
                self.depth_counter,
                self._depth_ready_event,
                self._error_queue,
            ),
            daemon=True,
        )
        self._depth_process.start()
        # print(f"Depth Process đã khởi động với PID: {self._depth_process.pid}")
        
        # Đợi camera process sẵn sàng với timeout dài hơn vì thường khởi tạo camera mất nhiều thời gian
        # print(f"Đang đợi Camera Process (tối đa {timeout}s)...")
        camera_ready = self._camera_ready_event.wait(timeout)
        
        # Đợi các process khác
        yolo_ready = self._yolo_ready_event.wait(timeout)
        depth_ready = self._depth_ready_event.wait(timeout)
        
        # Kiểm tra lỗi
        self._check_errors()
        
        # In thông tin trạng thái
        # print(f"Camera Process ready: {camera_ready}")
        # print(f"YOLO Process ready: {yolo_ready}")
        # print(f"Depth Process ready: {depth_ready}")
        
        # Nếu phát hiện camera process còn sống nhưng không sẵn sàng, thử kiểm tra lại một lần nữa
        if not camera_ready and self._camera_process.is_alive():
            # print("Camera process còn sống nhưng chưa sẵn sàng, kiểm tra lại...")
            # Thử kiểm tra lại nếu Process còn sống
            time.sleep(1.0)  # Đợi thêm chút nữa
            camera_ready = self._camera_ready_event.is_set()
            # print(f"Kiểm tra lại Camera Process ready: {camera_ready}")
        
        # Khởi động background thread để chuyển kết quả từ Queue vào QueueManager
        if camera_ready and yolo_ready and depth_ready:
            self._start_queue_workers()
            return True
        else:
            # In thông tin debug về các process đang chạy
            # print(f"Camera Process is alive: {self._camera_process.is_alive()}")
            # print(f"YOLO Process is alive: {self._yolo_process.is_alive()}")
            # print(f"Depth Process is alive: {self._depth_process.is_alive()}")
            return False
    
    def _start_queue_workers(self):
        """Khởi động các thread để chuyển dữ liệu từ Queue sang QueueManager."""
        
        # Thread để chuyển kết quả detection
        def detection_worker():
            while self._run_event.is_set():
                try:
                    result = self._detection_queue.get(timeout=0.5)
                    if result is not None:
                        self.detection_manager.put(result)
                except:
                    pass
        
        # Thread để chuyển kết quả depth
        def depth_worker():
            while self._run_event.is_set():
                try:
                    result = self._depth_result_queue.get(timeout=0.5)
                    if result is not None:
                        self.depth_manager.put(result)
                except:
                    pass
        
        # Khởi động các thread
        threading.Thread(target=detection_worker, daemon=True).start()
        threading.Thread(target=depth_worker, daemon=True).start()
    
    def stop(self):
        """Dừng tất cả các process."""
        print("Đang dừng tất cả các process...")
        self._run_event.clear()
        
        # ⭐ DISCONNECT PLC FIRST ⭐
        self.disconnect_plc()
        
        # Dừng và join từng process
        if self._camera_process:
            self._camera_process.join(timeout=2)
            print(f"Camera Process đã dừng, exit code: {self._camera_process.exitcode}")
            self._camera_process = None
            
        if self._yolo_process:
            self._yolo_process.join(timeout=2)
            print(f"YOLO Process đã dừng, exit code: {self._yolo_process.exitcode}")
            self._yolo_process = None
            
        if self._depth_process:
            self._depth_process.join(timeout=2)
            print(f"Depth Process đã dừng, exit code: {self._depth_process.exitcode}")
            self._depth_process = None
    
    def get_detection(self, timeout: float = 0.1):
        """
        Lấy kết quả detection mới nhất.
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            tuple: (frame, detections) hoặc None nếu không có kết quả mới
        """
        self._check_errors()
        return self.detection_manager.get(timeout=timeout)
    
    def get_depth(self, timeout: float = 0.1):
        """
        Lấy kết quả depth mới nhất.
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            tuple: (frame, depth_results) hoặc None nếu không có kết quả mới
        """
        self._check_errors()
        return self.depth_manager.get(timeout=timeout)
    
    def get_latest_detection(self, timeout: float = 0.1):
        """
        Lấy kết quả detection mới nhất, bỏ qua các kết quả cũ.
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            tuple: (frame, detections) hoặc None nếu không có kết quả mới
        """
        self._check_errors()
        return self.detection_manager.get_latest(timeout=timeout)
    
    def get_latest_depth(self, timeout: float = 0.1):
        """
        Lấy kết quả depth mới nhất, bỏ qua các kết quả cũ.
        
        Args:
            timeout: Thời gian tối đa chờ đợi (giây)
            
        Returns:
            tuple: (frame, depth_results) hoặc None nếu không có kết quả mới
        """
        self._check_errors()
        return self.depth_manager.get_latest(timeout=timeout)
    
    def _check_errors(self):
        """Kiểm tra và lưu lỗi từ các process con."""
        while not self._error_queue.empty():
            try:
                error = self._error_queue.get_nowait()
                self._errors.append(error)
                # print(f"Lỗi từ process con: {error}")
            except:
                break
    
    @property
    def errors(self):
        """Trả về danh sách lỗi từ các process con."""
        self._check_errors()
        return self._errors
    
    @property
    def is_running(self):
        """Kiểm tra xem pipeline có đang chạy không."""
        return (self._run_event.is_set() and 
                self._camera_process is not None and self._camera_process.is_alive() and
                self._yolo_process is not None and self._yolo_process.is_alive() and
                self._depth_process is not None and self._depth_process.is_alive())
    
    def get_stats(self):
        """
        Lấy thống kê về số lượng khung hình, detections và depth xử lý được.
        
        Returns:
            Dict: Thống kê về số lượng
                - 'frames': Số khung hình đã xử lý
                - 'detections': Số detections đã xử lý
                - 'depths': Số depth đã xử lý
        """
        return {
            'frames': self.frame_counter.value,
            'detections': self.detection_counter.value,
            'depths': self.depth_counter.value
        }
    
    def get_region_sequencer(self):
        """
        Lấy region sequencer proxy để sử dụng keyboard controls.
        
        Returns:
            SequencerProxy object để interact với RegionSequencer trong worker process
        """
        if not hasattr(self, '_sequencer_proxy'):
            self._sequencer_proxy = SequencerProxy(self)
        
        return self._sequencer_proxy
    
    # ⭐ PLC INTEGRATION METHODS ⭐
    def connect_plc(self) -> bool:
        """
        Kết nối PLC nếu PLC integration được bật.
        
        Returns:
            bool: True nếu kết nối thành công hoặc PLC không được bật
        """
        if not self.enable_plc:
            print(f"[Pipeline] ℹ️ PLC not enabled, skipping connection")
            return True  # Thành công mặc định nếu không sử dụng PLC
        
        if not self.plc_integration:
            print(f"[Pipeline] ❌ PLC integration not initialized!")
            return False
        
        print(f"[Pipeline] 🔌 Attempting to connect to PLC at {self.plc_ip}...")
        try:
            connected = self.plc_integration.connect_plc()
            if connected:
                print(f"[Pipeline] ✅ PLC connection successful!")
            else:
                print(f"[Pipeline] ❌ PLC connection failed!")
            return connected
        except Exception as e:
            print(f"[Pipeline] ❌ PLC connection error: {e}")
            return False
    
    def disconnect_plc(self):
        """Ngắt kết nối PLC."""
        if self.plc_integration:
            self.plc_integration.disconnect_plc()
    
    def get_plc_integration(self):
        """
        Lấy PLC integration object.
        
        Returns:
            RegionDivisionPLCIntegration hoặc None
        """
        return self.plc_integration
    
    def send_region_to_plc(self, detections: dict) -> bool:
        """
        Gửi regions từ detections vào PLC.
        
        Args:
            detections: Detection results từ pipeline
            
        Returns:
            bool: True nếu gửi thành công
        """
        if not self.plc_integration:
            print(f"[Pipeline] PLC integration not available")
            return False
        
        # Sử dụng existing method từ RegionDivisionPLCIntegration
        regions_data, send_success = self.plc_integration.process_detection_and_send_to_plc(
            detections, layer=1
        )
        
        return send_success


class SequencerProxy:
    """
    Proxy class để interact với RegionSequencer trong worker process từ main process.
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._last_status = None
    
    def _get_current_status(self):
        """Lấy status hiện tại từ detection results."""
        try:
            # ⭐ TỐI ƯU TIMEOUT ⭐ - Thử nhiều lần với timeout ngắn
            for attempt in range(3):  # 3 attempts
                detection_result = self.pipeline.get_latest_detection(timeout=0.3)
                if detection_result:
                    frame, detections = detection_result
                    
                    # ⭐ DEBUG: Log để trace data reception ⭐
                    sequencer_available = detections.get('sequencer_available', False)
                    has_status = 'sequencer_status' in detections
                    status = detections.get('sequencer_status')
                    
                    print(f"[SequencerProxy DEBUG] _get_current_status (attempt {attempt+1}):")
                    print(f"  sequencer_available: {sequencer_available}")
                    print(f"  has_status: {has_status}")
                    print(f"  status: {status is not None}")
                    
                    if sequencer_available and status:
                        self._last_status = status
                        print(f"  ✅ Status updated: {status['status']}")
                        return self._last_status
                    else:
                        print(f"  ❌ No valid status received")
                        if attempt < 2:  # Không log ở attempt cuối
                            print(f"  🔄 Retrying...")
                else:
                    print(f"[SequencerProxy DEBUG] Attempt {attempt+1}: No detection result")
        except Exception as e:
            print(f"[SequencerProxy DEBUG] Error: {e}")
        return self._last_status
    
    def get_queue_status(self):
        """Lấy queue status từ worker process."""
        status = self._get_current_status()
        if status:
            return status
        
        # Fallback status nếu không có thông tin
        return {
            'status': 'UNKNOWN',
            'current_pallet': None,
            'current_index': 0,
            'total_regions': 0,
            'completed_count': 0,
            'remaining_count': 0,
            'progress': '0/0',
            'completed_regions': [],
            'remaining_regions': [],
            'sequence': [1, 3, 2]
        }
    
    def get_next_region(self):
        """
        Trigger next region - TỰ ĐỘNG GỬI VÀO PLC với FALLBACK SENDING.
        """
        print("   🚀 Manual trigger: Get next region...")
        
        # ⭐ STEP 1: CHECK SEQUENCER STATUS ⭐
        current_status = self._get_current_status()
        if current_status:
            print(f"   📊 Current sequencer status: {current_status['status']}")
            print(f"   📊 Progress: {current_status['progress']}")
        else:
            print(f"   ⚠️ Sequencer status not available, using fallback sending...")
        
        # ⭐ STEP 2: DIRECT PLC SENDING (ALWAYS WORKS) ⭐
        try:
            # Lấy latest detection để gửi vào PLC
            detection_result = self.pipeline.get_latest_detection(timeout=1.0)
            if detection_result:
                frame, detections = detection_result
                
                # ⭐ CHECK PLC INTEGRATION AVAILABILITY ⭐
                plc_integration = self.pipeline.get_plc_integration()
                if not plc_integration:
                    print("   ❌ PLC integration not available!")
                    return None
                
                if not plc_integration.plc_connected:
                    print("   🔌 PLC not connected, attempting to connect...")
                    connected = plc_integration.connect_plc()
                    if not connected:
                        print("   ❌ Failed to connect to PLC!")
                        return None
                    else:
                        print("   ✅ PLC connected successfully!")
                
                # ⭐ SEND TO PLC USING DIRECT METHOD ⭐
                print("   📤 Sending regions to PLC...")
                regions_data, send_success = plc_integration.process_detection_and_send_to_plc(detections, layer=1)
                
                if send_success:
                    print("   ✅ Successfully sent regions to PLC!")
                    print(f"   📋 Processed {len(regions_data)} regions")
                    
                    # ⭐ HIỂN THỊ BAG PALLET TRACKING STATUS ⭐
                    bag_status = plc_integration.get_bag_pallet_status()
                    print(f"   📦 BAG PALLET TRACKING:")
                    print(f"      bag_pallet_1 = {bag_status['bag_pallet_1']}")
                    print(f"      bag_pallet_2 = {bag_status['bag_pallet_2']}")
                    print(f"      Active regions: {bag_status['active_regions_count']}")
                    
                    # ⭐ SHOW DETAILED REGION DATA ⭐
                    for region_name, region_data in bag_status['current_regions'].items():
                        if region_data:
                            pallet_id = region_data['pallet_id']
                            region_id = region_data['region_id']
                            robot_coords = region_data['robot_coords']
                            print(f"      {region_name}: P{pallet_id}R{region_id} → Px={robot_coords['px']:.2f}, Py={robot_coords['py']:.2f}")
                    
                    # ⭐ READ BACK FROM PLC TO VERIFY ⭐
                    print("   🔍 Reading back from PLC to verify...")
                    plc_data = plc_integration.read_regions_from_plc()
                    if plc_data:
                        print("   📊 PLC Memory Content:")
                        for region_name, data in plc_data.items():
                            print(f"      {region_name}: Px={data['px']:.2f} (DB26.{data['px_offset']}), Py={data['py']:.2f} (DB26.{data['py_offset']})")
                else:
                    print("   ❌ Failed to send regions to PLC")
                    if regions_data:
                        print(f"   📋 Regions were processed ({len(regions_data)}) but PLC sending failed")
            else:
                print("   ⚠️ No detection data available")
                
        except Exception as e:
            print(f"   ❌ Error in PLC sending process: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def mark_region_completed(self):
        """
        Mark region completed (chỉ để tương thích với keyboard controls).
        Thực tế cần implement command communication với worker process.
        """
        print("   ℹ️ Manual completion chưa được implement")
        print("   🔄 Sequential logic hiện đang auto-complete trong demo mode")
        return False
    
    def reset_sequence(self):
        """
        Reset sequence (chỉ để tương thích với keyboard controls).
        Thực tế cần implement command communication với worker process.
        """
        print("   ℹ️ Manual reset chưa được implement")
        print("   🔄 Restart chương trình để reset sequence")
        return False
    
    def is_available(self):
        """Kiểm tra xem sequencer có available không."""
        status = self._get_current_status()
        return status is not None


# Ví dụ sử dụng
if __name__ == "__main__":
    # Đây là ví dụ sử dụng, cần import các module thực tế
    from detection.camera import CameraInterface
    from detection.utils.depth import DepthEstimator
    import cv2
    import time
    
    # Các factory functions
    def create_camera():
        camera = CameraInterface(camera_index=0)
        camera.initialize()
        return camera
    
    def create_yolo():
        return YOLOInference(model_path="best.pt", conf=0.25)
    
    def create_depth():
        return DepthEstimator()
    
    # Tạo pipeline
    pipeline = ProcessingPipeline(
        camera_factory=create_camera,
        yolo_factory=create_yolo,
        depth_factory=create_depth
    )
    
    # Khởi động pipeline
    if pipeline.start(timeout=60.0):
        print("Pipeline đã khởi động thành công!")
        
        try:
            # Lặp và xử lý kết quả
            for _ in range(100):  # Xử lý 100 khung hình
                # Lấy kết quả detection mới nhất
                detection_result = pipeline.get_latest_detection()
                if detection_result:
                    frame, detections = detection_result
                    # Xử lý kết quả detection
                    print(f"Đã phát hiện {len(detections.get('bounding_boxes', []))} đối tượng")
                
                # Lấy kết quả depth mới nhất
                depth_result = pipeline.get_latest_depth()
                if depth_result:
                    frame, depth_results = depth_result
                    # Xử lý kết quả depth
                    print(f"Đã ước tính độ sâu cho {len(depth_results)} đối tượng")
                
                # Hiển thị thống kê
                stats = pipeline.get_stats()
                print(f"Stats: Frames={stats['frames']}, Detections={stats['detections']}, Depths={stats['depths']}")
                
                time.sleep(0.1)  # Đợi một chút
                
        except KeyboardInterrupt:
            print("Đã nhận tín hiệu ngắt từ bàn phím")
        finally:
            # Dừng pipeline
            pipeline.stop()
            print("Pipeline đã dừng")
    else:
        print("Không thể khởi động pipeline!")
        # Kiểm tra lỗi
        for error in pipeline.errors:
            print(f"Lỗi: {error}") 